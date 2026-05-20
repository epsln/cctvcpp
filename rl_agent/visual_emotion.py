"""
visual_emotion.py
─────────────────
Maps rendered VJ frames → (arousal, valence) using CLIP zero-shot similarity.

No labels, no fine-tuning.  We define a set of text anchors that span the
arousal/valence plane, embed them once at startup, then for each frame compute
cosine similarities between the image embedding and each anchor.  The final
(arousal, valence) coordinates are the similarity-weighted centroid of the
anchor positions in that 2-D space.

Anchor design (Russell circumplex model):
  High arousal  / High valence  → "euphoric intense colorful explosion of light"
  High arousal  / Low valence   → "aggressive chaotic dark violent distortion"
  Low arousal   / High valence  → "peaceful serene calm gentle soft glow"
  Low arousal   / Low valence   → "melancholic dark still quiet emptiness"

We use 8 anchors (4 quadrants × 2 descriptions each) for robustness.

Frame capture:
  Reads the VJ engine window via mss (screen capture) or from a shared memory
  buffer written by the engine.  Falls back to reading vj_frame.png if neither
  is available (engine can write a frame every N ticks).

Usage:
  estimator = VisualEmotionEstimator()
  arousal, valence = estimator.estimate(pil_image)
  # or
  arousal, valence = estimator.capture_and_estimate(window_title="VJ Engine")
"""

from __future__ import annotations
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
#  Arousal/valence anchor definitions  (Russell circumplex)
#  Each entry: (arousal_target, valence_target, text_prompt)
# ─────────────────────────────────────────────────────────────────────────────

ANCHORS = [
    # High arousal, high valence  (excited, joyful)
    (1.0,  1.0, "euphoric intense bright colorful explosion of light and energy"),
    (0.9,  0.8, "vibrant electric pulsing neon colors rapid movement excitement"),
    # High arousal, low valence  (tense, angry)
    (1.0, -1.0, "aggressive chaotic dark violent glitch distortion noise"),
    (0.8, -0.8, "harsh abrasive flickering red and black darkness"),
    # Low arousal, high valence  (calm, content)
    (-1.0,  1.0, "peaceful serene calm gentle soft pastel glow slow drift"),
    (-0.8,  0.9, "tranquil dreamy smooth flowing warm soft light"),
    # Low arousal, low valence  (sad, depressed)
    (-1.0, -1.0, "melancholic dark still quiet emptiness cold blue grey"),
    (-0.9, -0.7, "lonely desolate fading dim muted colors slow decay"),
    # Centre (neutral)
    (0.0,  0.0, "neutral balanced moderate visual pattern"),
]

# Pre-extract anchor coordinates and texts
ANCHOR_COORDS = np.array([[a, v] for a, v, _ in ANCHORS], dtype=np.float32)  # (9, 2)
ANCHOR_TEXTS  = [t for _, _, t in ANCHORS]


# ─────────────────────────────────────────────────────────────────────────────
#  VisualEmotionEstimator
# ─────────────────────────────────────────────────────────────────────────────

class VisualEmotionEstimator:
    CLIP_MODEL = "openai/clip-vit-base-patch32"   # ~350 MB, CPU-friendly

    def __init__(self, device: str = "cpu", update_interval_sec: float = 1.0):
        self.device = device
        self.update_interval = update_interval_sec

        self._model      = None
        self._tokenizer  = None
        self._image_proc = None
        self._loaded     = False

        # Cached anchor embeddings (computed once after load)
        self._anchor_embeds: Optional[torch.Tensor] = None   # (N, D)

        # Last result cache
        self._last_arousal: float = 0.0
        self._last_valence: float = 0.0
        self._last_update:  float = 0.0

        self._load()

    def _load(self):
        try:
            from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor
            print(f"[VisualEmotion] Loading {self.CLIP_MODEL} ...")

            # Load tokenizer and image processor explicitly by class — avoids the
            # AutoProcessor image_processor_type discovery bug in transformers >= 4.x
            self._tokenizer = CLIPTokenizer.from_pretrained(self.CLIP_MODEL)
            self._image_proc = CLIPImageProcessor.from_pretrained(self.CLIP_MODEL)
            self._model = CLIPModel.from_pretrained(self.CLIP_MODEL).to(self.device)
            self._model.eval()
            self._anchor_embeds = self._embed_texts(ANCHOR_TEXTS)
            print(f"[VisualEmotion] Ready. {len(ANCHOR_TEXTS)} anchors embedded.")
            self._loaded = True
        except Exception as e:
            print(f"[VisualEmotion] CLIP load failed: {e}. Using heuristic fallback.")

    @torch.no_grad()
    def _embed_texts(self, texts: list[str]) -> torch.Tensor:
        inputs = self._tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        embeds = self._model.get_text_features(**inputs).pooler_output
        return F.normalize(embeds, dim=-1)   # (N, D)

    @torch.no_grad()
    def _embed_image(self, pil_image) -> torch.Tensor:
        inputs = self._image_proc(images=pil_image, return_tensors="pt").to(self.device)
        embed  = self._model.get_image_features(**inputs).pooler_output
        return F.normalize(embed, dim=-1)    # (1, D)

    # ── Main interface ───────────────────────────────────────────────────────

    def estimate(self, pil_image) -> tuple[float, float]:
        """
        Returns (arousal, valence) in [-1, 1].
        Uses cached result if called faster than update_interval.
        """
        now = time.time()
        if (now - self._last_update) < self.update_interval:
            return self._last_arousal, self._last_valence

        if not self._loaded:
            return self._heuristic(pil_image)

        try:
            img_embed = self._embed_image(pil_image)   # (1, D)
            # Cosine similarity against all anchors
            sims = (img_embed @ self._anchor_embeds.T).squeeze(0)  # (N,)
            # Softmax to get a probability distribution over anchors
            weights = torch.softmax(sims * 10.0, dim=0).cpu().numpy()  # sharpen
            # Weighted centroid in (arousal, valence) space
            coords = (weights[:, None] * ANCHOR_COORDS).sum(axis=0)   # (2,)
            arousal = float(np.clip(coords[0], -1.0, 1.0))
            valence = float(np.clip(coords[1], -1.0, 1.0))
        except Exception as e:
            print(f"[VisualEmotion] estimate error: {e}")
            arousal, valence = self._last_arousal, self._last_valence

        self._last_arousal = arousal
        self._last_valence = valence
        self._last_update  = now
        return arousal, valence

    def _heuristic(self, pil_image) -> tuple[float, float]:
        """
        Pure numpy fallback.
        Arousal ← image entropy + edge density (busy/active = high arousal)
        Valence ← mean hue + brightness (warm/bright = positive)
        """
        import numpy as np
        img = np.array(pil_image).astype(np.float32) / 255.0
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)

        # Arousal: local variance proxy
        gray  = img.mean(axis=-1)
        local_var = float(np.std(gray))
        arousal = float(np.clip(local_var * 6.0 - 1.0, -1.0, 1.0))

        # Valence: brightness and warm hue bias
        brightness = float(img.mean())
        r_bias = float(img[:,:,0].mean() - img[:,:,2].mean())  # warm=positive
        valence = float(np.clip((brightness - 0.5) * 2.0 + r_bias, -1.0, 1.0))

        self._last_arousal = arousal
        self._last_valence = valence
        self._last_update  = time.time()
        return arousal, valence

    # ── Frame capture helpers ────────────────────────────────────────────────

    def capture_window(self, window_x: int, window_y: int,
                       window_w: int, window_h: int):
        """
        Capture a region of the screen via mss.
        Returns PIL Image or None.
        """
        try:
            import mss
            from PIL import Image
            with mss.mss() as sct:
                mon = {"left": window_x, "top": window_y,
                       "width": window_w, "height": window_h}
                shot = sct.grab(mon)
                return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        except Exception as e:
            return None

    def capture_from_file(self, path: str = "vj_frame.ppm"):
        """Read a frame dumped by the C++ engine (PPM or PNG)."""
        try:
            from PIL import Image
            return Image.open(path).convert("RGB")
        except Exception:
            return None

    def capture_and_estimate(
        self,
        window_x: int = 0, window_y: int = 0,
        window_w: int = 1280, window_h: int = 720,
        frame_file: str = "vj_frame.png",
    ) -> tuple[float, float]:
        img = self.capture_window(window_x, window_y, window_w, window_h)
        if img is None:
            img = self.capture_from_file(frame_file)
        if img is None:
            return self._last_arousal, self._last_valence
        return self.estimate(img)
