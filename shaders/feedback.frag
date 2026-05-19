#version 330 core
uniform sampler2D uTex;         // current source
uniform sampler2D uFeedbackTex; // previous frame (if bound)
uniform float     uTime;
uniform float     uEnergy;
uniform float     uPulse;
// tunables
uniform float     uDecay;   // trail persistence 0-1 (default 0.85)
uniform float     uZoom;    // feedback zoom (default 0.995)
uniform float     uSpin;    // rotation per frame in degrees (default 0.3)

in  vec2 vUV;
out vec4 fragColor;

mat2 rot2(float a){ float c=cos(a),s=sin(a); return mat2(c,-s,s,c); }

void main() {
    vec2 center = vec2(0.5);

    // Feedback UV: slight zoom + rotation
    float angle = uSpin * 3.14159 / 180.0 * (1.0 + uEnergy);
    vec2 fbUV   = center + rot2(angle) * (vUV - center) * uZoom;

    vec4 current  = texture(uTex, vUV);
    vec4 feedback = texture(uFeedbackTex, fbUV);

    float decay = uDecay * (1.0 - uPulse * 0.2);  // beat cuts decay
    vec4  trail = feedback * decay;

    fragColor = max(current, trail);
}
