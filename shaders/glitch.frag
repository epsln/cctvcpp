#version 330 core
uniform sampler2D uTex;
uniform vec2      uResolution;
uniform float     uTime;
uniform int       uFrame;
uniform float     uEnergy;
uniform float     uPulse;
// tunables
uniform float     uAmount;    // glitch intensity  (default 0.05)
uniform float     uSpeed;     // glitch speed      (default 4.0)

in  vec2 vUV;
out vec4 fragColor;

float rand(float x) { return fract(sin(x * 127.1) * 43758.5453); }
float rand2(vec2 x) { return fract(sin(dot(x, vec2(12.9898,78.233))) * 43758.5453); }

void main() {
    float t     = floor(uTime * uSpeed) / uSpeed;
    float amt   = uAmount * (1.0 + uEnergy * 3.0);

    // Horizontal band glitch
    float band   = floor(vUV.y * 20.0 + t * 7.0);
    float jitter = (rand(band + t) * 2.0 - 1.0) * amt;

    // Random strong glitch line
    float glitch_line = rand(floor(uTime * 13.0)) * 0.95;
    float near_line   = step(abs(vUV.y - glitch_line), 0.01);
    jitter += near_line * (rand(uTime) * 2.0 - 1.0) * amt * 5.0;

    vec2 uvG = vUV + vec2(jitter, 0.0);
    vec2 uvR = uvG + vec2(amt * rand(t + 0.3) * 0.5, 0.0);
    vec2 uvB = uvG - vec2(amt * rand(t + 0.7) * 0.5, 0.0);

    float r = texture(uTex, clamp(uvR, 0.0, 1.0)).r;
    float g = texture(uTex, clamp(uvG, 0.0, 1.0)).g;
    float b = texture(uTex, clamp(uvB, 0.0, 1.0)).b;

    // Digital scanline artifact
    float scan = sin(vUV.y * uResolution.y * 1.5) * 0.03 * uEnergy;

    fragColor = vec4(r + scan, g, b - scan, 1.0);
}
