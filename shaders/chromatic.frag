#version 330 core
// ─── Built-in uniforms (always injected by pipeline) ───
uniform sampler2D uTex;
uniform vec2      uResolution;
uniform float     uTime;
uniform int       uFrame;
// Crowd
uniform float     uEnergy;
uniform float     uPulse;
// ─── Per-pass tunables ───
uniform float     uStrength;   // chromatic split amount (default 0.008)
uniform float     uBarrel;     // barrel distortion (default 0.15)

in  vec2 vUV;
out vec4 fragColor;

vec2 barrel(vec2 uv, float k) {
    vec2 c = uv - 0.5;
    float r2 = dot(c, c);
    return 0.5 + c * (1.0 + k * r2);
}

void main() {
    // Modulate strength with crowd energy
    float str    = uStrength * (1.0 + uEnergy * 2.0);
    float barrel = uBarrel   * (1.0 + uPulse  * 0.5);

    vec2 center  = vec2(0.5);
    vec2 dir     = normalize(vUV - center + 0.001);

    vec2 uvR = barrel(vUV + dir * str,        barrel);
    vec2 uvG = barrel(vUV,                    barrel);
    vec2 uvB = barrel(vUV - dir * str,        barrel);

    float r = texture(uTex, uvR).r;
    float g = texture(uTex, uvG).g;
    float b = texture(uTex, uvB).b;
    float a = texture(uTex, uvG).a;

    fragColor = vec4(r, g, b, a);
}
