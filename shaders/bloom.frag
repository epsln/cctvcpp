#version 330 core
uniform sampler2D uTex;
uniform vec2      uResolution;
uniform float     uEnergy;
// tunables
uniform float     uThreshold;  // luminance threshold for bloom (default 0.6)
uniform float     uIntensity;  // glow strength (default 1.5)
uniform int       uRadius;     // blur radius in pixels (default 8)

in  vec2 vUV;
out vec4 fragColor;

float luma(vec3 c){ return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

void main() {
    vec2 texel = 1.0 / uResolution;
    vec4 base  = texture(uTex, vUV);

    // Simple box blur for bright areas only
    vec3 bloom = vec3(0.0);
    float total = 0.0;
    int r = max(1, uRadius);

    for (int x = -r; x <= r; x++) {
        for (int y = -r; y <= r; y++) {
            vec2 offset = vec2(float(x), float(y)) * texel;
            vec3 s = texture(uTex, vUV + offset).rgb;
            float bright = max(0.0, luma(s) - uThreshold);
            bloom  += s * bright;
            total  += bright;
        }
    }
    if (total > 0.0) bloom /= total;

    float intensity = uIntensity * (1.0 + uEnergy * 2.0);
    fragColor = vec4(base.rgb + bloom * intensity, base.a);
}
