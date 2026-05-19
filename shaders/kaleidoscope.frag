#version 330 core
uniform sampler2D uTex;
uniform float     uTime;
uniform float     uEnergy;
// tunables
uniform float     uSegments;   // mirror count (default 6.0)
uniform float     uRotation;   // base rotation speed rad/sec (default 0.2)

in  vec2 vUV;
out vec4 fragColor;

#define PI 3.14159265358979

void main() {
    vec2 uv  = vUV - 0.5;
    float r  = length(uv);
    float a  = atan(uv.y, uv.x);

    float seg = PI / uSegments;
    a = mod(a + uTime * uRotation * (1.0 + uEnergy), 2.0 * seg);
    if (a > seg) a = 2.0 * seg - a;

    vec2 lookup = vec2(r * cos(a), r * sin(a)) + 0.5;
    lookup = clamp(lookup, 0.0, 1.0);

    fragColor = texture(uTex, lookup);
}
