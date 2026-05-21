#version 330 core
uniform sampler2D uTex;
uniform float     uTime;
uniform float     uEnergy;
uniform float     uSentiment;
// tunables
uniform float     uSaturation;  // 0=grey 1=normal 2=hyper (default 1.2)
uniform float     uContrast;    // (default 1.1)
uniform float     uHueShift;    // degrees/sec (default 0.0)

in  vec2 vUV;
out vec4 fragColor;

/*
 * The main program
 */
void main() {
    // Calculate the pixel color based on the mouse position
    vec3 pixel_color;

    // Apply the edge detection kernel
    pixel_color += -1.0 * texture2D(uTex, vUV + vec2(-1, -1)).rgb;
    pixel_color += -1.0 * texture2D(uTex, vUV + vec2(-1, 0)).rgb;
    pixel_color += -1.0 * texture2D(uTex, vUV + vec2(-1, 1)).rgb;
    pixel_color += -1.0 * texture2D(uTex, vUV + vec2(0, -1)).rgb;
    pixel_color += 8.0 * texture2D(uTex, vUV + vec2(0, 0)).rgb;
    pixel_color += -1.0 * texture2D(uTex, vUV + vec2(0, 1)).rgb;
    pixel_color += -1.0 * texture2D(uTex, vUV + vec2(1, -1)).rgb;
    pixel_color += -1.0 * texture2D(uTex, vUV + vec2(1, 0)).rgb;
    pixel_color += -1.0 * texture2D(uTex, vUV + vec2(1, 1)).rgb;

    // Use the most extreme color value
    float min_value = min(pixel_color.r, min(pixel_color.g, pixel_color.b));
    float max_value = max(pixel_color.r, max(pixel_color.g, pixel_color.b));

    if (abs(min_value) > abs(max_value)) {
        pixel_color = vec3(min_value);
    } else {
        pixel_color = vec3(max_value);
    }

    // Rescale the pixel color using the mouse y position
    pixel_color = 0.5 + 1 * pixel_color;

    // Fragment shader output
    fragColor = vec4(pixel_color, 1.0);
}

