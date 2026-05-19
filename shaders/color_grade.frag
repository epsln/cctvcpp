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

vec3 rgb2hsl(vec3 c) {
    float mx = max(c.r,max(c.g,c.b)), mn = min(c.r,min(c.g,c.b));
    float h,s,l=(mx+mn)/2.;
    if(mx==mn){h=s=0.;}else{
        float d=mx-mn;
        s=l>.5?d/(2.-mx-mn):d/(mx+mn);
        if(mx==c.r)      h=(c.g-c.b)/d+(c.g<c.b?6.:0.);
        else if(mx==c.g) h=(c.b-c.r)/d+2.;
        else             h=(c.r-c.g)/d+4.;
        h/=6.;
    }
    return vec3(h,s,l);
}

float hue2rgb(float p,float q,float t){
    if(t<0.)t+=1.;if(t>1.)t-=1.;
    if(t<1./6.)return p+(q-p)*6.*t;
    if(t<1./2.)return q;
    if(t<2./3.)return p+(q-p)*(2./3.-t)*6.;
    return p;
}

vec3 hsl2rgb(vec3 c){
    if(c.y==0.) return vec3(c.z);
    float q=c.z<.5?c.z*(1.+c.y):c.z+c.y-c.z*c.y;
    float p=2.*c.z-q;
    return vec3(hue2rgb(p,q,c.x+1./3.),hue2rgb(p,q,c.x),hue2rgb(p,q,c.x-1./3.));
}

void main() {
    vec4 col = texture(uTex, vUV);
    vec3 hsl = rgb2hsl(col.rgb);

    // Hue shift driven by time + crowd sentiment
    float shift = uHueShift / 360.0 * uTime + uSentiment * 0.1;
    hsl.x = fract(hsl.x + shift);

    // Saturation boost with energy
    hsl.y = clamp(hsl.y * uSaturation * (1.0 + uEnergy * 0.5), 0.0, 1.0);

    vec3 rgb = hsl2rgb(hsl);

    // Contrast
    rgb = (rgb - 0.5) * uContrast + 0.5;

    fragColor = vec4(clamp(rgb, 0.0, 1.0), col.a);
}
