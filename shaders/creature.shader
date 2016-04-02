[Vertex shader]
#version 330

layout(location = 0)in vec2 position;
layout(location = 1)in vec2 texture_coordinate;
layout(location = 2)in vec4 creature_position;
layout(location = 3)in vec4 creature_texture;
layout(location = 4)in vec4 beat_swirl;

uniform mat4 MVP;
uniform float t;

out vec2 uv;
out vec4 lhsa;
out vec2 whirl;
//out float texture_ind;

void main()
{
    float theta = creature_position.z;
    mat2 z_rot = mat2(cos(theta), -sin(theta),
                      sin(theta), cos(theta));
    float t_offset = beat_swirl.x;
    float beat_frequency = beat_swirl.y;
    float scale = creature_position.w*(1 + 0.1*sin(beat_frequency*t + t_offset));
    gl_Position = MVP * vec4(z_rot*scale*position + creature_position.xy, 0.0, 1.0);
    uv = texture_coordinate;
    lhsa = creature_texture;
    whirl = beat_swirl.zw;
    //texture_ind = creature_texture.x;
}


[Fragment shader]
#version 330

in vec4 beat_swirl;

in vec2 uv;
in vec4 lhsa;
in vec2 whirl;
//in float texture_ind;

//uniform sampler2D image;
uniform sampler2DArray image;
uniform float t;

vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

vec2 swirl(vec2 uv)
{
    vec2 texSize = vec2(1.0, 1.0);
    float radius = whirl.x;
    float angle = sin(t)*0.1;

    vec2 center = texSize * 0.5;
    vec2 tc = (uv * texSize) - center;
    float dist = abs(length(tc) - radius);
    if( dist < 0.1)
    {
        float percent = 1.0 - (dist / 0.1);
        //float percent = (radius - dist) / radius;
        float theta = percent * percent * angle * 8.0;
        float s = sin(theta);
        float c = cos(theta);
        tc = vec2(dot(tc, vec2(c, -s)), dot(tc, vec2(s, c)));
    }
    tc += center;
    return tc / texSize;
}

void main()
{
    vec2 swirl_uv = swirl(uv);
    vec4 hsva = texture(image, vec3(swirl_uv, lhsa.x));
    hsva.x = mod(hsva.x + lhsa.y, 1.0);
    hsva.y = hsva.y * lhsa.z;
    hsva.a = hsva.a * lhsa.a;
    vec3 rgb = hsv2rgb(hsva.xyz);
    gl_FragColor = vec4(rgb, hsva.a);
}



