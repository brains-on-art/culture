[Vertex shader]
#version 330

layout(location = 0)in vec2 position;
layout(location = 1)in vec2 texture_coordinate;
layout(location = 2)in vec3 creature_position;
layout(location = 3)in vec4 creature_texture;

uniform mat4 MVP;

out vec2 uv;
out vec4 lhsa;
//out float texture_ind;

void main()
{
    float theta = creature_position.z;
    mat2 z_rot = mat2(cos(theta), -sin(theta),
                      sin(theta), cos(theta));
    gl_Position = MVP * vec4(z_rot*position + creature_position.xy, 0.0, 1.0);
    uv = texture_coordinate;
    lhsa = creature_texture;
    //texture_ind = creature_texture.x;
}


[Fragment shader]
#version 330

in vec2 uv;
in vec4 lhsa;
//in float texture_ind;

//uniform sampler2D image;
uniform sampler2DArray image;

vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main()
{
    vec4 hsva = texture(image, vec3(uv, lhsa.x));
    hsva.x = mod(hsva.x + lhsa.y, 1.0);
    vec3 rgb = hsv2rgb(hsva.xyz);
    gl_FragColor = vec4(rgb, hsva.a);
}

