[Vertex shader]
#version 330

layout(location = 0)in vec2 position;
layout(location = 1)in vec2 texture_coordinate;
layout(location = 2)in vec4 animation_position;
//layout(location = 3)in vec4 animation_texture;
layout(location = 3)in vec4 animation_param;

uniform mat4 MVP;
uniform float t;

out vec2 uv;
//out vec4 lhsa;
out float current_frame;

void main()
{
    float theta = animation_position.z;
    mat2 z_rot = mat2(cos(theta), -sin(theta),
                      sin(theta), cos(theta));
    float scale = animation_position.w;
    //gl_Position = MVP * vec4(scale*position, 0.0, 1.0);
    gl_Position = MVP * vec4(z_rot*scale*position + animation_position.xy, 0.0, 1.0);
    uv = texture_coordinate;
    //lhsa = animation_texture;

    float num_frames = animation_param.x;
    float start_t = animation_param.y;
    float frame_t = animation_param.z;
    current_frame = floor(mod((t-start_t)/frame_t, num_frames));
    //whirl = beat_swirl.zw;
    //texture_ind = creature_texture.x;
}


[Fragment shader]
#version 330

in vec2 uv;
//in vec4 lhsa;
in float current_frame;
//in float texture_ind;

//uniform sampler2D image;
uniform sampler2DArray image;
uniform float t;

void main()
{

    //vec2 swirl_uv = swirl(uv);
    //vec4 hsva = texture(image, vec3(swirl_uv, lhsa.x));
    //hsva.x = mod(hsva.x + lhsa.y, 1.0);
    //hsva.y = hsva.y * lhsa.z;
    //hsva.a = hsva.a * lhsa.a;
    //vec3 rgb = hsv2rgb(hsva.xyz);
    gl_FragColor = texture(image, vec3(uv, current_frame));
    //gl_FragColor = vec4(rgb, hsva.a);
}



