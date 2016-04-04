[Vertex shader]
#version 330

layout(location = 0)in vec2 position;
layout(location = 1)in vec2 texture_coordinate;
layout(location = 2)in vec4 animation_position;
layout(location = 3)in vec4 animation_param1;
layout(location = 4)in vec4 animation_param2;

uniform mat4 MVP;
uniform float t;

out vec2 uv;
out float current_frame;

void main()
{
    float theta = animation_position.z;
    mat2 z_rot = mat2(cos(theta), -sin(theta),
                      sin(theta), cos(theta));
    float scale = animation_position.w;
    gl_Position = MVP * vec4(z_rot*scale*position + animation_position.xy, 0.0, 1.0);
    uv = texture_coordinate;

    float start_frame = animation_param1.x;
    float end_frame = animation_param1.y;
    float start_time = animation_param1.z;
    float loop_time = animation_param1.w;

    float num_loops = animation_param2.x;
    float alpha_frame = animation_param2.y;
    float omega_frame = animation_param2.z;

    float num_frames = end_frame - start_frame + 1;
    float anim_t = (t - start_time);
    float anim_loop = floor(anim_t/loop_time);

    if (anim_t < 0.0) {
        current_frame = alpha_frame;
    } else if (anim_loop >= num_loops) {
        current_frame = omega_frame;
    } else {
        current_frame = start_frame + floor(mod(anim_t, loop_time)*(num_frames/loop_time));
    }
}


[Fragment shader]
#version 330

in vec2 uv;
in float current_frame;

uniform sampler2DArray image;
uniform float t;

void main()
{

    vec4 rgba = texture(image, vec3(uv, current_frame));
    rgba.w *= rgba.w*0.8;
    gl_FragColor = rgba;
}



