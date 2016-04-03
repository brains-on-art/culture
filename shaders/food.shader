[Vertex shader]
#version 330

layout(location = 0)in vec2 position;
layout(location = 1)in vec2 texture_coordinate;
layout(location = 2)in vec4 food_position;
//layout(location = 3)in vec4 creature_texture;

uniform mat4 MVP;

out vec2 uv;

void main()
{
    float theta = food_position.z;
    mat2 z_rot = mat2(cos(theta), -sin(theta),
                      sin(theta), cos(theta));

    float scale = food_position.w;
    gl_Position = MVP * vec4(z_rot*scale*position + food_position.xy, 0.0, 1.0);
    uv = texture_coordinate;
}

[Fragment shader]
#version 330

in vec2 uv;

uniform sampler2D image;

void main()
{
    gl_FragColor = texture(image, uv);
}


