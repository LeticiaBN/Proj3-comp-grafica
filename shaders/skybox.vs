#version 330 core

layout (location = 0) in vec3 in_position;

uniform mat4 u_view;   // view SEM translação
uniform mat4 u_proj;

out vec3 v_dir;

void main() {
    v_dir = in_position;
    vec4 pos = u_proj * u_view * vec4(in_position, 1.0);
    // truque: força z=w para que o skybox seja sempre desenhado no fundo
    gl_Position = pos.xyww;
}
