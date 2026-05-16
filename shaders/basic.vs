#version 330 core

// Atributos do vertice (mesmo layout do projeto 2: pos + uv + normal).
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec2 in_uv;
layout (location = 2) in vec3 in_normal;

// Matrizes padrao do pipeline moderno
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;

// Matriz para transformar normais (3x3, sem componente de translacao).
// O ideal e transpose(inverse(mat3(model))) — o calculo e feito no Python
// para evitar inversao na GPU.
uniform mat3 u_normal_mat;

// Saidas usadas pelo fragment shader para calcular iluminacao Phong
out vec2 v_uv;
out vec3 v_world_pos; // posicao do fragmento em coordenadas de mundo
out vec3 v_normal;    // normal em coordenadas de mundo

void main() {
    v_uv = in_uv;
    // Posicao em mundo: usada pelas direcoes de luz e camera no fragment shader
    vec4 world = u_model * vec4(in_position, 1.0);
    v_world_pos = world.xyz;
    // Normais em mundo (sem escala nao-uniforme distorcer)
    v_normal = normalize(u_normal_mat * in_normal);
    gl_Position = u_proj * u_view * world;
}
