#version 330 core

// Saidas interpoladas vindas do vertex shader (Phong por fragmento).
in vec2 v_uv;
in vec3 v_world_pos;
in vec3 v_normal;

// Textura difusa (cor base da superficie).
uniform sampler2D u_tex;

// ---- Material (proprio de cada objeto, definido no Python).
// Substitui os parametros do .mtl, conforme exigido no enunciado (req. 7).
uniform vec3  u_kd;          // cor difusa do objeto (multiplica a textura)
uniform vec3  u_ks;          // cor especular do objeto
uniform float u_shininess;   // expoente especular (Ns no .mtl)

// ---- Coeficientes globais ajustaveis por teclado (req. 4, 5, 6).
uniform float u_ka_strength;        // intensidade da luz ambiente
uniform float u_diffuse_strength;   // multiplicador da reflexao difusa
uniform float u_specular_strength;  // multiplicador da reflexao especular

// ---- Luz ambiente.
uniform vec3 u_ambient_color;   // cor da luz ambiente
uniform int  u_ambient_on;      // 1 = ligada, 0 = desligada

// ---- Luzes pontuais.
// Indices fixos: 0 = farol do rover (externa)
//                1 = lampada de teto (interna A)
// (slot da luz 2 reservado para o futuro objeto-fonte interno B —
//  remover esses comentarios quando re-adicionar a luz; tambem
//  vai precisar bumpar N_LIGHTS de volta para 3.)
const int N_LIGHTS = 2;
uniform vec3 u_light_pos[N_LIGHTS];
uniform vec3 u_light_color[N_LIGHTS];
uniform int  u_light_on[N_LIGHTS];

// Mascara de bits indicando quais luzes podem afetar este objeto.
// bit 0 = luz 0 (externa), bit 1 = luz 1 (interna), bit 2 = luz 2 (interna).
// Externos recebem mascara 1; internos recebem mascara 6; "shared" recebe 7.
uniform int u_light_mask;

// Posicao do observador (para reflexao especular).
uniform vec3 u_view_pos;

// Flags utilitarias.
uniform int u_wireframe;  // 1 = desenha em cor solida (mesma logica do projeto 2)
uniform int u_unlit;      // 1 = objeto emissivo (lampadas) — nao aplica iluminacao

out vec4 frag_color;

void main() {
    // Caminho do wireframe: cor uniforme, sem iluminacao.
    if (u_wireframe == 1) {
        frag_color = vec4(0.95, 0.95, 0.10, 1.0);
        return;
    }

    // Cor base do fragmento = textura * cor difusa do objeto.
    vec4 texel = texture(u_tex, v_uv);
    vec3 base_color = texel.rgb * u_kd;

    // Objetos "unlit" sao as proprias lampadas (sphere/cube emissivo).
    // Eles brilham com a propria cor, mesmo com luzes apagadas, para
    // ficar claro visualmente onde esta cada fonte de luz.
    if (u_unlit == 1) {
        frag_color = vec4(base_color, 1.0);
        return;
    }

    // Normaliza vetor normal interpolado (req. essencial do modelo Phong).
    vec3 N = normalize(v_normal);
    // Para superficies de "duas faces" (paredes da base desenhadas
    // sem backface culling, planeta, etc.), a face de tras herda a
    // mesma normal apontando pra fora — o que faz a luz do lado errado
    // "vazar" pra dentro do volume e impede a luz interna de iluminar
    // a parede pelo lado de dentro. gl_FrontFacing nos diz se o pixel
    // pertence ao lado da frente do triangulo; invertemos a normal
    // quando vemos o verso. Em superficies com culling normal essa
    // branch nunca dispara (back faces sao descartadas antes), entao
    // nao afeta nenhum outro objeto.
    if (!gl_FrontFacing) {
        N = -N;
    }
    // Direcao do observador (usada na reflexao especular).
    vec3 V = normalize(u_view_pos - v_world_pos);

    // -----------------------------------------------------------------
    // (1) Reflexao AMBIENTE — afeta todos os objetos por igual.
    //     Ka * cor_ambiente * cor_base.
    // -----------------------------------------------------------------
    vec3 ambient = vec3(0.0);
    if (u_ambient_on == 1) {
        ambient = u_ka_strength * u_ambient_color * base_color;
    }

    // -----------------------------------------------------------------
    // (2) Reflexao DIFUSA + ESPECULAR — uma contribuicao por luz ativa
    //     que esteja autorizada pela mascara deste objeto. Atenuacao por
    //     distancia (constante + linear + quadratico) suaviza o alcance
    //     das luzes pontuais.
    // -----------------------------------------------------------------
    vec3 diffuse_sum  = vec3(0.0);
    vec3 specular_sum = vec3(0.0);

    for (int i = 0; i < N_LIGHTS; i++) {
        if (u_light_on[i] == 0) continue;
        // Mascara: este objeto e afetado por esta luz?
        if ((u_light_mask & (1 << i)) == 0) continue;

        // Vetor da superficie ate a luz.
        vec3 to_light = u_light_pos[i] - v_world_pos;
        float dist = length(to_light);
        vec3 L = to_light / max(dist, 0.0001);

        // Atenuacao quadratica (modelo OpenGL classico): luzes pontuais
        // perdem intensidade com a distancia, o que permite a luz do
        // rover destacar so o que esta proximo dele.
        float att = 1.0 / (1.0 + 0.022 * dist + 0.0019 * dist * dist);

        // --- Difusa (lei de Lambert) ---
        float diff = max(dot(N, L), 0.0);
        diffuse_sum += diff * u_light_color[i] * att;

        // --- Especular (Phong classico com reflect()) ---
        // Apenas calcula se a face estiver virada para a luz, evitando
        // brilho "vazando" no lado oposto.
        if (diff > 0.0) {
            vec3 R = reflect(-L, N);
            float spec = pow(max(dot(R, V), 0.0), u_shininess);
            specular_sum += spec * u_light_color[i] * att;
        }
    }

    // Aplica os multiplicadores globais (teclas inc/dec) e a cor do material.
    vec3 diffuse  = diffuse_sum  * base_color * u_diffuse_strength;
    vec3 specular = specular_sum * u_ks       * u_specular_strength;

    // Modelo final: I = Ia + soma(Id + Is) por luz ativa.
    vec3 result = ambient + diffuse + specular;
    frag_color = vec4(result, 1.0);
}
