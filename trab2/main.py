import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import glm
import math
from PIL import Image

glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.FALSE);
altura = 1600
largura = 1200
window = glfw.create_window(largura, altura, "Trabalho 2", None, None)
glfw.make_context_current(window)


#vertex_code = """
#        attribute vec3 position;
#        attribute vec2 texture_coord;
#        varying vec2 out_texture;
#                
#        uniform mat4 model;
#        uniform mat4 view;
#        uniform mat4 projection;        
#        
#        void main(){
#            gl_Position = projection * view * model * vec4(position,1.0);
#            out_texture = vec2(texture_coord);
#        }
#        """

vertex_code = """
        attribute vec3 position;
        attribute vec2 texture_coord;
        attribute vec3 normals;
        
       
        varying vec2 out_texture;
        varying vec3 out_fragPos;
        varying vec3 out_normal;
                
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;        
        
        void main(){
            gl_Position = projection * view * model * vec4(position,1.0);
            out_texture = vec2(texture_coord);
            out_fragPos = vec3(position);
            out_normal = normals;
        }
        """

fragment_code = """

        uniform vec3 lightPos; // define coordenadas de posicao da luz
        uniform float ka; // coeficiente de reflexao ambiente
        uniform float kd; // coeficiente de reflexao difusa
        
        vec3 lightColor = vec3(1.0, 1.0, 1.0);
        

        varying vec2 out_texture; // recebido do vertex shader
        varying vec3 out_normal; // recebido do vertex shader
        varying vec3 out_fragPos; // recebido do vertex shader
        uniform sampler2D samplerTexture;
        
        
        
        void main(){
            vec3 ambient = ka * lightColor;             
        
            vec3 norm = normalize(out_normal); // normaliza vetores perpendiculares
            vec3 lightDir = normalize(lightPos - out_fragPos); // direcao da luz
            float diff = max(dot(norm, lightDir), 0.0); // verifica limite angular (entre 0 e 90)
            vec3 diffuse = kd * diff * lightColor; // iluminacao difusa
            
            vec4 texture = texture2D(samplerTexture, out_texture);
            vec4 result = vec4((ambient + diffuse),1.0) * texture; // aplica iluminacao
            gl_FragColor = result;

        }"""

program  = glCreateProgram()
vertex   = glCreateShader(GL_VERTEX_SHADER)
fragment = glCreateShader(GL_FRAGMENT_SHADER)

glShaderSource(vertex, vertex_code)
glShaderSource(fragment, fragment_code)

glCompileShader(vertex)
if not glGetShaderiv(vertex, GL_COMPILE_STATUS):
    error = glGetShaderInfoLog(vertex).decode()
    print(error)
    raise RuntimeError("Erro de compilacao do Vertex Shader")

glCompileShader(fragment)
if not glGetShaderiv(fragment, GL_COMPILE_STATUS):
    error = glGetShaderInfoLog(fragment).decode()
    print(error)
    raise RuntimeError("Erro de compilacao do Fragment Shader")

glAttachShader(program, vertex)
glAttachShader(program, fragment)

glLinkProgram(program)
if not glGetProgramiv(program, GL_LINK_STATUS):
    print(glGetProgramInfoLog(program))
    raise RuntimeError('Linking error')
    
glUseProgram(program)


def load_model_from_file(filename):
    """Loads a Wavefront OBJ file. """
    objects = {}
    vertices = []
    normals = []
    texture_coords = []
    faces = []

    material = None

    # abre o arquivo obj para leitura
    for line in open(filename, "r"): ## para cada linha do arquivo .obj
        if line.startswith('#'): continue ## ignora comentarios
        values = line.split() # quebra a linha por espaço
        if not values: continue


        ### recuperando vertices
        if values[0] == 'v':
            vertices.append(values[1:4])

        ### recuperando vertices
        if values[0] == 'vn':
            normals.append(values[1:4])

        ### recuperando coordenadas de textura
        elif values[0] == 'vt':
            texture_coords.append(values[1:3])

        ### recuperando faces 
        elif values[0] in ('usemtl', 'usemat'):
            material = values[1]
        elif values[0] == 'f':
            face = []
            face_texture = []
            face_normals = []
            for v in values[1:]:
                w = v.split('/')
                face.append(int(w[0]))
                face_normals.append(int(w[2]))
                if len(w) >= 2 and len(w[1]) > 0:
                    face_texture.append(int(w[1]))
                else:
                    face_texture.append(0)

            faces.append((face, face_texture, face_normals, material))

    model = {}
    model['vertices'] = vertices
    model['texture'] = texture_coords
    model['faces'] = faces
    model['normals'] = normals

    return model

glEnable(GL_TEXTURE_2D)
qtd_texturas = 10
textures = glGenTextures(qtd_texturas)

def load_texture_from_file(texture_id, img_textura):
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    img = Image.open(img_textura)
    img_width = img.size[0]
    img_height = img.size[1]
    image_data = img.tobytes("raw", "RGB", 0, -1)
    #image_data = np.array(list(img.getdata()), np.uint8)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img_width, img_height, 0, GL_RGB, GL_UNSIGNED_BYTE, image_data)

vertices_list = []  
normals_list = []      
textures_coord_list = []

# Carregar modelos e aplicar texturas
modelo = load_model_from_file('terreno/terreno2.obj')

print(modelo['normals'])

### inserindo vertices do modelo no vetor de vertices
print('Processando modelo terreno.obj. Vertice inicial:',len(vertices_list))
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append( modelo['vertices'][vertice_id-1] )
    for texture_id in face[1]:
        textures_coord_list.append( modelo['texture'][texture_id-1] )
    for normal_id in face[2]:
        normals_list.append( modelo['normals'][normal_id-1] )
print('Processando modelo terreno.obj. Vertice final:',len(vertices_list))

load_texture_from_file(0,'terreno/grama.jpg')

# Sol
modelo = load_model_from_file('ceu/sphere1.obj')

### inserindo vertices do modelo no vetor de vertices
print('Processando modelo sunobj.obj. Vertice inicial:',len(vertices_list))
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append( modelo['vertices'][vertice_id-1] )
    for texture_id in face[1]:
        textures_coord_list.append( modelo['texture'][texture_id-1] )
    for normal_id in face[2]:
        normals_list.append( modelo['normals'][normal_id-1] )
print('Processando modelo sunobj.obj. Vertice final:',len(vertices_list))


load_texture_from_file(1,'ceu/sol.jpg')


# Dude
modelo = load_model_from_file('person/cat.obj')

### inserindo vertices do modelo no vetor de vertices
print('Processando modelo dude.obj. Vertice inicial:',len(vertices_list))
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append( modelo['vertices'][vertice_id-1] )
    for texture_id in face[1]:
        textures_coord_list.append( modelo['texture'][texture_id-1] )
    for normal_id in face[2]:
        normals_list.append( modelo['normals'][normal_id-1] )
print('Processando modelo dude.obj. Vertice final:',len(vertices_list))

#Casa
load_texture_from_file(3,'house/casaSimples_D.png')
modelo = load_model_from_file('house/CasaSimplesT.obj')

### inserindo vertices do modelo no vetor de vertices
print('Processando modelo casa.obj. Vertice inicial:',len(vertices_list))
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append( modelo['vertices'][vertice_id-1] )
    for texture_id in face[1]:
        textures_coord_list.append( modelo['texture'][texture_id-1] )
print('Processando modelo casa.obj. Vertice final:',len(vertices_list))

# Arvore
load_texture_from_file(4,'tree/BarkDecidious0194_7_S.jpg')
modelo = load_model_from_file('tree/Lowpoly_tree_sample.obj')

### inserindo vertices do modelo no vetor de vertices
print('Processando modelo tree.obj. Vertice inicial:',len(vertices_list))
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append( modelo['vertices'][vertice_id-1] )
    for texture_id in face[1]:
        textures_coord_list.append( modelo['texture'][texture_id-1] )
print('Processando modelo tree.obj. Vertice final:',len(vertices_list))

# Mesa
load_texture_from_file(5,'table/Wood_Cherry_Original.jpg')
modelo = load_model_from_file('table/Table.obj')

### inserindo vertices do modelo no vetor de vertices
print('Processando modelo mesa.obj. Vertice inicial:',len(vertices_list))
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append( modelo['vertices'][vertice_id-1] )
    for texture_id in face[1]:
        textures_coord_list.append( modelo['texture'][texture_id-1] )
print('Processando modelo mesa.obj. Vertice final:',len(vertices_list))


# Luz
modelo = load_model_from_file('ceu/sphere1.obj')

### inserindo vertices do modelo no vetor de vertices
print('Processando modelo luz.obj. Vertice inicial:',len(vertices_list))
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append( modelo['vertices'][vertice_id-1] )
    for texture_id in face[1]:
        textures_coord_list.append( modelo['texture'][texture_id-1] )
print('Processando modelo luz.obj. Vertice final:',len(vertices_list))


load_texture_from_file(6,'ceu/discoBall.jpeg')
# Rua
modelo = load_model_from_file('terreno/terreno2.obj')

### inserindo vertices do modelo no vetor de vertices
print('Processando modelo terreno.obj. Vertice inicial:',len(vertices_list))
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append( modelo['vertices'][vertice_id-1] )
    for texture_id in face[1]:
        textures_coord_list.append( modelo['texture'][texture_id-1] )
print('Processando modelo terreno.obj. Vertice final:',len(vertices_list))

load_texture_from_file(7,'terreno/pedra.jpg')

# skybox-top
modelo = load_model_from_file('terreno/terreno2.obj')

### inserindo vertices do modelo no vetor de vertices
print('Processando modelo topo.obj. Vertice inicial:',len(vertices_list))
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append( modelo['vertices'][vertice_id-1] )
    for texture_id in face[1]:
        textures_coord_list.append( modelo['texture'][texture_id-1] )
print('Processando modelo topo.obj. Vertice final:',len(vertices_list))

load_texture_from_file(8,'skybox/ceu.jpeg')


# skybox-direita
modelo = load_model_from_file('terreno/terreno2.obj')

### inserindo vertices do modelo no vetor de vertices
print('Processando modelo direita.obj. Vertice inicial:',len(vertices_list))
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append( modelo['vertices'][vertice_id-1] )
    for texture_id in face[1]:
        textures_coord_list.append( modelo['texture'][texture_id-1] )
print('Processando modelo direita.obj. Vertice final:',len(vertices_list))

load_texture_from_file(9,'skybox/ceu.jpeg')

# skybox-esquerd
modelo = load_model_from_file('terreno/terreno2.obj')

### inserindo vertices do modelo no vetor de vertices
print('Processando modelo esquerd.obj. Vertice inicial:',len(vertices_list))
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append( modelo['vertices'][vertice_id-1] )
    for texture_id in face[1]:
        textures_coord_list.append( modelo['texture'][texture_id-1] )
print('Processando modelo esquerd.obj. Vertice final:',len(vertices_list))

load_texture_from_file(9,'skybox/ceu.jpeg')

# skybox-atras
modelo = load_model_from_file('terreno/terreno2.obj')

### inserindo vertices do modelo no vetor de vertices
print('Processando modelo atras.obj. Vertice inicial:',len(vertices_list))
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append( modelo['vertices'][vertice_id-1] )
    for texture_id in face[1]:
        textures_coord_list.append( modelo['texture'][texture_id-1] )
print('Processando modelo atras.obj. Vertice final:',len(vertices_list))

load_texture_from_file(9,'skybox/ceu.jpeg')

# skybox-frente
modelo = load_model_from_file('terreno/terreno2.obj')

### inserindo vertices do modelo no vetor de vertices
print('Processando modelo frente.obj. Vertice inicial:',len(vertices_list))
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append( modelo['vertices'][vertice_id-1] )
    for texture_id in face[1]:
        textures_coord_list.append( modelo['texture'][texture_id-1] )
print('Processando modelo frente.obj. Vertice final:',len(vertices_list))

load_texture_from_file(9,'skybox/ceu.jpeg')

buffer = glGenBuffers(2)


vertices = np.zeros(len(vertices_list), [("position", np.float32, 3)])
vertices['position'] = vertices_list

load_texture_from_file(2,'person/monstro.jpg')

# Upload data
glBindBuffer(GL_ARRAY_BUFFER, buffer[0])
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
stride = vertices.strides[0]
offset = ctypes.c_void_p(0)
loc_vertices = glGetAttribLocation(program, "position")
glEnableVertexAttribArray(loc_vertices)
glVertexAttribPointer(loc_vertices, 3, GL_FLOAT, False, stride, offset)

textures = np.zeros(len(textures_coord_list), [("position", np.float32, 2)]) # duas coordenadas
textures['position'] = textures_coord_list


# Upload data
glBindBuffer(GL_ARRAY_BUFFER, buffer[1])
glBufferData(GL_ARRAY_BUFFER, textures.nbytes, textures, GL_STATIC_DRAW)
stride = textures.strides[0]
offset = ctypes.c_void_p(0)
loc_texture_coord = glGetAttribLocation(program, "texture_coord")
glEnableVertexAttribArray(loc_texture_coord)
glVertexAttribPointer(loc_texture_coord, 2, GL_FLOAT, False, stride, offset)


normals = np.zeros(len(normals_list), [("position", np.float32, 3)]) # três coordenadas
normals['position'] = normals_list


glBindBuffer(GL_ARRAY_BUFFER, buffer[2])
glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
stride = normals.strides[0]
offset = ctypes.c_void_p(0)
loc_normals_coord = glGetAttribLocation(program, "normals")
glEnableVertexAttribArray(loc_normals_coord)
glVertexAttribPointer(loc_normals_coord, 3, GL_FLOAT, False, stride, offset)

#loc_light_pos = glGetUniformLocation(program, "lightPos") # recuperando localizacao da variavel lightPos na GPU
#glUniform3f(loc_light_pos, 0.0, 10.0, 0.0) ### posicao da fonte de luz

ka = 0.5
kd = 0.5
def desenha_terreno():
    # aplica a matriz model
    
    # rotacao
    angle = 0.0;
    r_x = 0.0; r_y = 0.0; r_z = 1.0;
    
    # translacao
    t_x = 0.0; t_y = -1.01; t_z = 0.0;
    
    # escala
    s_x = 20.0; s_y = 20.0; s_z = 20.0;
    
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
       

    
    #### define parametros de ilumincao do modelo
    ka = ka # coeficiente de reflexao ambiente do modelo
    kd = kd # coeficiente de reflexao difusa do modelo
    
    loc_ka = glGetUniformLocation(program, "ka") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka, ka) ### envia ka pra gpu
    
    loc_kd = glGetUniformLocation(program, "kd") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_kd, kd) ### envia kd pra gpu    
    
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, 0)
    
    
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES, 0, 6) ## renderizando

def desenha_sol():
    # aplica a matriz model
    
    # rotacao
    angle = 0.0;
    r_x = 0.0; r_y = 0.0; r_z = 1.0;
    
    # translacao
    t_x = 0.0; t_y = 15.0; t_z = 15.0;
    
    # escala
    s_x = 2.0; s_y = 2.0; s_z = 2.0;
    
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
       

     # iluminação ambiente
    ka = ka # coeficiente de reflexao ambiente do modelo
    kd = kd # coeficiente de reflexao difusa do modelo
    
    loc_ka = glGetUniformLocation(program, "ka") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka, ka) ### envia ka pra gpu
    
    loc_kd = glGetUniformLocation(program, "kd") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_kd, kd) ### envia kd pra gpu    
    
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, 1)
    
    
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES, 6, 2886-6) ## renderizando


def desenha_dude(incoming_z):
    
    
    # aplica a matriz model
    
    # rotacao
    angle = 90.0;
    r_x = 0.0; r_y = 1.0; r_z = 0.0;
    
    # translacao
    t_x = -6.65; t_y = -1.0; t_z = incoming_z;
    
    # escala
    s_x = 0.001; s_y = 0.001; s_z = 0.001;
    
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
       

    ka = ka # coeficiente de reflexao ambiente do modelo
    kd = kd # coeficiente de reflexao difusa do modelo
    
    loc_ka = glGetUniformLocation(program, "ka") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka, ka) ### envia ka pra gpu
    
    loc_kd = glGetUniformLocation(program, "kd") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_kd, kd) ### envia kd pra gpu    

    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, 1)
    
    
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES, 2886, 9132-1990) ## renderizando
    
def desenha_casa():
    
    
    # aplica a matriz model
    
    # rotacao
    angle = 0.0;
    r_x = 0.0; r_y = 1.0; r_z = 0.0;
    
    # translacao
    t_x = -5.2; t_y = 1.65; t_z = -5.2;
    
    # escala
    s_x = 1.0; s_y = 1.0; s_z = 1.0;
    
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
       
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, 3)
    
    
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES, 9132, 14832-9132) ## renderizando
    
def desenha_tree():
    
    
    # aplica a matriz model
    
    # rotacao
    angle = 0.0;
    r_x = 0.0; r_y = 1.0; r_z = 0.0;
    
    # translacao
    t_x = 3.2; t_y = 0.0; t_z = 3.2;
    
    # escala
    s_x = 0.5; s_y = 0.5; s_z = 0.5;
    
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
       
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, 4)
    
    
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES, 14832, 15956-14832) ## renderizando


def desenha_mesa():
    
    
    # aplica a matriz model
    
    # rotacao
    angle = 0.0;
    r_x = 0.0; r_y = 1.0; r_z = 0.0;
    
    # translacao
    t_x = -4.85; t_y = -0.5; t_z = -4.85;
    
    # escala
    s_x = 0.005; s_y = 0.003; s_z = 0.003;
    
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
       
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, 5)
    
    
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES, 15956, 16080-15956) ## renderizando

def desenha_luz():
    # aplica a matriz model
    
    # rotacao
    angle = 0.0;
    r_x = 0.0; r_y = 0.0; r_z = 1.0;
    
    # translacao
    t_x = -4.85; t_y = 4.5; t_z = -4.85;
    
    # escala
    s_x = 0.4; s_y = 0.4; s_z = 0.4;
    
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
       
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, 6)
    
    
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES, 16080, 18960-16080) ## renderizando

def desenha_rua():
    # aplica a matriz model
    
    # rotacao
    angle = 0.0;
    r_x = 0.0; r_y = 0.0; r_z = 1.0;
    
    # translacao
    t_x = -6.65; t_y = -1.0; t_z = 3.15;
    
    # escala
    s_x = 1.0; s_y = 1.0; s_z = 5.0;
    
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
       
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, 7)
    
    
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES,18960, 18966 - 18960) ## renderizando

def desenha_ceu_topo():
    # aplica a matriz model
    
    # rotacao
    angle = 0.0;
    r_x = 0.0; r_y = 0.0; r_z = 1.0;
    
    # translacao
    t_x = 0.0; t_y = 19.0; t_z = 0.0;
    
    # escala
    s_x = 20.0; s_y = 20.0; s_z = 20.0;
    
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
       
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, 8)
    
    glDrawArrays(GL_TRIANGLES,18966, 18972 - 18966) ## renderizando
    
    
def desenha_ceu_direita(rotacao_inc):
    # aplica a matriz model
    
    # rotacao
    angle = 90;
    r_x = 0.0; r_y = 0.0; r_z = 1.0;
    
    # translacao
    t_x = 20.0; t_y = -1.0; t_z = 0.0;
    
    # escala
    s_x = 20.0; s_y = 21.0; s_z = 20.0;
    
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
       
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, 9)
    
    
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES,18972, 18978 - 18972) ## renderizando

def desenha_ceu_esquerda(rotacao_inc):
    # aplica a matriz model
    
    # rotacao
    angle = 90;
    r_x = 0.0; r_y = 0.0; r_z = 1.0;
    
    # translacao
    t_x = -20.0; t_y = -1.0; t_z = 0.0;
    
    # escala
    s_x = 20.0; s_y = 21.0; s_z = 20.0;
    
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
       
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, 9)
    
    
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES,18978, 18984-18978) ## renderizando

def desenha_ceu_atras(rotacao_inc):
    # aplica a matriz model
    
    # rotacao
    angle = 90;
    r_x = 1.0; r_y = 0.0; r_z = 0.0;
    
    # translacao
    t_x = 0.0; t_y = -1.0; t_z = -20.0;
    
    # escala
    s_x = 20.0; s_y = 21.0; s_z = 20.0;
    
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
       
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, 9)
    
    
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES,18984, 18990-18984) ## renderizando

def desenha_ceu_frente(rotacao_inc):
    # aplica a matriz model
    
    # rotacao
    angle = 90;
    r_x = 1.0; r_y = 0.0; r_z = 0.0;
    
    # translacao
    t_x = 0.0; t_y = -1.0; t_z = 20.0;
    
    # escala
    s_x = 20.0; s_y = 21.0; s_z = 20.0;
    
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
       
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, 9)
    
    
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES,18990, 18996-18990) ## renderizando

cameraPos   = glm.vec3(0.0,  0.0,  1.0);
cameraFront = glm.vec3(0.0,  0.0, -1.0);
cameraUp    = glm.vec3(0.0,  1.0,  0.0);


polygonal_mode = False

def check_boundary(position):
    if (position[0] < -20 or position[0] > 20) or (position[1] < 0 or position[0] > 20) or (position[2] < -20 or position[2] > 20):
        return False
    else:
        return True

def key_event(window,key,scancode,action,mods):
    global cameraPos, cameraFront, cameraUp, polygonal_mode
    
    cameraSpeed = 0.2
    if key == 87 and (action==1 or action==2) and check_boundary(cameraPos + (cameraSpeed * cameraFront)): # tecla W
        cameraPos += cameraSpeed * cameraFront
    
    if key == 83 and (action==1 or action==2) and check_boundary(cameraPos - (cameraSpeed * cameraFront)): # tecla S
        cameraPos -= cameraSpeed * cameraFront
    
    if key == 65 and (action==1 or action==2) and check_boundary(cameraPos - (glm.normalize(glm.cross(cameraFront, cameraUp)) * cameraSpeed)): # tecla A
        cameraPos -= glm.normalize(glm.cross(cameraFront, cameraUp)) * cameraSpeed
        
    if key == 68 and (action==1 or action==2) and check_boundary(cameraPos + (glm.normalize(glm.cross(cameraFront, cameraUp)) * cameraSpeed)): # tecla D
        cameraPos += glm.normalize(glm.cross(cameraFront, cameraUp)) * cameraSpeed
        
    if key == 80 and action==1 and polygonal_mode==True:
        polygonal_mode=False
    else:
        if key == 80 and action==1 and polygonal_mode==False:
            polygonal_mode=True
        
        
        
        
firstMouse = True
yaw = -90.0 
pitch = 0.0
lastX =  largura/2
lastY =  altura/2

def mouse_event(window, xpos, ypos):
    global firstMouse, cameraFront, yaw, pitch, lastX, lastY
    if firstMouse:
        lastX = xpos
        lastY = ypos
        firstMouse = False

    xoffset = xpos - lastX
    yoffset = lastY - ypos
    lastX = xpos
    lastY = ypos

    sensitivity = 0.3 
    xoffset *= sensitivity
    yoffset *= sensitivity

    yaw += xoffset;
    pitch += yoffset;

    
    if pitch >= 90.0: pitch = 90.0
    if pitch <= -90.0: pitch = -90.0

    front = glm.vec3()
    front.x = math.cos(glm.radians(yaw)) * math.cos(glm.radians(pitch))
    front.y = math.sin(glm.radians(pitch))
    front.z = math.sin(glm.radians(yaw)) * math.cos(glm.radians(pitch))
    cameraFront = glm.normalize(front)


    
glfw.set_key_callback(window,key_event)
glfw.set_cursor_pos_callback(window, mouse_event)

def model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z):
    
    angle = math.radians(angle)
    
    matrix_transform = glm.mat4(1.0) # instanciando uma matriz identidade

    
    # aplicando translacao
    matrix_transform = glm.translate(matrix_transform, glm.vec3(t_x, t_y, t_z))    
    
    # aplicando rotacao
    matrix_transform = glm.rotate(matrix_transform, angle, glm.vec3(r_x, r_y, r_z))
    
    # aplicando escala
    matrix_transform = glm.scale(matrix_transform, glm.vec3(s_x, s_y, s_z))
    
    matrix_transform = np.array(matrix_transform)
    
    return matrix_transform

def view():
    global cameraPos, cameraFront, cameraUp
    mat_view = glm.lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
    mat_view = np.array(mat_view)
    return mat_view

def projection():
    global altura, largura
    # perspective parameters: fovy, aspect, near, far
    mat_projection = glm.perspective(glm.radians(45.0), largura/altura, 0.1, 1000.0)
    mat_projection = np.array(mat_projection)    
    return mat_projection

glfw.show_window(window)
glfw.set_cursor_pos(window, lastX, lastY)

glEnable(GL_DEPTH_TEST) ### importante para 3D
   

rotacao_inc = 0
incoming_z = 3.15
mov_flag = False
while not glfw.window_should_close(window):

    glfw.poll_events() 
    
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    glClearColor(0.2, 0.2, 0.2, 1.0)
    
    if polygonal_mode==True:
        glPolygonMode(GL_FRONT_AND_BACK,GL_LINE)
    if polygonal_mode==False:
        glPolygonMode(GL_FRONT_AND_BACK,GL_FILL)
    


    if incoming_z <= -3.15:
        mov_flag = False
    elif incoming_z >= 3.15:
        mov_flag = True


    if mov_flag == False:
        incoming_z += 0.01
    elif mov_flag == True:
        incoming_z -= 0.01
    #else:
    #    incoming_z -= 0.005

    desenha_terreno()
    desenha_sol()
    desenha_dude(incoming_z)
    desenha_casa()
    desenha_tree()
    desenha_mesa()
    desenha_luz()
    desenha_rua()
    desenha_ceu_topo()
    desenha_ceu_direita(rotacao_inc)
    desenha_ceu_esquerda(rotacao_inc)
    desenha_ceu_atras(rotacao_inc)
    desenha_ceu_frente(rotacao_inc)
    
    
    mat_view = view()
    loc_view = glGetUniformLocation(program, "view")
    glUniformMatrix4fv(loc_view, 1, GL_TRUE, mat_view)

    mat_projection = projection()
    loc_projection = glGetUniformLocation(program, "projection")
    glUniformMatrix4fv(loc_projection, 1, GL_TRUE, mat_projection)    
    
    

    
    glfw.swap_buffers(window)

glfw.terminate()