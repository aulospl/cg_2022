import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
from shared import matrixops
import math


def draw_circle_left(radius,position_x,position_y):
    num_vertices = 64 # define a "qualidade" do circulo
    pi = 3.14
    counter = 0
    radius = radius
    vertices = []

    angle = 0.0
    for counter in range(num_vertices):
        angle += 2*pi/num_vertices 
        x = math.cos(angle)*radius - position_x
        y = math.sin(angle)*radius - position_y
        vertices.append([x,y])  
    
    return vertices

def draw_circle_right(radius,position_x,position_y):
    num_vertices = 64 # define a "qualidade" do circulo
    pi = 3.14
    counter = 0
    radius = radius
    vertices = []

    angle = 0.0
    for counter in range(num_vertices):
        angle += 2*pi/num_vertices 
        x = math.cos(angle)*radius + position_x
        y = math.sin(angle)*radius + position_y
        vertices.append([x,y])  
    
    return vertices

iden = np.array([          1.0, 0.0, 0.0, 0.0, 
                                    0.0, 1.0, 0.0, 0.0, 
                                    0.0, 0.0, 1.0, 0.0, 
                                    0.0, 0.0, 0.0, 1.0], np.float32)


# f, flag de alternar botoes (seta direita e esquerda)
# tx, ty, deslocamento do braço do kirby
# sx, sy, escala da agua
# tree_sy, escala da arvore
# d, graus de rotação da alavanca
# rf, flag de limitador de rotação da alavanca (ir e voltar ao rotacionar 45º)
# c, contador de vezes que o botão foi apertado
# signal, flag de sinal, atualmente não utilizado

tx = 0
ty = 0
sx = 1
sy = 1
tree_sy = 1
d = 0
rf = True
c = 0
signal = False
def key_event(window,key,scancode,action,mods):
    global tx, sy, d, rf,c, signal, tree_sy
  
    if c > 53:
        print("FIM!")
        return 1

    elif action == glfw.PRESS and key == glfw.KEY_SPACE:
        signal = True
        c += 1
        sy -= 0.018518519
        tree_sy += 0.018518519
       # ty  += 0.01
        if d == -45:
            rf = True
        elif d == 0:
            rf = False
        if rf:
            d += 5
            tx  -= 0.01
        else:
            d -= 5
            tx  += 0.01
    

def shaders_config():
    vertex_code = """
            attribute vec2 position;
            uniform mat4 mat;
            void main(){
                gl_Position = mat * vec4(position,0.0,1.0);
            }
            """
    fragment_code = """
            uniform vec4 color;
            void main(){
                gl_FragColor = color;
            }
            """
    programs  = [glCreateProgram() for x in range(11)]
  
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

    for prog in programs:
        glAttachShader(prog, vertex)
        glAttachShader(prog, fragment)

        glLinkProgram(prog)
        if not glGetProgramiv(prog, GL_LINK_STATUS):
            print(glGetProgramInfoLog(program))
            raise RuntimeError('Linking error')

    return programs

if __name__ == "__main__":
    glfw.init()

    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(800, 600, "Trabalho 1", None, None)
    glfw.make_context_current(window)
    
    glfw.set_key_callback(window,key_event)

    glfw.show_window(window)

    programs = shaders_config()
    
    for prog in programs:
        glUseProgram(prog)


    tank = [(0, 0.25),(0, +0.75),(-0.25, 0.25),(-0.25, +0.75)]
    water = [(-0.05, 0.30),(-0.05, +0.75),(-0.2, 0.30),(-0.2, +0.75)]
    handle =  [(-0.35, -0.45),(-0.35, -0.15),(-0.40, -0.45),(-0.40, -0.15)]
    machine = [(0.5, -0.5), (0.5, -0.2), (-0.15, -0.2), (-0.15, -0.5)]
    piston = [(0.3, -0.4), (0.3, -0.3), (0, -0.3), (0, -0.4)]
    tree = [(0.75, -0.45),(0.75, -0.15),(0.60, -0.45),(0.60, -0.15)]
    tree_top = [(0.8, -0.15),(0.67, 0.1),(0.55, -0.15)]
    pipe_tank = [(0, 0.25),(0, -0.2),(-0.05, 0.25),(-0.05, -0.2)]
    pipe_tree = [(0.5, -0.5), (0.5, -0.45), (0.75, -0.45), (0.75, -0.5)]
    kirby = draw_circle_left(radius=0.17, position_x=0.75, position_y=0.25)
    kirbyarm = [(-0.75, -0.25),(-0.75, -0.2),(-0.40, -0.25),(-0.40, -0.2)]
    vertices = tank + water+ handle + machine + piston + tree + tree_top + pipe_tank + pipe_tree + kirby + kirbyarm  # Concatenação de todos os vértices
    vertices = np.array(vertices,  dtype=np.float32)
    buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, buffer)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, buffer)
    stride = vertices.strides[0]
    offset = ctypes.c_void_p(0)
    locs = [glGetAttribLocation(prog, "position") for prog in programs]
    for loc in locs:
        glEnableVertexAttribArray(loc)
        glVertexAttribPointer(loc, 2, GL_FLOAT, False, stride, offset)

    # loc_color = glGetUniformLocation(program, "color")
    # R = 1.0
    # G = 0.0
    # B = 0.0
    
    # fix streching problem
    #glViewport(0, 0, 800, 600);
    #glMatrixMode(GL_PROJECTION);
    #aspect = 800.0 / 600.0;
    #glOrtho(-aspect, aspect, -1, 1, -1, 1);

    #glMatrixMode(GL_MODELVIEW);
    #glLoadIdentity();
        # endfix
    glfw.show_window(window)

    while not glfw.window_should_close(window):

        

        signal = False
        # funcao interna do glfw para gerenciar eventos de mouse, teclado, etc
        glfw.poll_events() 
        # limpa a cor de fundo da janela e preenche com outra no sistema RGBA
        glClear(GL_COLOR_BUFFER_BIT)
        
        # definindo a cor da janela      
        glClearColor(1.0, 1.0, 1.0, 1.0)
        
        
        #Modificando cor do objeto
        # tank draw
        tank_color = glGetUniformLocation(programs[0], "color")
        glUniform4f(tank_color,0, 0, 0, 0)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, len(tank))
        # water_tank draw
        esc = np.array([            sx, 0.0, 0.0, 0.0, 
                                    0.0, sy, 0.0, 0.0, 
                                    0.0, 0.0, 1.0, 0.0, 
                                    0.0, 0.0, 0.0, 1.0], np.float32)
        
      

        water_scale = glGetUniformLocation(programs[1], "mat")
        glUniformMatrix4fv(water_scale, 1, GL_TRUE, esc)
        water_color = glGetUniformLocation(programs[1], "color")
        glUniform4f(water_color,0, 0, 1, 0)
        glDrawArrays(GL_TRIANGLE_STRIP, len(tank), len(water))
        glUniformMatrix4fv(water_scale, 1, GL_TRUE, iden)
        
        # handle draw

        hand_trans = np.array([     1.0, 0.0, 0.0, 0.35, 
                                    0.0, 1.0, 0.0, 0.45, 
                                    0.0, 0.0, 1.0, 0.0, 
                                    0.0, 0.0, 0.0, 1.0], np.float32)
        
        hand_trans_reverse = np.array([     1.0, 0.0, 0.0, -0.35, 
                                    0.0, 1.0, 0.0, -0.45, 
                                    0.0, 0.0, 1.0, 0.0, 
                                    0.0, 0.0, 0.0, 1.0], np.float32)

        rot = np.array([            matrixops.cos(-d), -matrixops.sin(-d), 0.0, d/300, 
                                    matrixops.sin(-d), matrixops.cos(-d), 0.0, -d/250, 
                                    0.0, 0.0, 1.0, 0.0, 
                                    0.0, 0.0, 0.0, 1.0], np.float32)
        handle_rot = glGetUniformLocation(programs[2], "mat")
        
        # transformation = np.multiply(rot,hand_trans_reverse)
        # transformation = np.multiply(transformation,hand_trans)
        # print(transformation)
        
        # transformation = np.matmul(transformation,hand_trans_reverse).reshape(4,4)
        glUniformMatrix4fv(handle_rot, 1, GL_TRUE, rot)
        

        handle_color = glGetUniformLocation(programs[2], "color")
        glUniform4f(handle_color,0, 0, 0, 0)

        glDrawArrays(GL_TRIANGLE_STRIP, len(tank) + len(water), len(handle))

        
        handle_rot2 = glGetUniformLocation(programs[2], "mat")
        glUniformMatrix4fv(handle_rot2, 1, GL_TRUE, iden)

        # machine draw
        glDrawArrays(GL_TRIANGLE_FAN, len(tank)+ len(water) +len(handle), len(machine))
        # piston draw
        piston_color = glGetUniformLocation(programs[3], "color")
        glUniform4f(piston_color,0.752941, 0.752941, 0.752941, 0)
        glDrawArrays(GL_TRIANGLE_FAN, len(tank)+ len(water)+ len(handle) + len(machine), len(piston))
        # tree draw
        esc = np.array([            1, 0.0, 0.0, 0.0, 
                                    0.0, tree_sy, 0.0, 0 + tree_sy/10, 
                                    0.0, 0.0, 1.0, 0.0, 
                                    0.0, 0.0, 0.0, 1.0], np.float32)
        
        # tretran = np.array([        1.0, 0.0, 0.0, 1.0, 
        #                             0.0, 1.0, 0.0, 1.0, 
        #                             0.0, 0.0, 1.0, 0.0, 
        #                             0.0, 0.0, 0.0, 1.0], np.float32)

        #treemat = np.matmul(tretran, esc)
        tree_scale = glGetUniformLocation(programs[4], "mat")
        glUniformMatrix4fv(tree_scale, 1, GL_TRUE, esc)

        tree_color = glGetUniformLocation(programs[4], "color")
        glUniform4f(tree_color,0.65, 0.5, 0.26, 1.0)
        glDrawArrays(GL_TRIANGLE_STRIP, len(tank)+ len(water)+ len(handle) + len(machine) + len(piston), len(tree))
        # tree top draw
        glUniformMatrix4fv(tree_scale, 1, GL_TRUE, iden)
        tree_color = glGetUniformLocation(programs[5], "color")
        glUniform4f(tree_color,0, 1, 0, 1.0)
        glDrawArrays(GL_TRIANGLE_FAN, len(tank)+ len(water)+ len(handle) + len(machine) + len(tree) + len(piston),len(tree_top))
        # pipe_tank
        pipe_tank_color = glGetUniformLocation(programs[6], "color")
        glUniform4f(pipe_tank_color,0.752941, 0.752941, 0.752941, 0)
        glDrawArrays(GL_TRIANGLE_STRIP,len(tank)+ len(water)+ len(handle) + len(machine) + len(tree) + len(piston) + len(tree_top),len(pipe_tank))
        # pipe_tank
        pipe_tree_color = glGetUniformLocation(programs[7], "color")
        glUniform4f(pipe_tree_color,0.752941, 0.752941, 0.752941, 0)
        glDrawArrays(GL_TRIANGLE_FAN,len(tank)+ len(water)+ len(handle) + len(machine) + len(tree) + len(piston) + len(tree_top) + len(pipe_tank),len(pipe_tree))
        
        # Kirby draw
        kirby_color = glGetUniformLocation(programs[9], "color")
        glUniform4f(kirby_color,1, 0.43, 0.78, 0)
        glDrawArrays(GL_TRIANGLE_FAN,len(tank)+ len(water)+ len(handle) + len(machine) + len(tree) + len(piston) + len(tree_top) + len(pipe_tank) + len(pipe_tree),len(kirby))
        
        # KirbyArm_draw
        
        tra = np.array([            1.0, 0.0, 0.0, tx, 
                                    0.0, 1.0, 0.0, ty, 
                                    0.0, 0.0, 1.0, 0.0, 
                                    0.0, 0.0, 0.0, 1.0], np.float32)

      
        kirby_arm_translattion = glGetUniformLocation(programs[10], "mat")
        glUniformMatrix4fv(kirby_arm_translattion, 1, GL_TRUE, tra)

        kirby_arm_color = glGetUniformLocation(programs[10], "color")
        glUniform4f(kirby_arm_color,1, 0.43, 0.78, 0)
        glDrawArrays(GL_TRIANGLE_STRIP,len(tank)+ len(water)+ len(handle) + len(machine) + len(tree) + len(piston) + len(tree_top) + len(pipe_tank) + len(pipe_tree) + len(kirby),len(kirbyarm))
        glUniformMatrix4fv(kirby_arm_translattion, 1, GL_TRUE, iden)
        
        

        # gerencia troca de dados entre janela e o OpenGL
        glfw.swap_buffers(window)
    
    glfw.terminate()
    
        