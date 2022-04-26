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



def key_event(window,key,scancode,action,mods):
    print('[key event] key=',key)
    print('[key event] scancode=',scancode)
    print('[key event] action=',action)
    print('[key event] mods=',mods)
    print('-------')
    

def shaders_config():
    vertex_code = """
            attribute vec2 position;
            void main(){
                gl_Position = vec4(position,0.0,1.0);
            }
            """
    fragment_code = """
            uniform vec4 color;
            void main(){
                gl_FragColor = color;
            }
            """
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
    return program

if __name__ == "__main__":
    glfw.init()

    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(800, 600, "Trabalho 1", None, None)
    glfw.make_context_current(window)
    
    glfw.set_key_callback(window,key_event)

    glfw.show_window(window)

    program = shaders_config()
        
    glUseProgram(program)


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
    loc = glGetAttribLocation(program, "position")
    glEnableVertexAttribArray(loc)
    glVertexAttribPointer(loc, 2, GL_FLOAT, False, stride, offset)

    # loc_color = glGetUniformLocation(program, "color")
    # R = 1.0
    # G = 0.0
    # B = 0.0
    
    
    
    glfw.show_window(window)

    while not glfw.window_should_close(window):

    
        # funcao interna do glfw para gerenciar eventos de mouse, teclado, etc
        glfw.poll_events() 

        # limpa a cor de fundo da janela e preenche com outra no sistema RGBA
        glClear(GL_COLOR_BUFFER_BIT)
        
        # definindo a cor da janela      
        glClearColor(1.0, 1.0, 1.0, 1.0)
        
        
        #Modificando cor do objeto
        # tank draw
        tank_color = glGetUniformLocation(program, "color")
        glUniform4f(tank_color,0, 0, 0, 0)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, len(tank))
        # water_tank draw
        water_color = glGetUniformLocation(program, "color")
        glUniform4f(water_color,0, 0, 1, 0)
        glDrawArrays(GL_TRIANGLE_STRIP, len(tank), len(water))
        # handle draw
        handle_color = glGetUniformLocation(program, "color")
        glUniform4f(handle_color,0, 0, 0, 0)
        glDrawArrays(GL_TRIANGLE_STRIP, len(tank) + len(water), len(handle))
        # machine draw
        glDrawArrays(GL_TRIANGLE_FAN, len(tank)+ len(water) +len(handle), len(machine))
        # piston draw
        piston_color = glGetUniformLocation(program, "color")
        glUniform4f(piston_color,0.752941, 0.752941, 0.752941, 0)
        glDrawArrays(GL_TRIANGLE_FAN, len(tank)+ len(water)+ len(handle) + len(machine), len(piston))
        # tree draw
        tree_color = glGetUniformLocation(program, "color")
        glUniform4f(tree_color,0.65, 0.5, 0.26, 1.0)
        glDrawArrays(GL_TRIANGLE_STRIP, len(tank)+ len(water)+ len(handle) + len(machine) + len(piston), len(tree))
        # tree top draw
        tree_color = glGetUniformLocation(program, "color")
        glUniform4f(tree_color,0, 1, 0, 1.0)
        glDrawArrays(GL_TRIANGLE_FAN, len(tank)+ len(water)+ len(handle) + len(machine) + len(tree) + len(piston),len(tree_top))
        # pipe_tank
        pipe_tank_color = glGetUniformLocation(program, "color")
        glUniform4f(pipe_tank_color,0.752941, 0.752941, 0.752941, 0)
        glDrawArrays(GL_TRIANGLE_STRIP,len(tank)+ len(water)+ len(handle) + len(machine) + len(tree) + len(piston) + len(tree_top),len(pipe_tank))
        # pipe_tank
        pipe_tree_color = glGetUniformLocation(program, "color")
        glUniform4f(pipe_tree_color,0.752941, 0.752941, 0.752941, 0)
        glDrawArrays(GL_TRIANGLE_FAN,len(tank)+ len(water)+ len(handle) + len(machine) + len(tree) + len(piston) + len(tree_top) + len(pipe_tank),len(pipe_tree))
        # Kirby draw
        kirby_color = glGetUniformLocation(program, "color")
        glUniform4f(kirby_color,1, 0.43, 0.78, 0)
        glDrawArrays(GL_TRIANGLE_FAN,len(tank)+ len(water)+ len(handle) + len(machine) + len(tree) + len(piston) + len(tree_top) + len(pipe_tank) + len(pipe_tree),len(kirby))
        # KirbyArm_draw
        kirby_arm_color = glGetUniformLocation(program, "color")
        glUniform4f(kirby_arm_color,1, 0.43, 0.78, 0)
        glDrawArrays(GL_TRIANGLE_STRIP,len(tank)+ len(water)+ len(handle) + len(machine) + len(tree) + len(piston) + len(tree_top) + len(pipe_tank) + len(pipe_tree) + len(kirby),len(kirbyarm))
        # gerencia troca de dados entre janela e o OpenGL
        glfw.swap_buffers(window)
    
    glfw.terminate()
    
        