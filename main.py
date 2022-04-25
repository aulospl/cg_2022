import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
from shared import matrixops
import math


def draw_circle(radius):
    num_vertices = 64 # define a "qualidade" do circulo
    pi = 3.14
    counter = 0
    radius = radius
    vertices = []

    angle = 0.0
    for counter in range(num_vertices):
        angle += 2*pi/num_vertices 
        x = math.cos(angle)*radius - 0.35
        y = math.sin(angle)*radius - 0.35
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
            void main(){
                gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
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
    handle =  draw_circle(radius=0.2)
    machine = [(0.5, -0.5), (0.5, -0.2), (-0.15, -0.2), (-0.15, -0.5)]
    tree = [(0.8, -0.45),(0.8, -0.15),(0.55, -0.45),(0.55, -0.15)]
    vertices = tank + handle + machine + tree # Concatenação de todos os vértices
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

    glfw.show_window(window)

    while not glfw.window_should_close(window):

    
        # funcao interna do glfw para gerenciar eventos de mouse, teclado, etc
        glfw.poll_events() 

        # limpa a cor de fundo da janela e preenche com outra no sistema RGBA
        glClear(GL_COLOR_BUFFER_BIT)
        
        # definindo a cor da janela      
        glClearColor(1.0, 1.0, 1.0, 1.0)
        
        #Modificando cor do objeto
        
        glDrawArrays(GL_TRIANGLE_STRIP, 0, len(tank))
        glDrawArrays(GL_TRIANGLE_FAN, len(tank), len(handle))
        glDrawArrays(GL_TRIANGLE_STRIP, len(tank)+ len(handle), len(machine))
        glDrawArrays(GL_TRIANGLE_STRIP, len(tank)+ len(handle) + len(machine), len(tree))

        # gerencia troca de dados entre janela e o OpenGL
        glfw.swap_buffers(window)
    
    glfw.terminate()
    
        