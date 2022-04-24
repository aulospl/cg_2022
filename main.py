import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
from shared import matrixops


glfw.init()

glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
window = glfw.create_window(960, 720, "Trabalho 1", None, None)
glfw.make_context_current(window)


f = True
def key_event(window,key,scancode,action,mods):
    global f
    if f and action == glfw.PRESS and key == glfw.KEY_RIGHT:
        print("right")
        f = False
    elif (not f) and action == glfw.PRESS and key == glfw.KEY_LEFT:
        print("left")
        f = True

glfw.set_key_callback(window,key_event)

glfw.show_window(window)


while not glfw.window_should_close(window):

    
    # funcao interna do glfw para gerenciar eventos de mouse, teclado, etc
    glfw.poll_events() 

    # limpa a cor de fundo da janela e preenche com outra no sistema RGBA
    glClear(GL_COLOR_BUFFER_BIT)
    
    # definindo a cor da janela      
    glClearColor(1.0, 1.0, 1.0, 1.0)

    # gerencia troca de dados entre janela e o OpenGL
    glfw.swap_buffers(window)

glfw.terminate()