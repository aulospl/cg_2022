import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
from shared import matrixops


glfw.init()

glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
window = glfw.create_window(800, 600, "Trabalho 1", None, None)
glfw.make_context_current(window)

def key_event(window,key,scancode,action,mods):
    print('[key event] key=',key)
    print('[key event] scancode=',scancode)
    print('[key event] action=',action)
    print('[key event] mods=',mods)
    print('-------')
    
glfw.set_key_callback(window,key_event)

glfw.show_window(window)


while not glfw.window_should_close(window):

    
    # funcao interna do glfw para gerenciar eventos de mouse, teclado, etc
    glfw.poll_events() 

    # limpa a cor de fundo da janela e preenche com outra no sistema RGBA
    glClear(GL_COLOR_BUFFER_BIT)
    
    # definindo a cor da janela      
    glClearColor(0, 0, 0, 1.0)

    # gerencia troca de dados entre janela e o OpenGL
    glfw.swap_buffers(window)

glfw.terminate()