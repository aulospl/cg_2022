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


while not glfw.window_should_close(window):
    pass
glfw.terminate()