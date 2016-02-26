import os
import numpy as np
from PIL import Image

from OpenGL import GL

import boa_gfx
from boa_gfx import Mesh, VBO, VAO
from boa_gfx.gl_texture import TextureArray
import ctypes

def create_texture_array(texture_directory):
    texture_files = []
    for root, dirs, files in os.walk(texture_directory):
        texture_files.extend([os.path.join(root, x) for x in files if '.png' in x])
    im = []
    for image_path in texture_files:
        im.append(np.array(Image.open(image_path, 'r'))[::-1, :, :])
    image_data = np.stack(im)
    tex_array = TextureArray('creature_textures', image_data, im[0].shape[0], im[0].shape[1], len(im), im[0].shape[2])
    return tex_array

ALIVE = 0
ID = 1
BASE_POS_X = 2
BASE_POS_Y = 3
BASE_ROT = 4
AGE = 5
MAX_AGE = 6
SPEED = 7
AGGR = 8
MOJO = 9
FERT = 10
INT_COOL = 11

# ALIVE
# ID
# BASE_POS_X
# BASE_POS_Y
# BASE_ROT
POS_X = 5
POS_Y = 6
ROT = 7

square = np.array([[-1.0,  1.0, 0.0, 1.0],  # X, Y, S, T
                   [-1.0, -1.0, 0.0, 0.0],
                   [ 1.0,  1.0, 1.0, 1.0],
                   [ 1.0, -1.0, 1.0, 0.0]], dtype=np.float32)


CREATURE_DATA = [{'name': 'creature_position',
                  'index': 2,
                  'components': 3,
                  'gl_type': GL.GL_FLOAT,
                  'normalized': GL.GL_FALSE,
                  'stride': 5*4,
                  'offset': ctypes.c_void_p(0),
                  'divisor': 1},
                 {'name': 'creature_texture',
                  'index': 3,
                  'components': 2,
                  'gl_type': GL.GL_FLOAT,
                  'normalized': GL.GL_FALSE,
                  'stride': 5*4,
                  'offset': ctypes.c_void_p(3*4),
                  'divisor': 1}]


class Culture(Mesh):
    def __init__(self, max_creatures=4000):
        super().__init__()

        self.max_creatures = max_creatures

        #self.creature_data = np.random.randint(0, 10, self.max_creatures).astype(np.float32)
        self.creature_data = np.random.random((self.max_creatures, 5)).astype(np.float32)*20.0 - 10.0
        self.creature_data[:, 2] = np.random.random(self.max_creatures).astype(np.float32)*2*np.pi - np.pi
        self.creature_data[:, 3] = np.random.randint(0, 10, self.max_creatures).astype(np.float32)
        self.creature_data[:, 4] = np.random.randint(0, 4, self.max_creatures).astype(np.float32)/4.0
        #self.creature_data = np.array([[10.0, 10.0],
        #                               [10.0, -10.0],
        #                               [-10.0, 10.0],
        #                               [-10.0, -10.0]], dtype=np.float32)

        self.shader = boa_gfx.gl_shader.shader_manager.get_shader('creature.shader')
        self.shader.bind()
        image_loc = self.shader.uniform_location('image')
        GL.glUniform1i(image_loc, 0)
        self.shader.unbind()

        self.texture = boa_gfx.gl_texture.texture_manager.get_texture('molli2.png')

        self.texture_array = create_texture_array('./128x128/')
        print(self.texture_array.layers)

        self.quad_vbo = VBO(square)
        self.creature_vbo = VBO(self.creature_data, usage=GL.GL_DYNAMIC_DRAW)
        self.vao = VAO()
        self.vao.add_vbo(self.quad_vbo, VAO.TEXTURED_DATA_2D)
        self.vao.add_vbo(self.creature_vbo, CREATURE_DATA)

        self.scene.drawable.append(self)
        # self.creature_data = np.zeros((12, max_creatures))
        # self.creature_parts = np.zeros((8, 5*max_creatures))

        # self.texture = create_texture_array('./textures/128x128')

    def update(self, dt):
        self.creature_data[:, 2] += dt
        self.creature_vbo.update_data(self.creature_data, 0)

    def draw(self):
        self.shader.m_matrix = self.m_matrix

        self.texture_array.bind()
        self.shader.bind()
        self.vao.bind()

        GL.glDrawArraysInstanced(GL.GL_TRIANGLE_STRIP, 0, 4, self.creature_data.shape[0])

        self.vao.unbind()
        self.shader.unbind()
        self.texture_array.unbind()
