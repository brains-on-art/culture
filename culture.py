import time
import numpy as np

from OpenGL import GL

import boa_gfx
from boa_gfx import Mesh, VBO, VAO
from boa_gfx.gl_texture import TextureArray, Texture
from boa_gfx.gl_particle_system import ParticleSystem
import ctypes

import SharedArray as sa


square = np.array([[-1.0,  1.0, 0.0, 1.0],  # X, Y, S, T
                   [-1.0, -1.0, 0.0, 0.0],
                   [ 1.0,  1.0, 1.0, 1.0],
                   [ 1.0, -1.0, 1.0, 0.0]], dtype=np.float32)


CREATURE_DATA = [{'name': 'creature_position',
                  'index': 2,
                  'components': 4,
                  'gl_type': GL.GL_FLOAT,
                  'normalized': GL.GL_FALSE,
                  'stride': 12*4,
                  'offset': ctypes.c_void_p(0),
                  'divisor': 1},
                 {'name': 'creature_texture',
                  'index': 3,
                  'components': 4,
                  'gl_type': GL.GL_FLOAT,
                  'normalized': GL.GL_FALSE,
                  'stride': 12*4,
                  'offset': ctypes.c_void_p(4*4),
                  'divisor': 1},
                 {'name': 'beat_swirl',
                  'index': 4,
                  'components': 4,
                  'gl_type': GL.GL_FLOAT,
                  'normalized': GL.GL_FALSE,
                  'stride': 12*4,
                  'offset': ctypes.c_void_p(8*4),
                  'divisor': 1}]


def creature_system():
    static_data = square
    static_data_attributes = VAO.TEXTURED_DATA_2D

    instance_data = sa.attach('creature_parts')
    instance_data_attributes = CREATURE_DATA

    shader = boa_gfx.gl_shader.shader_manager.get_shader('creature.shader')
    texture = TextureArray.from_directory('./128x128/')

    system = ParticleSystem(static_data, static_data_attributes,
                            instance_data, instance_data_attributes,
                            shader, texture)
    return system

FOOD_DATA = [{'name': 'food_position',
              'index': 2,
              'components': 4,
              'gl_type': GL.GL_FLOAT,
              'normalized': GL.GL_FALSE,
              'stride': 4 * 4,
              'offset': ctypes.c_void_p(0),
              'divisor': 1}]


def food_system():
    static_data = square
    static_data_attributes = VAO.TEXTURED_DATA_2D

    instance_data = sa.attach('food_gfx')
    instance_data_attributes = FOOD_DATA

    shader = boa_gfx.gl_shader.shader_manager.get_shader('food.shader')
    texture = Texture.from_file('./textures/food/food.png')

    system = ParticleSystem(static_data, static_data_attributes,
                            instance_data, instance_data_attributes,
                            shader, texture)
    return system

ANIMATION_DATA = [{'name': 'animation_position',
                   'index': 2,
                   'components': 4,
                   'gl_type': GL.GL_FLOAT,
                   'normalized': GL.GL_FALSE,
                   'stride': 12 * 4,
                   'offset': ctypes.c_void_p(0),
                   'divisor': 1},
                  {'name': 'animation_param1',
                   'index': 3,
                   'components': 4,
                   'gl_type': GL.GL_FLOAT,
                   'normalized': GL.GL_FALSE,
                   'stride': 12 * 4,
                   'offset': ctypes.c_void_p(4 * 4),
                   'divisor': 1},
                  {'name': 'animation_param1',
                   'index': 4,
                   'components': 4,
                   'gl_type': GL.GL_FLOAT,
                   'normalized': GL.GL_FALSE,
                   'stride': 12 * 4,
                   'offset': ctypes.c_void_p(8 * 4),
                   'divisor': 1}]


def animation_system():
    static_data = square
    static_data_attributes = VAO.TEXTURED_DATA_2D

    instance_data = sa.attach('animation_gfx')
    instance_data_attributes = ANIMATION_DATA

    shader = boa_gfx.gl_shader.shader_manager.get_shader('animation.shader')
    texture = TextureArray.from_directory('./textures/animations')

    system = ParticleSystem(static_data, static_data_attributes,
                            instance_data, instance_data_attributes,
                            shader, texture)
    return system

#class Animations(Mesh):
#    def __init__(self):
#        super().__init__()
#
#        self.animation_parts = np.zeros((9, 8), dtype=np.float32)
#        a = [-6.0, 0.0, 6.0]
#        x, y = np.meshgrid(a, a)
#        #x += 2.0
#        # POS_X_Y
#        self.animation_parts[:, :2] = np.vstack([x.flatten(), y.flatten()]).T
#        # ROT
#        self.animation_parts[:, 2] = 2*np.random.rand(9)*np.pi
#        # SCALE
#        self.animation_parts[:, 3] = 2.0
#        # NUM FRAMES
#        self.animation_parts[:, 4] = 13.0
#        # FRAME START TIME
#        self.animation_parts[:, 5] = 1.0
#        # FRAME DURATION
#         self.animation_parts[:, 6] = 0.05
#
#         self.shader = boa_gfx.gl_shader.shader_manager.get_shader('animation.shader')
#         self.shader.bind()
#         self.shader.t_loc = self.shader.uniform_location('t')
#         GL.glUniform1f(self.shader.t_loc, 2.0)
#         self.shader.image_loc = self.shader.uniform_location('image')
#         GL.glUniform1i(self.shader.image_loc, 0)
#         self.shader.unbind()
#         #print(type(self.shader))
#
#         self.texture_array = TextureArray.from_directory('./birth/')
#         print('Animation frames:', self.texture_array.layers)
#
#         self.quad_vbo = VBO(square)
#         self.animation_vbo = VBO(self.animation_parts, usage=GL.GL_DYNAMIC_DRAW)
#         self.vao = VAO()
#         self.vao.add_vbo(self.quad_vbo, VAO.TEXTURED_DATA_2D)
#         self.vao.add_vbo(self.animation_vbo, ANIMATION_DATA)
#
#         self.scene.drawable.append(self)
#
#     def update(self, dt):
#         self.animation_vbo.update_data(self.animation_parts, 0)
#
#     def draw(self):
#         #print(self.shader.vp_matrix)
#         t = time.perf_counter()
#         self.shader.bind()
#         GL.glUniform1f(self.shader.t_loc, t)
#         self.shader.unbind()
#         self.shader.m_matrix = self.m_matrix
#
#         self.texture_array.bind()
#         self.shader.bind()
#         self.vao.bind()
#
#         GL.glDrawArraysInstanced(GL.GL_TRIANGLE_STRIP, 0, 4, self.animation_parts.shape[0])
#
#         self.vao.unbind()
#         self.shader.unbind()
#         self.texture_array.unbind()
#
#
