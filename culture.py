import numpy as np

from OpenGL import GL

import boa_gfx
from boa_gfx import VAO
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
    texture = TextureArray.from_directory('./textures/creatures')

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