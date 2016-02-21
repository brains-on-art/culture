import os
import numpy as np
from PIL import Image

from boa_gfx.gl_texture import TextureArray


def create_texture_array(texture_directory):
    texture_files = []
    for root,dirs,files in os.walk(texture_directory):
        texture_files.extend([os.path.join(root, x) for x in files if '.png' in x])
    im = []
    for image_path in texture_files:
        im.append(np.array(Image.open(image_path, 'r'))[::-1,:,:])
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

#ALIVE
#ID
#BASE_POS_X
#BASE_POS_Y
#BASE_ROT
POS_X = 5
POS_Y = 6
ROT = 7


class Culture(object):
    def __init__(self, max_creatures):
        self.max_creatures = max_creatures
        self.creatures = np.zeros((12, max_creatures))
        self.creature_parts = np.zeros((8, 5*max_creatures))

        self.texture = create_texture_array('./textures/128x128')


