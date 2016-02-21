import numpy as np
import sdl2

from boa_gfx.core import SpatialObject
from boa_gfx.gl_mesh import TexturedTriangleStrip
from boa_gfx.transformer import linear, sinusoidal, interpolate


class Culture(object):
    def __init__(self, event_manager):
        self.event_manager = event_manager
        self.creatures = []

        self.event_manager.add_keydown_callback(sdl2.SDLK_RETURN, lambda event: self.add_creature())

    def add_creature(self):
        #print('added')
        creature_types = [HaloCreature, WormCreature]
        c = np.random.choice(creature_types)()
        starting_position = (np.random.rand(3)*2-1)*13
        starting_position[2] = 0.01
        c.position = starting_position
        c.scale = 0.9
        self.creatures.append(c)

    def update(self):
        for c in self.creatures:
            c.update()


class Creature(SpatialObject):
    def __init__(self):
        super().__init__()
        self.movement = interpolate(self, 'position', self.position, self.position+(1.0, 0.0, 0.0), run=False)

    def update(self):
        if not self.movement.running:
            angle = np.random.rand() * 2 * np.pi
            x = np.cos(angle)
            y = np.sin(angle)
            self.movement = interpolate(self, 'position', self.position, self.position+(x, y, 0.0), duration=3.0)


class HaloCreature(Creature):
    def __init__(self, body_texture='halo1.png', halo_texture='halo1.png'):
        super().__init__()

        self.body = TexturedTriangleStrip(texture_name=body_texture, parent=self)
        self.body_rotation = linear(self.body, 'z_rotation', speed=-0.1)
        self.body_pulse = sinusoidal(self.body, 'scale', amplitude=0.03, frequency=3.0, phase=(0.0, 1.5, 0.0))

        self.halo = TexturedTriangleStrip(texture_name=halo_texture, parent=self)
        self.halo.scale = 0.6
        self.halo_rotation = linear(self.halo, 'z_rotation', speed=0.5)

    #@SpatialObject.position.setter
    #def position(self, new_position):
    #    new_position = np.atleast_1d(new_position)
    #    self._translation_matrix[:3, 3] = new_position

        #self.body.position = new_position
        #self.halo.position = new_position


class WormCreature(Creature):
    def __init__(self, nose_texture='meduusa1.png', body_texture='meduusa2.png', tail_texture='meduusa3.png'):
        super().__init__()

        self.tail = TexturedTriangleStrip(texture_name=tail_texture, parent=self)
        self.tail.scale = 0.5
        self.tail.position = (0.0, -1.2, 0.0)
        self.tail_move = sinusoidal(self.tail, 'position', amplitude=(0.0, 0.1, 0.0))

        self.body = TexturedTriangleStrip(texture_name=body_texture, parent=self)
        self.body.scale = 0.5
        self.body.position = (0.0, -0.6, 0.0)
        self.body_move = sinusoidal(self.body, 'position', amplitude=(0.0, 0.1, 0.0), phase=(0.0, 1.2, 0.0))

        self.nose = TexturedTriangleStrip(texture_name=nose_texture, parent=self)
        self.nose.scale = 0.5
        #self.nose_move = sinusoidal(self.nose, 'position', amplitude=(0.0, 0.1, 0.0))


