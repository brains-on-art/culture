import os
import signal
import subprocess
import sys
import time
import zmq

import numba
import numpy as np
import SharedArray as sa
from scipy.spatial.distance import cdist as distm


sys.path.append('./pymunk')
import pymunk as pm

from boa_gfx.time import TimeAware, TimeKeeper, Interpolator

zmq_port = '5556'
max_creatures = 25
max_parts = 5
offscreen_position = (30.0, 30.0, 0.0)

max_food = 100
max_animations = 100

resting_period = 60

@numba.jit
def _find_first(vec, item):
    for i in range(len(vec)):
        if vec[i] == item:
            return i
    return -1


@numba.jit(nopython=True)
def random_circle_point():
    theta = np.random.rand()*2*np.pi
    x,y = 5*np.cos(theta), 5*np.sin(theta)
    return x, y


def create_new_sa_array(name, shape, dtype):
    try:
        sa.delete(name)
    except FileNotFoundError:
        pass
    finally:
        sa_array = sa.create(name, shape, dtype=dtype)
    return sa_array


class Culture(TimeAware):
    def __init__(self):
        # Create creature parts array to share with visualization
        self.creature_gfx = create_new_sa_array('creature_gfx', (max_creatures*max_parts, 12), np.float32)

        #self.creature_parts = create_new_sa_array('creature_parts', (max_creatures*max_parts, 12), np.float32)
        self.creature_parts = np.zeros((max_creatures*max_parts, 12), dtype=np.float32)
        # FIXME: refactor to creature_gfx
        self.creature_parts[:, :3] = offscreen_position  # Off-screen coordinates
        self.creature_parts[:, 3:] = 1.0  # Avoid undefined behavior by setting everything to one

        # Creature data (no position!)
        # self.creature_data = np.zeros((max_creatures, 4))
        # self.creature_data[:, 1] = 100.0  # max_age
        # self.creature_data[:, 3] = 0.5  # creature size
        self.creature_data = np.recarray((max_creatures, ),
                                dtype=[('alive', bool),
                                       ('interactive', bool),
                                       ('max_age', float),
                                       ('age', float),
                                       ('size', float),
                                       ('mood', int),
                                       ('started_colliding', float),
                                       ('ended_interaction', float),
                                       ('agility_base', float),
                                       ('virility_base', float),
                                       ('mojo', float),
                                       ('aggressiveness_base', float),
                                       ('power', float),
                                       ('hunger', float),
                                       ('type', int),
                                       ('color', int),
                                       ('interacting_with', int)])
        self.creature_data.alive = False
        self.creature_data.interactive = 0
        self.creature_data.max_age = 100.0
        self.creature_data.size = 0.5

        root, dirs, files = next(os.walk('./textures/creatures/'))
        png_files = sorted([x for x in files if '.png' in x])
        textures = [x.replace('.png', '').split('_') + [i] for i, x in enumerate(png_files)]
        self.texture_ind = {'jelly': [x[4] for x in textures if x[0] == 'jelly'],
                            'simple': [x[4] for x in textures if x[0] == 'simple'],
                            'sperm': [x[4] for x in textures if x[0] == 'sperm'],
                            'feet': {'fat': {'bottom': [x[4] for x in textures if x[:3] == ['feet', 'fat', 'bottom']],
                                             'top': [x[4] for x in textures if x[:3] == ['feet', 'fat', 'top']]},
                                     'thin': {'bottom': [x[4] for x in textures if x[:3] == ['feet', 'thin', 'bottom']],
                                              'top': [x[4] for x in textures if x[:3] == ['feet', 'thin', 'top']]}}}

        #
        physics_skeleton = {'active': False,
                            'target': None,
                            'body': None,
                            'constraint': None,
                            'shape': None}
        self.creature_physics = [physics_skeleton.copy() for x in range(max_creatures)]

        self.pm_space = pm.Space()
        self.pm_space.damping = 0.4
        # self.pm_space.gravity = 0.0, -1.0

        # Create walls but ignore collisions between creatures
        def ignore_collision(*args):
            return False
        self.pm_space.add_collision_handler(1, 1, begin=ignore_collision)

        num_points = 30
        radius = 13.0
        theta = np.linspace(0, 2 * np.pi, num_points)
        x, y = radius * np.cos(theta), radius * np.sin(theta) # FIXME: skaalaa x täsmäämään altaan kanssa
        x[-1], y[-1] = x[0], y[0]
        walls = [pm.Segment(self.pm_space.static_body, (x[i], y[i]), (x[i+1], y[i+1]), 0.1) for i in range(num_points-1)]
        for wall in walls:
            wall.collision_type = 2
        self.pm_space.add(walls)

        # Create food graphics array to share with visualization
        self.food_gfx = create_new_sa_array('food_gfx', (max_food, 4), np.float32)
        self.food_gfx[:, :3] = offscreen_position # Off-screen coordinates
        self.food_gfx[:, 3:] = 1.0  # Avoid undefined behavior by setting everything to one
        self.next_food = 0

        # Create animation graphics array to share with visualization
        self.animation_gfx = create_new_sa_array('animation_gfx', (max_animations, 12), np.float32)
        self.animation_gfx[:, :3] = offscreen_position  # Off-screen coordinates
        self.animation_gfx[:, 3:] = 1.0  # Avoid undefined behavior by setting everything to one
        self.next_animation = 0

        self.scheduler.enter(3.0, 0.0, self.demo_init)
        #self.demo_init()

        self.prev_update = time.perf_counter()
        self.ct = time.perf_counter()

        # self.dt = p0.0

    def demo_init(self):
        def rand_pos():
            theta = np.random.rand()*2*np.pi
            radius = np.random.rand()*12
            return radius*np.cos(theta), radius*np.sin(theta)

        for i in range(3):
            self.add_creature('jelly', rand_pos())
            self.add_creature('feet', rand_pos())
            self.add_creature('simple', rand_pos())
            self.add_creature('sperm', rand_pos())

        # for i in range(10):
        #     self.add_food(np.random.rand(2)*20.0 - 10.0)

        # for i in range(3):
        #     self.add_animation('birth',
        #                        position=np.random.rand(2) * 20.0 - 10.0,
        #                        rotation=np.random.rand() * 2 * np.pi,
        #                        num_loops=5)
        #     self.add_animation('death',
        #                        position=np.random.rand(2) * 20.0 - 10.0,
        #                        rotation=np.random.rand() * 2 * np.pi,
        #                        num_loops=5)
        #     self.add_animation('contact',
        #                        position=np.random.rand(2) * 20.0 - 10.0,
        #                        rotation=np.random.rand() * 2 * np.pi,
        #                        num_loops=5)
        #     self.add_animation('fight',
        #                        position=np.random.rand(2) * 20.0 - 10.0,
        #                        rotation=np.random.rand() * 2 * np.pi,
        #                        num_loops=5)
        #     self.add_animation('reproduction',
        #                        position=np.random.rand(2) * 20.0 - 10.0,
        #                        rotation=np.random.rand() * 2 * np.pi,
        #                        num_loops=5)

    def get_texture(self, group, variant=None, part=None):
        if group == 'jelly':
            return np.random.choice(self.texture_ind['jelly'])
        elif group == 'simple':
            return np.random.choice(self.texture_ind['simple'])
        elif group == 'sperm':
            return np.random.choice(self.texture_ind['sperm'])
        elif group == 'feet':
            if variant is None:
                variant = np.random.choice(['fat', 'thin'])
            if part is None:  # Return both
                return [np.random.choice(self.texture_ind['feet'][variant]['bottom']),
                        np.random.choice(self.texture_ind['feet'][variant]['top'])]
            else:
                return np.random.choice(self.texture_ind['feet'][variant][part])
        else:
            print('get_texture: no group named:', group)
            return None

    def add_creature(self, type, position, data=None):
        print('Adding creature ({}) at position {}'.format(type, position))
        print('Using data: ', data)

        index = _find_first(self.creature_data['alive'], False)
        if index == -1: # Creature data is full
            print('Creature data is full, instakilling oldest creature')
            index = self.creature_data['age'].argmax()
            self.remove_creature(index)

        cp = self.creature_physics[index]

        cp['target'] = pm.Body(10.0, 10.0)
        cp['target'].position = position
        cp['target'].position += (0.0, 10.0)

        cp['body'] = [pm.Body(10.0, 5.0) for x in range(max_parts)]
        for i in range(max_parts):
            cp['body'][i].position = (30.0 + 20.0*index + 2*i, 30.0 + 20.0*index + 2*i)

        head = cp['body'][0]
        head_offset = pm.Vec2d((0.0, 0.4))
        cp['constraint'] = [pm.constraint.DampedSpring(head, cp['target'], head_offset, (0.0, 0.0), 0.0, 10.0, 15.0)]

        shape = pm.Circle(head, self.creature_parts[index * max_parts, 3])  # use the scale of the first creature part
        shape.collision_type = 1
        cp['shape'] = shape

        creature_function = {'jelly': self.add_jelly,
                             'feet': self.add_feet,
                             'simple': self.add_simple,
                             'sperm': self.add_sperm}[type]
        creature_function(index, position)

        #{'mojo': 4, 'max_age': 0.18328243481718498, 'agressiveness_base': 1, 'power': 32215.83847390449, 'virility_base': 183.04049763745167, 'slave_id': 1, 'agility_base': 139.6109314200933, 'type': 0.0}
        data = data if data is not None else {}
        self.creature_data[index]['max_age'] = data['max_age'] if ('max_age' in data) and (data['max_age'] is not None) else np.random.random(1) * 180 + 180
        self.creature_data[index]['alive'] = True

        self.creature_data[index]['age'] = 0
        self.creature_data[index]['size'] = 0.5
        self.creature_data[index]['mood'] = 1
        self.creature_data[index]['started_colliding'] = 0.0
        self.creature_data[index]['ended_interaction'] = 0.0
        self.creature_data[index]['agility_base'] = data['agility_base'] if ('agility_base' in data) and (data['agility_base'] is not None) else np.random.random()
        self.creature_data[index]['virility_base'] = data['virility_base'] if ('virility_base' in data) and (data['virility_base'] is not None) else np.random.random()
        self.creature_data[index]['mojo'] = data['mojo'] if ('mojo' in data) and (data['mojo'] is not None) else np.random.random()
        self.creature_data[index]['aggressiveness_base'] = data['agressiveness_base'] if ('aggressiveness_base' in data) and (data['agressiveness_base'] is not None) else np.random.random()
        self.creature_data[index]['power'] = data['power'] if ('power' in data) and (data['power'] is not None) else np.random.random()
        self.creature_data[index]['hunger'] = 0.5
        # self.creature_data[index]['color']
        self.creature_data[index]['interacting_with'] = -1

        f = lambda: self.activate_creature_physics(index)
        g = lambda: self.set_interactive(index)

        self.scheduler.enter(4.0, 0.0, f)
        self.scheduler.enter(5.0, 0.0, g)

        self.add_animation('birth', position, scale=1.5, relative_start_time=3.0)

    def set_interactive(self, index, value=True):
        self.creature_data[index]['interactive'] = value

    def activate_creature_physics(self, index):
        cp = self.creature_physics[index]

        self.pm_space.add(cp['shape'])
        self.pm_space.add(cp['body'])
        self.pm_space.add(cp['constraint'])

        cp['active'] = True

    def deactivate_creature_physics(self, index):
        cp = self.creature_physics[index]

        if cp['active']:
            self.pm_space.remove(cp['shape'])
            self.pm_space.remove(cp['body'])
            self.pm_space.remove(cp['constraint'])

            cp['active'] = False

    def add_jelly(self, index, position):
        print('Creating jelly at index {}'.format(index))
        self.creature_data[index]['type'] = 1  # JELLY

        cp = self.creature_physics[index]
        head, mid, tail = cp['body'][0:3]
        head.position = position
        mid.position = head.position + (0.0, -0.3)
        tail.position = head.position + (0.0, -0.6)

        cp['constraint'] += [pm.constraint.SlideJoint(head, mid, (0.0, -0.1), (0.0, 0.1), 0.1, 0.5),
                             pm.constraint.RotaryLimitJoint(head, mid, -0.5, 0.5),
                             pm.constraint.SlideJoint(mid, tail, (0.0, -0.1), (0.0, 0.1), 0.1, 0.5),
                             pm.constraint.RotaryLimitJoint(mid, tail, -0.5, 0.5)]


        animation_vec = [0.0, 1.0, 1.0, 1.0]  # Animation time offset, beat frequency, swirl radius, swirl frequency
        for i in range(3):
            position_vec = [position[0], position[1]-0.3*(i+1), 0.0, 0.5]  # Position, rotation, scale
            texture_vec = [self.get_texture('jelly'), 0.0, 1.0, 1.0]  # Texture index, color rotation, saturation, alpha
            self.creature_parts[max_parts*index+i, :] = position_vec + texture_vec + animation_vec

        Interpolator.add_interpolator(self.creature_parts,
                                      np.s_[max_parts*index:max_parts*index+3, 3],
                                      [0.0, 1.0],
                                      [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])

    def add_feet(self, index, position):
        print('Creating feet at index {}'.format(index))
        self.creature_data[index]['type'] = 2

        cp = self.creature_physics[index]

        top, bottom = cp['body'][:2]
        top.position = position
        bottom.position = position

        cp['constraint'] += [pm.constraint.PivotJoint(top, bottom, (0.0, 0.0), (0.0, 0.0)),
                             pm.constraint.GearJoint(top, bottom, 0.0, 1.0)]

        position_vec = [position[0], position[1], 0.0, 0.5]  # Position, rotation, scale
        animation_vec = [0.0, 1.0, 1.0, 1.0]  # Animation time offset, beat frequency, swirl radius, swirl frequency
        tex = self.get_texture('feet')
        texture_vec = [tex[1], 0.0, 1.0, 1.0]
        self.creature_parts[max_parts * index, :] = position_vec + texture_vec + animation_vec
        texture_vec = [tex[0], 0.25, 1.0, 0.01]
        self.creature_parts[max_parts * index + 1, :] = position_vec + texture_vec + animation_vec

        Interpolator.add_interpolator(self.creature_parts,
                                      np.s_[max_parts * index:max_parts * index + 2, 3],
                                      [0.0, 1.0],
                                      [[0.0, 0.0], [0.5, 0.5]])


    def add_simple(self, index, position):
        print('Creating simple at {} (index {})'.format(position, index))
        self.creature_data[index]['type'] = 3

        cp = self.creature_physics[index]

        head = cp['body'][0]
        head.position = position

        position_vec = [position[0], position[1], 0.0, 0.5]  # Position, rotation, scale
        animation_vec = [0.0, 1.0, 1.0, 1.0]  # Animation time offset, beat frequency, swirl radius, swirl frequency
        texture_vec = [self.get_texture('simple'), 0.0, 1.0, 1.0]
        self.creature_parts[max_parts*index, :] = position_vec + texture_vec + animation_vec

        Interpolator.add_interpolator(self.creature_parts,
                                      np.s_[max_parts * index, 3],
                                      [0.0, 1.0],
                                      [[0.0], [0.5]])

    def add_sperm(self, index, position):
        print('Creating sperm at index {}'.format(index))
        self.creature_data[index]['type'] = 4

        cp = self.creature_physics[index]

        for i in range(5):
            cp['body'][i].position = position
            cp['body'][i].position += (0.0, -0.5*i)

        for i in range(4):
            a = cp['body'][i]
            b = cp['body'][i+1]
            cp['constraint'].append(pm.constraint.SlideJoint(a, b, (0.0, -0.3), (0.0, 0.3), 0.1*(0.8**i), 0.2*(0.8**i)))
            cp['constraint'].append(pm.constraint.RotaryLimitJoint(a, b, -0.5, 0.5))

        position_vec = [position[0], position[1], 0.0, 0.5]  # Position, rotation, scale
        animation_vec = [0.0, 1.0, 1.0, 1.0]  # Animation time offset, beat frequency, swirl radius, swirl frequency
        tex_head, tex_tail = self.get_texture('sperm'), self.get_texture('sperm')
        texture_vec = [tex_head, 0.0, 1.0, 1.0]  # Texture index, color rotation, saturation, alpha
        self.creature_parts[max_parts * index, :] = position_vec + texture_vec + animation_vec
        for i in range(4):
            position_vec = [position[0], position[1]-0.5*(i+1), 0.0, 1.0]
            texture_vec = [tex_tail, 0.25, 1.0, 1.0]  # Texture index, color rotation, saturation, alpha
            self.creature_parts[max_parts * index + (i+1), :] = position_vec + texture_vec + animation_vec

        Interpolator.add_interpolator(self.creature_parts,
                                      np.s_[max_parts * index:max_parts * index + 5, 3],
                                      [0.0, 1.0],
                                      [[0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.25, 0.25*0.8, 0.25*0.8**2, 0.25*0.8**2]])

    def add_animation(self, type, position, rotation=None, scale=1.0, relative_start_time=0.0, num_loops=1):
        #print('Adding animation {} at position {}'.format(type, position))
        # Add animation to next slot
        index = self.next_animation

        # print('Adding {} animation at {} (index {})'.format(type, position, index))
        alpha_frame = omega_frame = 0.0

        # Get animation specific parameters
        if type == 'birth':
            alpha_frame = 1.0
            start_frame, end_frame = 1.0, 16.0 # FIXME: halutaanko kovakoodata nämä
            loop_time = 1.0
            Interpolator.add_interpolator(self.animation_gfx,
                                          np.s_[index, 3],
                                          [0.0, 1.0],
                                          [[0.0], [1.0]])
        elif type == 'contact':
            start_frame, end_frame = 17.0, 34.0
            loop_time = 2.0
        elif type == 'death':
            scale *= 1.5
            start_frame, end_frame = 35.0, 49.0
            loop_time = 3.0
        elif type == 'fight':
            start_frame, end_frame = 50.0, 67.0
            loop_time = 2.0
        elif type == 'reproduction':
            start_frame, end_frame = 68.0, 85.0
            loop_time = 2.0
        else:
            return None

        # Calculate absolute start time
        start_time = time.perf_counter() + relative_start_time

        # Randomize rotation by default
        if rotation is None:
            rotation = np.random.rand()*2*np.pi

        # Construct attribute vectors (matches GLSL)
        position_vec = [position[0], position[1], rotation, scale]
        param1_vec = [start_frame, end_frame, start_time, loop_time]
        param2_vec = [num_loops, alpha_frame, omega_frame]

        # Add animation data to shared array
        self.animation_gfx[index, :11] = position_vec + param1_vec + param2_vec

        if type in ['contact', 'fight', 'reproduction']:
            old_index = []

            # Add a second copy with time shift
            param1_vec[2] += loop_time / 3.0
            old_index.append(index)
            index = index + 1 if index < max_animations - 1 else 0
            # print('Adding {} animation at {} (index {})'.format(type, position, index))
            self.animation_gfx[index, :11] = position_vec + param1_vec + param2_vec

            param1_vec[2] += loop_time / 3.0
            old_index.append(index)
            index = index + 1 if index < max_animations - 1 else 0
            # print('Adding {} animation at {} (index {})'.format(type, position, index))
            self.animation_gfx[index, :11] = position_vec + param1_vec + param2_vec

            self.next_animation = index + 1 if index < max_animations - 1 else 0
            old_index.append(index)
            index = old_index
        else:
            # Advance to next animation array position
            self.next_animation = index + 1 if index < max_animations - 1 else 0

        return index

    def add_food(self, position, rotation=None):
        # Add food to next slot
        index = self.next_food

        print('Adding food at {} (index {})'.format(position, index))

        # Randomize rotation by default
        if rotation is None:
            rotation = np.random.rand() * 2 * np.pi

        # Construct attribute vectors (matches GLSL)
        position_vec = [position[0], position[1], rotation, 0.125]

        # Add animation data to shared array
        self.food_gfx[index, :] = position_vec

        # Advance to next food array position
        self.next_food += 1
        if self.next_food >= max_food:
            self.next_food = 0

        return index

    def remove_creature(self, index):
        print('Removing creature at index {}'.format(index))
        self.deactivate_creature_physics(index)

        cp = self.creature_physics[index]
        cp['target'] = None
        cp['body'] = None
        cp['constraint'] = None
        cp['shape'] = None

        self.creature_data[index] = np.zeros(len(self.creature_data.dtype.names))
        self.creature_data[index]['max_age'] = 1.0

        def f():
            self.creature_parts[index*max_parts:(index+1)*max_parts-1, :3] = offscreen_position

        self.scheduler.enter(8.0, 0.0, f)

    def kill_creature(self, index):
        print('Killing creature at index', index)
        self.set_interactive(index, False)

        creature_type = self.creature_data[index]['type']
        if creature_type == 1:  # JELLY
            parts = 3
        elif creature_type == 2:  # FEET
            parts = 2
        elif creature_type == 3:  # SIMPLE
            parts = 1
        elif creature_type == 4:  # SPERM
            parts = 5
        else:
            print('Unknown creature type:', creature_type)

        values = np.tile(self.creature_parts[max_parts * index:max_parts * index + parts, :8], (3, 1, 1))
        # During first second set saturation to zero and alpha to 0.5
        values[1, :, 6:] = (0.0, 0.5)  # saturation + alpha
        # During second second set alpha to zero, move parts to head, and scale to zero
        values[2, :, 6:] = (0.0, 0.0)  # saturation + alpha
        values[2, :, :2] = values[0, 0, :2]  # position
        values[2, :, 3] = 0.0  # scale

        Interpolator.add_interpolator(self.creature_parts,
                                      np.s_[max_parts * index:max_parts * index +parts, :8],
                                      [0.0, 1.0, 2.0],
                                      values)

        self.add_animation('death', self.creature_parts[max_parts*index,:2], relative_start_time=2.0)

        f = lambda: self.remove_creature(index)

        self.scheduler.enter(2.1, 0.0, f)

    def update(self, dt):
        self.ct = time.perf_counter()
        # if self.ct - self.prev_update > 5.0:
        #     index = np.random.randint(max_creatures)
        #     position = tuple(np.random.rand(2)*20.0 - 10.0)
        #     self.add_jelly(index, position)
        #     self.prev_update = self.ct

            #i = np.random.randint(0, max_creatures)
            #self.pm_target[i].position = tuple(np.random.random(2)*20.0 - 10.0)

        # Update creature age...
        alive = self.creature_data['alive']
        max_age = self.creature_data['max_age']
        cur_age = self.creature_data['age']
        cur_age[:] += dt
        # ...and compute the other dynamic params
        hunger = self.creature_data['hunger']
        hunger[:] += dt / 5
        self.creature_data['hunger'] = np.clip(hunger, 0, 1)
        # agility = self.creature_data['agility_base'] * (1 - (self.creature_data['age'] / self.creature_data['max_age']))
        # succulence = 1 - hunger
        # aggr = self.creature_data['aggressiveness_base'] + self.creature_data['hunger']
        # virility = self.creature_data['virility_base'] + 1 - self.creature_data['hunger']
        mood = self.creature_data['mood']
        creature_type = self.creature_data['type']
        started_colliding = self.creature_data['started_colliding']
        interacting_with = self.creature_data['interacting_with']
        last_interacted = self.creature_data['ended_interaction']

        # Update appearance changes from aging and remove dead creatures
        self.creature_parts[:, 6] = np.clip(1.0 - (cur_age / max_age), 0.0, 1.0).repeat(max_parts)
        self.creature_parts[:, 7] = np.clip(1.0 - (cur_age - max_age)/3.0, 0.0, 1.0).repeat(max_parts)
        dead_creatures = alive & (cur_age > max_age + 3.0)
        for ind in np.where(dead_creatures)[0]:
            self.remove_creature(ind)

        # Find colliding creatures and deal with them
        # time1 = time.perf_counter()
        collisions, distances, radii = self.get_collision_matrix()
        # time2 = time.perf_counter()
        # print((time2 - time1) * 1000)

        # see which rows have collisions...
        ids_in_collision = np.nonzero([sum(i) for i in collisions])[0]
        if ids_in_collision.any():
            # ...do stuff if there are some
            for i in ids_in_collision:
                if mood[i] != 1: # only go forward if i'm available
                    continue
                distances[i,i] = np.inf # make sure not to choose yourself
                other = np.argmin(distances[i]) # and then choose the closest one
                self.creature_parts[i*max_parts, 7] = 0.2

                if self.can_start_interaction([i,other]):
                    # print('{} found to be colliding with {}, with distance {} (radii sum {})'.format(i, other, distances[i,other], radii[i,other]))
                    mood[[i,other]] = 0
                    interacting_with[i] = other
                    interacting_with[other] = i
                    started_colliding[[i, other]] = self.ct

        # Handle behaviours
        for i in range(max_creatures):
            cp = self.creature_physics[i]
            if not alive[i]:
                # Dead creatures don't need to move...
               continue

            if mood[i] == 0:
                # just started colliding
                self.start_interaction(i, interacting_with[i])
                # so go to next state and fire animation

            elif mood[i] == -1:
                # already in an interaction
                if (self.ct - started_colliding[i]) < 5:
                    # not yet done with dancing
                    # map(lambda b: b.reset_forces(), cp['body'])
                    # for body in cp['body']:
                        # body.velocity = (0, 0)
                    self.deactivate_creature_physics(i)
                else:
                    self.end_interaction(i, interacting_with[i])

            elif mood[i] == 1:
                # Default mood, move creatures according to their type
                if creature_type[i] == 1:
                    if cp['body'][0].velocity.get_length() < 0.7:
                        cp['target'].position += random_circle_point()
                #    for j in range(3):
                #        self.creature_parts[3*i+j, :2] = tuple(self.pm_body[i][j].position)
                #        self.creature_parts[3*i+j, 2] = self.pm_body[i][j].angle


        # Advance the physics simulation and sync physics with graphics
        self.pm_space.step(dt)

        #t1 = time.perf_counter()
        mask = np.array([creature['active'] for creature in self.creature_physics]).repeat(max_parts)
        positions = np.array([[(body.position.x, body.position.y, -body.angle) for body in creature['body']]
                              for creature in self.creature_physics if creature['active']]).reshape(mask.sum(), 3)
        self.creature_parts[mask, :3] = positions
        self.creature_gfx[:,:] = self.creature_parts[::-1, :]
        #t2 = time.perf_counter()
        #print(t2-t1)

        #t1 = time.perf_counter()
        #positions = [[(body.position.x, body.position.y, -body.angle) for body in creature['body']]
        #             if creature['active']
        #             else [offscreen_position for x in range(max_parts)]
        #             for creature in self.creature_physics]
        #positions = np.array(positions).reshape(max_creatures*max_parts, 3)
        #self.creature_parts[:, :3] = positions
        #print(t2-t1)

    def can_start_interaction(self, pair):
        mood = self.creature_data['mood'][pair]
        alive = self.creature_data['alive'][pair]
        interactive = self.creature_data['interactive'][pair]
        #last_interacted = self.creature_data['ended_interaction']
        return (mood == 1).all() and alive.all() and interactive.all()  #((self.ct - last_interacted[arr_like]) > resting_period).all()

    def start_interaction(self, a, b):
        alive = self.creature_data['alive'][[a,b]]
        if not alive.all() or b == -1:
            return
        mood = self.creature_data['mood']
        checks = {a: {'aggr': self.aggr_check(a),
                    'virility': self.virility_check(a)},
                    b: {'aggr': self.aggr_check(b),
                    'virility': self.virility_check(b)}}
        
        # is b -1?

        # set animations immediately depending on aggression levels
        for i in [a,b]:
            anim = 'contact'
            if checks[i]['aggr']:
                # anim = 'fight'
                self.add_animation('fight',
                                   position=(self.creature_physics[i]['body'][0].position.x, self.creature_physics[i]['body'][0].position.y),
                                   rotation=np.random.rand() * 2 * np.pi,
                                   num_loops=1)
            if checks[i]['virility']:
                # anim = 'reproduction'
                self.add_animation('reproduction',
                                   position=(self.creature_physics[i]['body'][0].position.x, self.creature_physics[i]['body'][0].position.y),
                                   rotation=np.random.rand() * 2 * np.pi,
                                   num_loops=1,
                                   relative_start_time=0.08)
            if not (checks[i]['aggr'] or checks[i]['virility']):
                self.add_animation('contact',
                                   position=(self.creature_physics[i]['body'][0].position.x, self.creature_physics[i]['body'][0].position.y),
                                   rotation=np.random.rand() * 2 * np.pi,
                                   num_loops=1)
            mood[i] = -1

    def end_interaction(self, a, b):
        alive = self.creature_data['alive'][[a,b]]
        if not alive.all() or b == -1:
            return
        mood = self.creature_data['mood']
        checks = {a: {'aggr': self.aggr_check(a),
                    'virility': self.virility_check(a)},
                    b: {'aggr': self.aggr_check(b),
                    'virility': self.virility_check(b)}}

        # is b -1?

        # CASES:
        # 1. Both want a fight
        if checks[a]['aggr'] and checks[b]['aggr']:
            winner = self.power_check(a, b)
            loser = a if winner == b else b
            print('both wanted a fight: {} killed {}'.format(winner, loser))
            # self.add_animation('death',
            #                    position=(self.creature_physics[loser]['body'][0].position.x, self.creature_physics[loser]['body'][0].position.y),
            #                    rotation=np.random.rand() * 2 * np.pi,
            #                    num_loops=1)

            self.kill_creature(loser)
            #x = self.creature_physics[loser]['body'][0].position.x
            #y = self.creature_physics[loser]['body'][0].position.y
            #def play_animation(x,y):
            #    self.add_animation('death',
            #                   position=(x, y),
            #                   rotation=np.random.rand() * 2 * np.pi,
            #                   num_loops=1)
            #self.scheduler.enter(1, 0.0, play_animation, (x,y))

            #self.creature_data['hunger'][winner] = 0
            #self.remove_creature(loser)
            #self.activate_creature_physics(winner)
            #mood[winner] = 1

        # 2. One wants to fight, other wants to run away
        elif checks[a]['aggr'] or checks[b]['aggr']:
            # aggr = self.creature_data['aggressiveness_base'] + self.creature_data['hunger']
            aggressor = a if checks[a]['aggr'] else b
            escaper = b if aggressor == a else a
            escaped = self.escape_attempt(escaper, aggressor)
            if escaped:
                # mood[[a,b]] = 1
                # self.creature_data[[a,b]]['ended_interaction'] = self.ct
                # print('{} got away'.format(escaper))
                map(self.activate_creature_physics, [a,b])
                mood[[a,b]] = 1
                print('{} wanted to fight {}, but it got away'.format(aggressor, escaper))
            else:
                # self.add_animation('death',
                #                    position=(self.creature_physics[escaper]['body'][0].position.x, self.creature_physics[escaper]['body'][0].position.y),
                #                    rotation=np.random.rand() * 2 * np.pi,
                #                    num_loops=1)
                #x = self.creature_physics[escaper]['body'][0].position.x
                #y = self.creature_physics[escaper]['body'][0].position.y
                #def play_animation(x,y):
                #    self.add_animation('death',
                #                   position=(x, y),
                #                   rotation=np.random.rand() * 2 * np.pi,
                #                   num_loops=1)
                #self.scheduler.enter(1, 0.0, play_animation, (x,y))

                #self.remove_creature(escaper)
                self.kill_creature(escaper)
                self.creature_data['hunger'][aggressor] = 0
                self.activate_creature_physics(aggressor)
                mood[aggressor] = 1
                # print('{} was killed'.format(escaper))
                print('{} wanted to fight {}, AND KILLED IT'.format(aggressor, escaper))

        # 4. Both want to reproduce
        elif checks[a]['virility'] and checks[b]['virility']:
            # create new based on a, b
            map(self.activate_creature_physics, [a,b])
            mood[[a,b]] = 1
            print('HUBBA HUBBA, {} and {} reproduced'.format(a,b))

        # 5. One wants to reproduce, other wants to run
        elif checks[a]['virility'] or checks[b]['virility']:
            virility = self.creature_data['virility_base'] + 1 - self.creature_data['hunger']
            aggressor = a if checks[a]['virility'] else b
            escaper = b if aggressor == a else a
            escaped = self.escape_attempt(escaper, aggressor)
            if escaped:
                map(self.activate_creature_physics, [a,b])
                mood[[a,b]] = 1
                print('{} wanted to reproduce with {}, but they got away'.format(aggressor, escaper))
            else:
                # create new based on a, b
                map(self.activate_creature_physics, [a,b])
                mood[[a,b]] = 1
                print('{} wanted to reproduce with {}, but they got away'.format(aggressor, escaper))

        # go back to normal
        # mood[[a,b]] = 1
        print('interactions ended for {} and {}'.format(a,b))
        self.creature_data['ended_interaction'][[a,b]] = self.ct
        self.creature_data['interacting_with'][[a,b]] = -1

    # Roll a die -> if it's below creatures stat, success
    def aggr_check(self, i):
        aggr = self.creature_data['aggressiveness_base'] + self.creature_data['hunger']
        #print('aggro checking {}, base aggr {:.3f}, current aggr {:.3f}'.format(i, self.creature_data['aggressiveness_base'][i], aggr[i]))
        # return np.random.random() < np.clip(aggr[i], 0.0, 0.9)
        return np.random.random()*2 < np.clip(aggr[i], 0.0, 1.8)
    def virility_check(self, i):
        virility = self.creature_data['virility_base'] + 1 - self.creature_data['hunger']
        #print('virility checking {}, base virility {:.3f}, current virility {:.3f}'.format(i, self.creature_data['virility_base'][i], virility[i]))
        # return np.random.random() < np.clip(virility[i], 0.0, 0.9)
        return np.random.random()*2 < np.clip(virility[i], 0.0, 1.8)
    # Just compare the powers
    def power_check(self, a, b):
        power = self.creature_data['power']
        return a if power[a] > power[b] else b

    # A fight will happen if the aggressor is faster
    def escape_attempt(self, escaper, aggressor):
        agility = self.creature_data['agility_base'] * (1 - (self.creature_data['age'] / self.creature_data['max_age']))
        return agility[escaper] > agility[aggressor]

    def get_collision_matrix(self):
        alive = self.creature_data['alive']
        scale = self.creature_data['size']

        dists = distm(self.creature_parts[::max_parts,:2], self.creature_parts[::max_parts,:2], 'euclidean')
        sizes = self.creature_data['size']
        radii = np.array([[sizes[i]+sizes[j] for j in np.arange(max_creatures)] for i in np.arange(max_creatures)])
        # collisions = dists < radii
        collisions = np.array([[dists[i,j] < radii[i,j] if (i!=j) and alive[[i,j]].all() else False for j in range(max_creatures)] for i in range(max_creatures)])
        # collisions = np.zeros((max_creatures, max_creatures))
        # for i in range(max_creatures):
        #     for j in range(max_creatures):
        #         collisions[i,j] = dists[i,j] < radii[i,j] if (i != j) and alive[[i,j]].all() else False
        # for i in range(max_creatures):
            # collisions[i,i] = False
        # return collisions, dists, radii
        return collisions, dists, radii

    @staticmethod
    def cleanup():
        print('Cleaning up')
        sa.delete('creature_gfx')


def main():
    timekeeper = TimeKeeper()
    culture = Culture()

    gfx_p = subprocess.Popen(['python', 'main.py'])

    running = True



    def signal_handler(signal_number, frame):
        print('Received signal {} in frame {}'.format(signal_number, frame))
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    print('Press Ctrl+C to quit')

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind('tcp://*:{}'.format(zmq_port))

    while running:
        try:
            message = socket.recv_json(zmq.NOBLOCK)
            print(message)
            creature_types = ['feet', 'simple', 'jelly', 'sperm']
            if message and 'type' in message:
                creature_type = creature_types[np.clip(int(message['type']),0,3)]
            else:
                creature_type = np.random.choice(creature_types)
            box_id = message['slave_id']
            #box_id = np.random.choice([1,2,3])
            if box_id == 1:
                position = (-7.5,-5.5)
            elif box_id == 2:
                position = (0,9)
            elif box_id == 3:
                position = (7.5,-5.5)
            culture.add_creature(creature_type, position, message)
            socket.send_string('OK', zmq.NOBLOCK)
        except zmq.error.Again:
            pass # no messages from sensor stations
        except:
            print("I don't know what happened: ", sys.exc_info()[0])
            raise

        timekeeper.update()
        culture.update(0.01)
        time.sleep(0.01)
        if gfx_p.poll() == 0:
            break

    culture.cleanup()


if __name__ == "__main__":
    main()
