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

zmq_port = '5556'
max_creatures = 50
max_parts = 5
offscreen_position = (30.0, 30.0, 0.0)

max_food = 100
max_animations = 100

resting_period = 30

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


class Culture(object):
    def __init__(self):
        # Create creature parts array to share with visualization
        self.creature_parts = create_new_sa_array('creature_parts', (max_creatures*max_parts, 12), np.float32)
        # FIXME: refactor to creature_gfx
        self.creature_parts[:, :3] = offscreen_position  # Off-screen coordinates
        self.creature_parts[:, 3:] = 1.0  # Avoid undefined behavior by setting everything to one

        # Creature data (no position!)
        # self.creature_data = np.zeros((max_creatures, 4))
        # self.creature_data[:, 1] = 100.0  # max_age
        # self.creature_data[:, 3] = 0.5  # creature size
        self.creature_data = np.recarray((max_creatures, ),
                                dtype=[('alive', int),
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
                                ('color', int)])
        self.creature_data.alive = 0
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
                            'constraint': None}
        self.creature_physics = [physics_skeleton.copy() for x in range(max_creatures)]

        self.pm_space = pm.Space()
        self.pm_space.damping = 0.4
        # self.pm_space.gravity = 0.0, -1.0

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

        self.demo_init()

        self.prev_update = time.perf_counter()
        self.ct = time.perf_counter()

        # self.dt = p0.0

    def demo_init(self):
        for i in range(5):
            self.add_jelly(4*i, tuple(np.random.rand(2)*20.0 - 10.0))
            self.add_simple(4*i + 1, tuple(np.random.rand(2) * 20.0 - 10.0))
            self.add_feet(4*i + 2, tuple(np.random.rand(2) * 20.0 - 10.0))
            self.add_sperm(4*i + 3, tuple(np.random.rand(2) * 20.0 - 10.0))

        for i in range(10):
            self.add_food(np.random.rand(2)*20.0 - 10.0)

        for i in range(3):
            self.add_animation('birth',
                               position=np.random.rand(2) * 20.0 - 10.0,
                               rotation=np.random.rand() * 2 * np.pi,
                               num_loops=5)
            self.add_animation('death',
                               position=np.random.rand(2) * 20.0 - 10.0,
                               rotation=np.random.rand() * 2 * np.pi,
                               num_loops=5)
            self.add_animation('contact',
                               position=np.random.rand(2) * 20.0 - 10.0,
                               rotation=np.random.rand() * 2 * np.pi,
                               num_loops=5)
            self.add_animation('fight',
                               position=np.random.rand(2) * 20.0 - 10.0,
                               rotation=np.random.rand() * 2 * np.pi,
                               num_loops=5)
            self.add_animation('reproduction',
                               position=np.random.rand(2) * 20.0 - 10.0,
                               rotation=np.random.rand() * 2 * np.pi,
                               num_loops=5)

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

    def add_jelly(self, index, position):
        print('Creating jelly at index {}'.format(index))
        # if self.creature_data[index, 0] == 1.0:
        if self.creature_data[index]['alive'] == 1:
            self.remove_creature(index)

        cp = self.creature_physics[index]

        cp['target'] = pm.Body(10.0, 10.0)
        cp['target'].position = position
        cp['target'].position += (0.0, 30.0)

        cp['body'] = [pm.Body(10.0, 5.0) for x in range(max_parts)]
        head, mid, tail = cp['body'][0:3]
        head.position = position
        mid.position = head.position + (0.0, -1.0)
        tail.position = head.position + (0.0, -2.0)
        cp['body'][3].position = (30.0, 30.0)  # UNUSED
        cp['body'][4].position = (30.0, 30.0)  # UNUSED

        head_offset = pm.Vec2d((0.0, 0.4))
        print(head_offset)
        cp['constraint'] = [pm.constraint.DampedSpring(head, cp['target'], head_offset, (0.0, 0.0), 0.0, 10.0, 15.0),
                            pm.constraint.SlideJoint(head, mid, (0.0, -0.1), (0.0, 0.1), 0.1, 0.5),
                            pm.constraint.RotaryLimitJoint(head, mid, -0.5, 0.5),
                            pm.constraint.SlideJoint(mid, tail, (0.0, -0.1), (0.0, 0.1), 0.1, 0.5),
                            pm.constraint.RotaryLimitJoint(mid, tail, -0.5, 0.5)]

        self.pm_space.add(cp['body'])
        self.pm_space.add(cp['constraint'])

        cp['active'] = True

        # self.creature_data[index, :] = [1.0, np.random.random(1)*10+10, 0.0, 0.5]  # Alive, max_age, age, size
        self.creature_data[index]['alive'] = 1
        self.creature_data[index]['max_age'] = np.random.random(1)*180+180
        self.creature_data[index]['age'] = 0
        self.creature_data[index]['size'] = 0.5
        self.creature_data[index]['mood'] = 1
        self.creature_data[index]['started_colliding'] = 0.0
        self.creature_data[index]['ended_interaction'] = 0.0
        self.creature_data[index]['agility_base'] = np.random.random()
        self.creature_data[index]['virility_base'] = np.random.random()
        self.creature_data[index]['mojo'] = np.random.random()
        self.creature_data[index]['aggressiveness_base'] = np.random.random()
        self.creature_data[index]['power'] = np.random.random()
        self.creature_data[index]['hunger'] = 0.5
        self.creature_data[index]['type'] = 1
        # self.creature_data[index]['color']

        position_vec = [position[0], position[1], 0.0, 0.5]  # Position, rotation, scale
        animation_vec = [0.0, 1.0, 1.0, 1.0]  # Animation time offset, beat frequency, swirl radius, swirl frequency
        for i in range(3):
            texture_vec = [self.get_texture('jelly'), 0.0, 1.0, 1.0]  # Texture index, color rotation, saturation, alpha
            self.creature_parts[max_parts*index+i, :] = position_vec + texture_vec + animation_vec

    def add_feet(self, index, position):
        print('Creating feet at index {}'.format(index))
        # if self.creature_data[index, 0] == 1.0:
        if self.creature_data[index]['alive'] == 1:
            self.remove_creature(index)

        cp = self.creature_physics[index]

        cp['target'] = pm.Body(10.0, 10.0)
        cp['target'].position = position
        cp['target'].position += (0.0, 10.0)

        cp['body'] = [pm.Body(10.0, 5.0) for x in range(max_parts)]
        top, bottom = cp['body'][0:2]
        top.position = position
        bottom.position = position
        for i in range(2, 5):
            cp['body'][i].position = offscreen_position[:2]

        head_offset = pm.Vec2d((0.0, 0.4))
        cp['constraint'] = [pm.constraint.DampedSpring(top, cp['target'], head_offset, (0.0, 0.0), 0.0, 10.0, 15.0),
                            pm.constraint.PivotJoint(top, bottom, (0.0, 0.0), (0.0, 0.0)),
                            pm.constraint.GearJoint(top, bottom, 0.0, 1.0)]

        self.pm_space.add(cp['body'])
        self.pm_space.add(cp['constraint'])

        cp['active'] = True

        self.creature_data[index]['alive'] = 1
        self.creature_data[index]['max_age'] = np.random.random(1) * 180 + 180
        self.creature_data[index]['age'] = 0
        self.creature_data[index]['size'] = 0.5
        self.creature_data[index]['mood'] = 1
        self.creature_data[index]['started_colliding'] = 0.0
        self.creature_data[index]['ended_interaction'] = 0.0
        self.creature_data[index]['agility_base'] = np.random.random()
        self.creature_data[index]['virility_base'] = np.random.random()
        self.creature_data[index]['mojo'] = np.random.random()
        self.creature_data[index]['aggressiveness_base'] = np.random.random()
        self.creature_data[index]['power'] = np.random.random()
        self.creature_data[index]['hunger'] = 0.5
        self.creature_data[index]['type'] = 3

        position_vec = [position[0], position[1], 0.0, 0.5]  # Position, rotation, scale
        animation_vec = [0.0, 1.0, 1.0, 1.0]  # Animation time offset, beat frequency, swirl radius, swirl frequency
        tex = self.get_texture('feet')
        texture_vec = [tex[1], 0.0, 1.0, 1.0]
        self.creature_parts[max_parts * index, :] = position_vec + texture_vec + animation_vec
        texture_vec = [tex[0], 0.25, 1.0, 0.01]
        self.creature_parts[max_parts * index + 1, :] = position_vec + texture_vec + animation_vec

    def add_simple(self, index, position):
        print('Creating simple at {} (index {})'.format(position, index))

        if self.creature_data[index]['alive'] == 1:
            self.remove_creature(index)

        cp = self.creature_physics[index]

        cp['target'] = pm.Body(10.0, 10.0)
        cp['target'].position = position
        cp['target'].position += (0.0, 30.0)

        cp['body'] = [pm.Body(10.0, 5.0) for x in range(max_parts)]
        head = cp['body'][0]
        head.position = position
        for i in range(1, 5):
            cp['body'][i].position = offscreen_position[:2]

        head_offset = pm.Vec2d((0.0, 0.4))
        cp['constraint'] = [pm.constraint.DampedSpring(head, cp['target'], head_offset, (0.0, 0.0), 0.0, 10.0, 15.0)]

        self.pm_space.add(cp['body'])
        self.pm_space.add(cp['constraint'])

        cp['active'] = True

        self.creature_data[index]['alive'] = 1
        self.creature_data[index]['max_age'] = np.random.random(1) * 180 + 180
        self.creature_data[index]['age'] = 0
        self.creature_data[index]['size'] = 0.5
        self.creature_data[index]['mood'] = 1
        self.creature_data[index]['started_colliding'] = 0.0
        self.creature_data[index]['ended_interaction'] = 0.0
        self.creature_data[index]['agility_base'] = np.random.random()
        self.creature_data[index]['virility_base'] = np.random.random()
        self.creature_data[index]['mojo'] = np.random.random()
        self.creature_data[index]['aggressiveness_base'] = np.random.random()
        self.creature_data[index]['power'] = np.random.random()
        self.creature_data[index]['hunger'] = 0.5
        self.creature_data[index]['type'] = 2

        position_vec = [position[0], position[1], 0.0, 0.5]  # Position, rotation, scale
        animation_vec = [0.0, 1.0, 1.0, 1.0]  # Animation time offset, beat frequency, swirl radius, swirl frequency
        texture_vec = [self.get_texture('simple'), 0.0, 1.0, 1.0]
        self.creature_parts[max_parts*index, :] = position_vec + texture_vec + animation_vec

    def add_sperm(self, index, position):
        print('Creating sperm at index {}'.format(index))
        # if self.creature_data[index, 0] == 1.0:
        if self.creature_data[index]['alive'] == 1:
            self.remove_creature(index)

        cp = self.creature_physics[index]

        cp['target'] = pm.Body(10.0, 10.0)
        cp['target'].position = position
        cp['target'].position += (0.0, 30.0)

        cp['body'] = [pm.Body(10.0, 5.0) for x in range(max_parts)]
        for i in range(5):
            cp['body'][i].position = position
            cp['body'][i].position += (0.0, -1.0*i)

        head_offset = pm.Vec2d((0.0, 0.4))
        print(head_offset)
        cp['constraint'] = [pm.constraint.DampedSpring(cp['body'][0], cp['target'], head_offset, (0.0, 0.0), 0.0, 10.0, 15.0)]
        for i in range(4):
            a = cp['body'][i]
            b = cp['body'][i+1]
            cp['constraint'].append(pm.constraint.SlideJoint(a, b, (0.0, -0.3), (0.0, 0.3), 0.1*(0.8**i), 0.2*(0.8**i)))
            cp['constraint'].append(pm.constraint.RotaryLimitJoint(a, b, -0.5, 0.5))

        self.pm_space.add(cp['body'])
        self.pm_space.add(cp['constraint'])

        cp['active'] = True

        # self.creature_data[index, :] = [1.0, np.random.random(1)*10+10, 0.0, 0.5]  # Alive, max_age, age, size
        self.creature_data[index]['alive'] = 1
        self.creature_data[index]['max_age'] = np.random.random(1) * 180 + 180
        self.creature_data[index]['age'] = 0
        self.creature_data[index]['size'] = 0.5
        self.creature_data[index]['mood'] = 1
        self.creature_data[index]['started_colliding'] = 0.0
        self.creature_data[index]['ended_interaction'] = 0.0
        self.creature_data[index]['agility_base'] = np.random.random()
        self.creature_data[index]['virility_base'] = np.random.random()
        self.creature_data[index]['mojo'] = np.random.random()
        self.creature_data[index]['aggressiveness_base'] = np.random.random()
        self.creature_data[index]['power'] = np.random.random()
        self.creature_data[index]['hunger'] = 0.5
        self.creature_data[index]['type'] = 4
        # self.creature_data[index]['color']

        position_vec = [position[0], position[1], 0.0, 0.5]  # Position, rotation, scale
        animation_vec = [0.0, 1.0, 1.0, 1.0]  # Animation time offset, beat frequency, swirl radius, swirl frequency
        tex_head, tex_tail = self.get_texture('sperm'), self.get_texture('sperm')
        texture_vec = [tex_head, 0.0, 1.0, 1.0]  # Texture index, color rotation, saturation, alpha
        self.creature_parts[max_parts * index, :] = position_vec + texture_vec + animation_vec
        for i in range(4):
            position_vec = [position[0], position[1], 0.0, 0.25*(0.8**i)]
            texture_vec = [tex_tail, 0.25, 1.0, 1.0]  # Texture index, color rotation, saturation, alpha
            self.creature_parts[max_parts * index + (i+1), :] = position_vec + texture_vec + animation_vec

    def add_animation(self, type, position, rotation=None, scale=1.0, relative_start_time=0.0, num_loops=1):
        # Add animation to next slot
        index = self.next_animation

        print('Adding {} animation at {} (index {})'.format(type, position, index))

        alpha_frame = omega_frame = 0.0

        # Get animation specific parameters
        if type == 'birth':
            start_frame, end_frame = 1.0, 16.0 # FIXME: halutaanko kovakoodata nämä
            loop_time = 1.0
        elif type == 'contact':
            start_frame, end_frame = 17.0, 34.0
            loop_time = 2.0
        elif type == 'death':
            scale *= 1.5
            start_frame, end_frame = 35.0, 56.0
            loop_time = 3.0
        elif type == 'fight':
            start_frame, end_frame = 57.0, 74.0
            loop_time = 2.0
        elif type == 'reproduction':
            start_frame, end_frame = 75.0, 92.0
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
            print('Adding {} animation at {} (index {})'.format(type, position, index))
            self.animation_gfx[index, :11] = position_vec + param1_vec + param2_vec

            param1_vec[2] += loop_time / 3.0
            old_index.append(index)
            index = index + 1 if index < max_animations - 1 else 0
            print('Adding {} animation at {} (index {})'.format(type, position, index))
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

        print('Addind food at {} (index {})'.format(position, index))

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
        cp = self.creature_physics[index]
        self.pm_space.remove(cp['constraint'])
        self.pm_space.remove(cp['body'])

        cp['active'] = False
        cp['target'] = None
        cp['body'] = None
        cp['constraint'] = None

        self.creature_data[index] = np.zeros(len(self.creature_data.dtype.names))
        self.creature_parts[index:index+max_parts, :3] = offscreen_position

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
        hunger = self.creature_data['hunger'] + dt
        agility = self.creature_data['agility_base'] * (1 - (self.creature_data['age'] / self.creature_data['max_age']))
        succulence = 1 - hunger
        aggr = self.creature_data['aggressiveness_base'] + self.creature_data['hunger']
        virility = self.creature_data['virility_base'] + 1 - self.creature_data['hunger']
        mood = self.creature_data['mood']
        creature_type = self.creature_data['type']
        last_interacted = self.creature_data['ended_interaction']

        # Update appearance changes from aging and remove dead creatures
        self.creature_parts[:, 6] = np.clip(1.0 - (cur_age / max_age), 0.0, 1.0).repeat(5)
        self.creature_parts[:, 7] = np.clip(1.0 - (cur_age - max_age)/5.0, 0.0, 1.0).repeat(5)
        dead_creatures = (alive == 1.0) & (cur_age > max_age + 5.0)
        for ind in np.where(dead_creatures)[0]:
            self.remove_creature(ind)

        #t1 = time.perf_counter()
        # Find colliding creatures and deal with them
        collisions = self.get_collision_matrix()
        #t2 = time.perf_counter()
        #print('collision_update', (t2 - t1) * 1000)
        ids_in_collision = np.nonzero([sum(i) for i in collisions])[0]
        if ids_in_collision.any():
            for id in ids_in_collision:
                # creatures can start an interaction if they are at mood 1
                other = collisions[id].nonzero()[0][0]
                if self.can_start_interaction([id, other]):
                    mood[[id,other]] = 0
                    print('{} were noticed to be colliding (mood=0) at ts {}'.format([id, other], time.perf_counter()))
            #modify colliders' alpha to see them
            self.creature_parts[ids_in_collision * max_parts, 7] = 0.2


        # Move creatures
        for i in range(max_creatures):
            cp = self.creature_physics[i]
            if not alive[i]:
                # Dead creatures don't need to move...
               continue

            if mood[i] == 0:
                # ...(WIP: for now, just mark the time and go to "stopped" state)...
                self.creature_data[i]['started_colliding'] = time.perf_counter()
                mood[i] = -1
                print('{} marked as started colliding (mood=-1) at ts {}'.format(i, self.creature_data[i]['started_colliding']))
                continue
            elif mood[i] == -1:
                # ...but occupied creatures need to stop
                cp['target'].position = cp['body'][0].position
                for body in cp['body']:
                    body.velocity = (0,0)
                    body.reset_forces()
                # in essence: stay still for 5s before revealing the result
                if (time.perf_counter() - self.creature_data[i]['started_colliding']) > 5:
                    # TODO: shoot in some direction
                    # cp['target'].position += (0,-300)
                    # map(lambda b: b.apply_impulse(0,-10000), cp['body'])
                    mood[i] = 1
                    self.creature_data[i]['ended_interaction'] = time.perf_counter()
                    print('{} ended interaction (back to mood = 1) at ts {}'.format(i, self.creature_data[i]['ended_interaction']))
                    continue

            elif mood[i] == 1:
                # Default mood, move creatures according to their type
                if creature_type[i] == 1:
                    if self.creature_physics[i]['body'][0].velocity.get_length() < 0.5:
                        self.creature_physics[i]['target'].position += random_circle_point()
                #    for j in range(3):
                #        self.creature_parts[3*i+j, :2] = tuple(self.pm_body[i][j].position)
                #        self.creature_parts[3*i+j, 2] = self.pm_body[i][j].angle


        # Advance the physics simulation and sync physics with graphics
        self.pm_space.step(dt)

        positions = [[(body.position.x, body.position.y, -body.angle) for body in creature['body']]
                     if creature['active']
                     else [offscreen_position for x in range(max_parts)]
                     for creature in self.creature_physics]
        positions = np.array(positions).reshape(max_creatures*max_parts, 3)
        self.creature_parts[:, :3] = positions

    def can_start_interaction(self, arr_like):
        mood = self.creature_data['mood']
        last_interacted = self.creature_data['ended_interaction']
        return (mood[arr_like] == 1).all() and ((self.ct - last_interacted[arr_like]) > resting_period).all()

    def get_collision_matrix(self):
        alive = self.creature_data['alive']
        scale = self.creature_data['size']

        dists = distm(self.creature_parts[::5,:2], self.creature_parts[::5,:2], 'euclidean')
        sizes = self.creature_data['size']
        radii = np.array([[sizes[i]+sizes[j] for j in np.arange(max_creatures)] for i in np.arange(max_creatures)])
        collisions = dists < radii
        for i in range(max_creatures):
            collisions[i,i] = False
        return collisions

    @staticmethod
    def cleanup():
        print('Cleaning up')
        sa.delete('creature_parts')


def main():
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
            message = socket.recv_string(zmq.NOBLOCK)
            index = np.random.randint(max_creatures)
            position = tuple(np.random.rand(2)*20.0 - 10.0)
            culture.add_jelly(index, position)
            socket.send_string('OK', zmq.NOBLOCK)
        except zmq.error.Again:
            pass # no messages from sensor stations
        except:
            print("I don't know what happened: ", sys.exc_info()[0])
            raise

        culture.update(0.01)
        time.sleep(0.01)
        if gfx_p.poll() == 0:
            break

    culture.cleanup()


if __name__ == "__main__":
    main()
