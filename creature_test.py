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
max_creatures = 25
max_parts = 5
offscreen_position = (30.0, 30.0)

max_food = 100
max_animations = 100

resting_period = 10

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
        self.creature_parts[:, :2] = offscreen_position  # Off-screen coordinates
        self.creature_parts[:, 2:] = 1.0  # Avoid undefined behavior by setting everything to one

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
        self.food_gfx[:, :2] = offscreen_position # Off-screen coordinates
        self.food_gfx[:, 2:] = 1.0  # Avoid undefined behavior by setting everything to one
        self.next_food = 0

        # Create animation graphics array to share with visualization
        self.animation_gfx = create_new_sa_array('animation_gfx', (max_animations, 12), np.float32)
        self.animation_gfx[:, :2] = offscreen_position  # Off-screen coordinates
        self.animation_gfx[:, 2:] = 1.0  # Avoid undefined behavior by setting everything to one
        self.next_animation = 0

        self.demo_init()

        self.prev_update = time.perf_counter()
        self.ct = time.perf_counter()

        # self.dt = p0.0

    def demo_init(self):
        self.add_jelly(0, (0.0, 0.0))

        for i in range(10):
            self.add_food(np.random.rand(2)*20.0 - 10.0)

        # for i in range(3):
        #     self.add_animation('birth',
        #                        position=np.random.rand(2) * 20.0 - 10.0,
        #                        rotation=np.random.rand() * 2 * np.pi,
        #                        num_loops=5)
        #     self.add_animation('death',
        #                        position=np.random.rand(2) * 20.0 - 10.0,
        #                        rotation=np.random.rand() * 2 * np.pi,
        #                        num_loops=20)

    def add_jelly(self, index, position):
        print('Creating jelly at index {}'.format(index))
        # if self.creature_data[index, 0] == 1.0:
        if self.creature_data[index]['alive']  == 1:
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

        head_offset = pm.vec2d.Vec2d((0.0, 0.8)) * float(0.5)
        cp['constraint'] = [pm.constraint.DampedSpring(head, cp['target'], head_offset, (0.0, 0.0), 0.0, 10.0, 15.0),
                            pm.constraint.SlideJoint(head, mid, (0.4, -0.3), (0.4, 0.3), 0.1, 0.2),
                            pm.constraint.SlideJoint(head, mid, (-0.4, -0.3), (-0.4, 0.3), 0.1, 0.2),
                            pm.constraint.SlideJoint(mid, tail, (0.0, -0.1), (0.0, 0.1), 0.1, 0.5)]

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

        for i in range(3):
            # Position, rotation, scale
            self.creature_parts[max_parts*index+i, :4] = [position[0], position[1], 0.0, 0.5]
            # Texture index, color rotation, saturation, alpha
            self.creature_parts[max_parts*index+i, 4:8] = [np.random.randint(0, 10), 0.0, 1.0, 1.0]
            # Animation time offset, beat frequency, swirl radius, swirl frequency
            self.creature_parts[max_parts*index+i, 8:12] = [0.0, 1.0, 1.0, 1.0]

    def add_animation(self, type, position, rotation=None, scale=1.0, relative_start_time=0.0, num_loops=1):
        # Add animation to next slot
        index = self.next_animation

        # print('Adding {} animation at {} (index {})'.format(type, position, index))
        # Get animation specific parameters
        if type == 'birth':
            start_frame, end_frame = 0.0, 15.0 # FIXME: halutaanko kovakoodata nämä
            loop_time = 1.0
        elif type == 'death':
            start_frame, end_frame = 16.0, 37.0
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
        param2_vec = [num_loops, start_frame, end_frame]

        # Add animation data to shared array
        self.animation_gfx[index, :11] = position_vec + param1_vec + param2_vec

        # Advance to next animation array position
        self.next_animation += 1
        if self.next_animation >= max_animations:
            self.next_animation = 0

        return index

    def add_food(self, position, rotation=None):
        # Add food to next slot
        index = self.next_food

        print('Addind food at {} (index {})'.format(position, index))

        # Randomize rotation by default
        if rotation is None:
            rotation = np.random.rand() * 2 * np.pi

        # Construct attribute vectors (matches GLSL)
        position_vec = [position[0], position[1], rotation, 0.25]

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
        self.creature_parts[index*max_parts:(index*max_parts)+max_parts-1, :2] = offscreen_position
        # self.creature_parts[index:index+max_parts, :2] = offscreen_position

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
        started_colliding = self.creature_data['started_colliding']
        last_interacted = self.creature_data['ended_interaction']

        # Update appearance changes from aging and remove dead creatures
        self.creature_parts[:, 6] = np.clip(1.0 - (cur_age / max_age), 0.0, 1.0).repeat(5)
        self.creature_parts[:, 7] = np.clip(1.0 - (cur_age - max_age)/5.0, 0.0, 1.0).repeat(5)
        dead_creatures = (alive == 1.0) & (cur_age > max_age + 5.0)
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
                # NB DONT ASSERT IN PRODUCTION
                # assert (np.argmin(distances[other]) == i) and (distances[i,other] < radii[i,other])
                self.creature_parts[i*max_parts, 7] = 0.2

                if self.can_start_interaction([i,other]):
                    print('{} found to be colliding with {}, with distance {} (radii sum {})'.format(i, other, distances[i,other], radii[i,other]))
                    mood[[i,other]] = 0
                    started_colliding[[i, other]] = self.ct

            # if self.can_start_interaction([i, other]):
            #     print('{} started an interaction'.format([i, other]) )
            #     mood[[i,other]] = 0
            #     # print('{} were noticed to be colliding (mood=0) at ts {}'.format([i, other], time.perf_counter()))
            #     # print('handling collision between {}'.format([i, other]))
            #     self.start_interaction([i, other])
            # elif self.creature_data[i]['mood'] == -1:
            #     self.process_interactions([i, other])
            # else:
            #         print('{} couldnt start with {}, and {} wasnt in mood -1'.format(i, other, i))
            #modify colliders' alpha to see them

        # Move creatures that are able to move (mood == 1)
        for i in range(max_creatures):
            cp = self.creature_physics[i]
            if not alive[i]:
                # Dead creatures don't need to move...
               continue

            if mood[i] == 0:
                # just started colliding
                print('{} started interaction'.format(i))
                # so go to next state and fire animation
                mood[i] = -1
                self.add_animation('birth',
                                   position=cp['body'][0].position,
                                   rotation=np.random.rand() * 2 * np.pi,
                                   num_loops=1)
            elif mood[i] == -1:
                if (self.ct - started_colliding[i]) < 5:
                    # not done with dancing
                    map(lambda b: b.reset_forces(), cp['body'])
                    for body in cp['body']:
                        body.velocity = (0, 0)
                else:
                    #end interaction
                    self.add_animation('death',
                                       position=cp['body'][0].position,
                                       rotation=np.random.rand() * 2 * np.pi,
                                       num_loops=1)
                    mood[i] = 1
                    last_interacted[i] = self.ct

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
        positions = [[(body.position.x, body.position.y) for body in creature['body']]
                     if creature['active']
                     else [offscreen_position for x in range(max_parts)]
                     for creature in self.creature_physics]
        self.creature_parts[:, :2] = np.array(positions).reshape(max_creatures*max_parts, 2)

    def can_start_interaction(self, arr_like):
        mood = self.creature_data['mood']
        last_interacted = self.creature_data['ended_interaction']
        return (mood[arr_like] == 1).all() and ((self.ct - last_interacted[arr_like]) > resting_period).all()

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
