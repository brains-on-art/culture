import signal
import subprocess
import sys
import time
import zmq

import numba
import numpy as np
import SharedArray as sa

sys.path.append('./pymunk')
import pymunk as pm

zmq_port = '5556'
max_creatures = 9
max_parts = 5
offscreen_position = (30.0, 30.0)

max_food = 100
max_animations = 100


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
                                ('last_interacted', float),
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

        # Create animation graphics array to share with visualization
        self.animation_gfx = create_new_sa_array('animation_gfx', (max_animations, 8), np.float32)
        self.animation_gfx[:, :2] = offscreen_position  # Off-screen coordinates
        self.animation_gfx[:, 2:] = 1.0  # Avoid undefined behavior by setting everything to one

        self.used_animation = 0


        self.demo_init()



        self.prev_update = time.perf_counter()
        self.ct = time.perf_counter()

        # self.dt = p0.0

    def demo_init(self):
        self.add_jelly(0, (0.0, 0.0))

        self.food_gfx[:10, :2] = np.random.rand(10, 2)*20.0 - 10.0
        self.food_gfx[:10, 2] = np.random.rand(10)*2*np.pi
        self.food_gfx[:10, 3] = 0.25

        self.animation_gfx[:3, :2] = np.random.rand(3, 2) * 20.0 - 10.0
        self.animation_gfx[:3, 2] = np.random.rand(3) * 2 * np.pi
        self.animation_gfx[:3, 3] = 1.0
        self.animation_gfx[:3, 4] = 16.0
        self.animation_gfx[:3, 5] = np.random.rand(3)*5
        self.animation_gfx[:3, 6] = 0.1

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
        self.creature_data[index]['max_age' ] = 10
        self.creature_data[index]['age' ] = 0
        self.creature_data[index]['size' ] = 0.5
        self.creature_data[index]['mood'] = 1
        self.creature_data[index]['last_interacted' ] = time.perf_counter()
        self.creature_data[index]['agility_base' ] = np.random.random()
        self.creature_data[index]['virility_base' ] = np.random.random()
        self.creature_data[index]['mojo' ] = np.random.random()
        self.creature_data[index]['aggressiveness_base' ] = np.random.random()
        self.creature_data[index]['power' ] = np.random.random()
        self.creature_data[index]['hunger' ] = 0.5
        # self.creature_data[index]['type']
        # self.creature_data[index]['color']

        for i in range(3):
            # Position, rotation, scale
            self.creature_parts[max_parts*index+i, :4] = [position[0], position[1], 0.0, 0.5]
            # Texture index, color rotation, saturation, alpha
            self.creature_parts[max_parts*index+i, 4:8] = [np.random.randint(0, 10), 0.0, 1.0, 1.0]
            # Animation time offset, beat frequency, swirl radius, swirl frequency
            self.creature_parts[max_parts*index+i, 8:12] = [0.0, 1.0, 1.0, 1.0]

    def add_animation(self, type, position):
        if type == 'birth':
            pass

    def remove_creature(self, index):
        print('Removing creature at index {}'.format(index))
        cp = self.creature_physics[index]
        self.pm_space.remove(cp['constraint'])
        self.pm_space.remove(cp['body'])

        cp['active'] = False
        cp['target'] = None
        cp['body'] = None
        cp['constraint'] = None

        # self.creature_data[index, :] = [0.0, 1.0, 0.0, 1.0]  # Alive, max_age, age, size
        self.creature_data[index] = np.zeros(len(self.creature_data.dtype.names))
        self.creature_parts[index:index+max_parts, :2] = (30.0, 30.0)

    def update(self, dt):
        self.ct = time.perf_counter()
        # if self.ct - self.prev_update > 5.0:
        #     index = np.random.randint(max_creatures)
        #     position = tuple(np.random.rand(2)*20.0 - 10.0)
        #     self.add_jelly(index, position)
        #     self.prev_update = self.ct

            #i = np.random.randint(0, max_creatures)
            #self.pm_target[i].position = tuple(np.random.random(2)*20.0 - 10.0)
        #

        # Update creature age and remove dead creatures
        alive = self.creature_data['alive']
        max_age = self.creature_data['max_age']
        cur_age = self.creature_data['age']
        cur_age[:] += dt
        self.creature_parts[:, 6] = np.clip(1.0 - (cur_age / max_age), 0.0, 1.0).repeat(5)
        # dying_creatures = (alive == 1.0) & (cur_age > max_age)
        self.creature_parts[:, 7] = np.clip(1.0 - (cur_age - max_age)/5.0, 0.0, 1.0).repeat(5)
        dead_creatures = (alive == 1.0) & (cur_age > max_age + 5.0)
        for ind in np.where(dead_creatures)[0]:
            self.remove_creature(ind)
        #self.creature_data[dead_creatures, 0] = 0.0

        self.pm_space.step(dt)

        # Sync physics with graphics
        positions = [[(body.position.x, body.position.y) for body in creature['body']]
                     if creature['active']
                     else [offscreen_position for x in range(max_parts)]
                     for creature in self.creature_physics]
        self.creature_parts[:, :2] = np.array(positions).reshape(max_creatures*max_parts, 2)

        #for i in range(max_creatures):
        #    head_offset = pm.vec2d.Vec2d((0.0, 0.8)) * 0.5
        #    if alive[i] == 1.0 and \
        #       (self.pm_body[i][0].position - (self.pm_target[i].position - head_offset)).get_length() < 2.0:
        #        self.pm_target[i].position += random_circle_point()
        #    for j in range(3):
        #        self.creature_parts[3*i+j, :2] = tuple(self.pm_body[i][j].position)
        #        self.creature_parts[3*i+j, 2] = self.pm_body[i][j].angle

        #self.creature_data[:, 2] += dt

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
