import signal
import subprocess
import sys
import time

import numba
import numpy as np
import SharedArray as sa

sys.path.append('./pymunk')
import pymunk as pm

max_creatures = 9
part_per_creature = 5

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


class Culture(object):
    def __init__(self):
        # Create creature parts array to share with visualization
        try:
            self.creature_parts = sa.create('creature_parts', (max_creatures*part_per_creature, 12), dtype=np.float32)
        except FileExistsError:
            sa.delete('creature_parts')
            self.creature_parts = sa.create('creature_parts', (max_creatures*part_per_creature, 12), dtype=np.float32)

        # Creature data holds
        self.creature_data = np.zeros((max_creatures, 4))
        self.creature_data[:, 1] = 100.0  # max_age
        self.creature_data[:, 3] = 0.5  # creature size

        #
        physics_skeleton = {'active': False,
                            'target': None,
                            'body': None,
                            'constraints': None}
        self.creature_physics = [physics_skeleton.copy() for x in range(max_creatures)]

        self.pm_space = pm.Space()
        self.pm_space.damping = 0.4
        # self.pm_space.gravity = 0.0, -1.0

        self.init_creatures()

        self.pm_body = []
        self.pm_body_joint = []
        self.pm_target = []
        self.pm_target_spring = []
        for i in range(max_creatures):
            head = pm.Body(10.0, 5.0)
            head.position = tuple(self.creature_parts[i, :2])
            mid = pm.Body(1.0, 1.0)
            mid.position = head.position + (0.0, -1.0)
            tail = pm.Body(1.0, 1.0)
            tail.position = head.position + (0.0, -2.0)
            self.pm_body.append([head, mid, tail])

            head_mid_joint1 = pm.constraint.SlideJoint(head, mid, (0.4, -0.3), (0.4, 0.3), 0.1, 0.2)
            head_mid_joint2 = pm.constraint.SlideJoint(head, mid, (-0.4, -0.3), (-0.4, 0.3), 0.1, 0.2)
            mid_tail_joint = pm.constraint.SlideJoint(mid, tail, (0.0, -0.1), (0.0, 0.1), 0.1, 0.5)
            self.pm_body_joint.append([head_mid_joint1, head_mid_joint2, mid_tail_joint])

            target = pm.Body(10.0, 10.0)
            target.position = tuple(self.creature_parts[i, :2] + (0.0, 5.0))
            self.pm_target.append(target)

            head_offset = pm.vec2d.Vec2d((0.0, 0.8)) * float(0.5)
            target_spring = pm.constraint.DampedSpring(head, target, head_offset, (0.0, 0.0), 0.0, 10.0, 15.0)
            self.pm_target_spring.append(target_spring)

            self.pm_space.add([head, mid, tail])
            self.pm_space.add([head_mid_joint1, head_mid_joint2, mid_tail_joint])
            self.pm_space.add([target_spring])

        self.prev_update = time.perf_counter()
        self.ct = time.perf_counter()

        # self.dt = p0.0

    def init_creatures(self):
        # X POSITION, Y POSITION
        a = [-6, 0, 6]
        x, y = np.meshgrid(a, a)
        self.creature_parts[:, :2] = np.vstack([x.flatten(), y.flatten()]).T.repeat(5, axis=0)
        # ROTATION
        self.creature_parts[:, 2] = np.random.random(max_creatures*5)*2*np.pi - np.pi
        # SCALE
        self.creature_parts[:, 3] = 0.5
        # TEXTURE INDEX
        self.creature_parts[:, 4] = np.random.randint(0, 10, max_creatures*5)
        # COLOR ROTATION
        self.creature_parts[:, 5] = np.random.randint(0, 4, max_creatures*5)/4.0  # FIXME: 0, 30, -30, -60 deg ei punaisia!
        # SATURATION
        self.creature_parts[:, 6] = 1.0
        # ALPHA
        self.creature_parts[:, 7] = 1.0
        # TIME OFFSET (FOR ANIMATION
        self.creature_parts[:, 8] = np.random.random(max_creatures).repeat(5)*2*np.pi
        self.creature_parts[1::3, 8] += 0.4
        self.creature_parts[2::3, 8] += 0.8

        # BEAT ANIMATION FREQUENCY
        self.creature_parts[:, 9] = 2.0
        # SWIRL ANIMATON RADIUS
        self.creature_parts[:, 10] = 2.3
        # SWIRL ANIMATION FREQUENCY
        self.creature_parts[:, 11] = 1.0

        for i in range(9):
            pass

    def add_creature(self, type=None):
        if type is None:
            type = 0#np.random.randint(2)
        print('adding creature {}'.format(type))
        ind = _find_first(self.creature_data[:, 0], 0.0)
        if ind != -1:
            if type == 0: # Meduusa
                new_pos = pm.vec2d.Vec2d(tuple(np.random.random(2)*20.0 - 10.0))
                print('at position: ', new_pos)
                head_offset = pm.vec2d.Vec2d((0.0, 0.8)) * 0.5
                self.pm_target[ind].position = new_pos + head_offset
                self.pm_body[ind][0].position = new_pos #creature_data[ind, :2] = new_pos
                self.pm_body[ind][1].position = new_pos + (0.0, -0.5)
                self.pm_body[ind][2].position = new_pos + (0.0, -1.0)
                for i in range(3):
                    self.pm_body[ind][i].reset_forces()
                    self.pm_body[ind][i].velocity = 0.0, 0.0
                    self.creature_parts[ind*3+i, 3] = 0.5  # size/scale
                    self.creature_parts[ind*3+i, 6] = 1.0
                    self.creature_parts[ind*3+i, 4] = 2+i

                self.creature_data[ind, :] = [1.0, np.random.random(1)*10+10, 0.0, 0.5]  # Alive, max_age, age, size
            if type == 1: # Ötö
                pass


    def update(self, dt):
        self.ct = time.perf_counter()
        #if self.ct - self.prev_update > 5.0:
        #    self.add_creature()
            #i = np.random.randint(0, max_creatures)
            #self.pm_target[i].position = tuple(np.random.random(2)*20.0 - 10.0)
        #    self.prev_update = self.ct

        alive = self.creature_data[:, 0]
        max_age = self.creature_data[:, 1]
        cur_age = self.creature_data[:, 2]
        cur_age[:] += dt
        self.creature_parts[:, 6] = np.clip(1.0 - (cur_age / max_age), 0.0, 1.0).repeat(5)
        # dying_creatures = (alive == 1.0) & (cur_age > max_age)
        self.creature_parts[:, 7] = np.clip(1.0 - (cur_age - max_age)/5.0, 0.0, 1.0).repeat(5)
        dead_creatures = (alive == 1.0) & (cur_age > max_age + 5.0)
        self.creature_data[dead_creatures, 0] = 0.0

        self.pm_space.step(dt)

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

    while running:
        culture.update(0.01)
        time.sleep(0.01)
        if gfx_p.poll() == 0:
            break

    culture.cleanup()


if __name__ == "__main__":
    main()
