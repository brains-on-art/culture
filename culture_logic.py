import signal
import subprocess
import sys
import time

import numba
import numpy as np
import SharedArray as sa

sys.path.append('./pymunk')
import pymunk as pm

max_creatures = 50

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
    return x,y

class Culture(object):
    def __init__(self):
        try:
            self.creature_parts = sa.create('creature_parts', (max_creatures*3, 8), dtype=np.float32)
        except FileExistsError:
            sa.delete('creature_parts')
            self.creature_parts = sa.create('creature_parts', (max_creatures*3, 8), dtype=np.float32)
        # X POSITION, Y POSITION
        self.creature_parts[:, :2] = (30.0, 30.0)#np.random.random((max_creatures, 2)).astype(np.float32)*20.0 - 10.0
        # ROTATION
        self.creature_parts[:, 2] = np.random.random(max_creatures*3)*2*np.pi - np.pi
        # SCALE
        self.creature_parts[:, 3] = 0.5
        # TEXTURE INDEX
        self.creature_parts[:, 4] = np.random.randint(0, 10, max_creatures*3)
        # COLOR ROTATION
        self.creature_parts[:, 5] = np.random.randint(0, 4, max_creatures*3)/4.0
        # SATURATION
        self.creature_parts[:, 6] = 1.0
        # ALPHA
        self.creature_parts[:, 7] = 1.0

        self.creature_data = np.zeros((max_creatures, 5))
        self.creature_data[:, 1] = 1.0 # max_age
        self.creature_data[:, 3] = 0.5 # creature size
        self.creature_data[:, 4] = 1 # "mood" (state)
        # Mood can have these possible values:
        # 0 - Occupied in an interaction
        # 1 - Default: follow target aimlessly
        # 2 - ??

        self.pm_space = pm.Space()
        self.pm_space.damping = 0.7
        # self.pm_space.gravity = 0.0, -1.0
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

            head_mid_joint = pm.constraint.SlideJoint(head, mid, (0.0, -0.1), (0.0, 0.1), 0.1, 0.5)
            mid_tail_joint = pm.constraint.SlideJoint(mid, tail, (0.0, -0.1), (0.0, 0.1), 0.1, 0.5)
            self.pm_body_joint.append([head_mid_joint, mid_tail_joint])

            target = pm.Body(10.0, 10.0)
            target.position = tuple(self.creature_parts[i, :2] + (0.0, 5.0))
            self.pm_target.append(target)

            head_offset = pm.vec2d.Vec2d((0.0, 0.8)) * float(0.5)
            target_spring = pm.constraint.DampedSpring(head, target, head_offset, (0.0, 0.0), 0.0, 10.0, 15.0)
            self.pm_target_spring.append(target_spring)

            self.pm_space.add([head, mid, tail])
            self.pm_space.add([head_mid_joint, mid_tail_joint])
            self.pm_space.add([target_spring])

        self.prev_update = time.perf_counter()
        self.ct = time.perf_counter()

        #self.dt = p0.0

    def add_creature(self):
        # print('adding creature')
        ind = _find_first(self.creature_data[:, 0], 0.0)
        if ind != -1:
            new_pos = pm.vec2d.Vec2d(tuple(np.random.random(2)*20.0 - 10.0))
            # print('at position: ', new_pos)
            head_offset = pm.vec2d.Vec2d((0.0, 0.8)) * 0.5
            self.pm_target[ind].position = new_pos + head_offset
            self.pm_body[ind][0].position = new_pos #creature_data[ind, :2] = new_pos
            self.pm_body[ind][1].position = new_pos + (0.0, -0.5)
            self.pm_body[ind][2].position = new_pos + (0.0, -1.0)
            for i in range(3):
                self.pm_body[ind][i].reset_forces()
                self.pm_body[ind][i].velocity = 0.0, 0.0
                # self.creature_parts[ind*3+i, 3] = 0.5 # size/scale
                self.creature_parts[ind*3+i, 3] = 1.0 - i*0.3  # size/scale
                self.creature_parts[ind*3+i, 6] = 1.0

            self.creature_data[ind, :] = [1.0, np.random.random(1)*10+10, 0.0, 0.5, 1]  # Alive, max_age, age, size, mood


    def update(self, dt):
        self.ct = time.perf_counter()
        if self.ct - self.prev_update > 2.0:
            self.add_creature()
            #i = np.random.randint(0, max_creatures)
            #self.pm_target[i].position = tuple(np.random.random(2)*20.0 - 10.0)
            self.prev_update = self.ct

        alive = self.creature_data[:, 0]
        max_age = self.creature_data[:, 1]
        cur_age = self.creature_data[:, 2]
        cur_age[:] += dt
        # set saturation according to age
        self.creature_parts[:, 6] = np.clip(1.0 - (cur_age / max_age), 0.0, 1.0).repeat(3)

        self.creature_parts[:, 7] = np.clip(1.0 - (cur_age - max_age)/5.0, 0.0, 1.0).repeat(3)
        # dying_creatures = (alive == 1.0) & (cur_age > max_age)
        dead_creatures = (alive == 1.0) & (cur_age > max_age + 5.0)
        self.creature_data[dead_creatures, 0] = 0.0

        self.pm_space.step(dt)

        for i in range(max_creatures):
            head_offset = pm.vec2d.Vec2d((0.0, 0.8)) * 0.5
            if alive[i] == 1.0 and \
               (self.pm_body[i][0].position - (self.pm_target[i].position - head_offset)).get_length() < 2.0:
                self.pm_target[i].position += random_circle_point()
            for j in range(3):
                self.creature_parts[3*i+j, :2] = tuple(self.pm_body[i][j].position)
                self.creature_parts[3*i+j, 2] = self.pm_body[i][j].angle
            #colliding? change color
            if not self.creature_data[i, 4]:
                self.creature_parts[3*i, 5] = np.random.randint(0, 5)/4.0

        #self.creature_data[:, 2] += dt

        collisions = self.get_collision_matrix()
        in_collision = np.nonzero([sum(i) for i in collisions])[0]
        if in_collision.any():
            print('colliding: ', in_collision)
            for ind in range(max_creatures):
                self.creature_data[ind, 4] = 0 if ind in in_collision else 1

    def get_collision_matrix(self):
        collisions = np.zeros((max_creatures, max_creatures))
        alive = self.creature_data[:, 0]
        x = self.creature_parts[:, 0]
        y = self.creature_parts[:, 1]
        scale = self.creature_parts[:, 3]
        # we only need one half of the matrix separated by the diagonal, but
        # we still compute the whole thing D:
        for i in range(max_creatures):
            for j in range(max_creatures):

                # don't collide with self or if you're dead
                if i == j or (not alive[i] or not alive[j]):
                    collisions[i,j] = False
                    continue

                xdist = x[3*i] - x[3*j]
                ydist = y[3*i] - y[3*j]

                squaredist = np.sqrt((xdist * xdist) + (ydist * ydist))
                collisions[i,j] = squaredist <= scale[3*i] + scale[3*j]

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

    while running:
        culture.update(0.01)
        time.sleep(0.01)
        if gfx_p.poll() == 0:
            break

    culture.cleanup()


if __name__ == "__main__":
    main()
