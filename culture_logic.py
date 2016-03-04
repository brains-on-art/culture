import signal
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

class Culture(object):
    def __init__(self):
        self.creature_parts = sa.create('creature_parts', (max_creatures, 8), dtype=np.float32)
        # X POSITION, Y POSITION
        self.creature_parts[:, :2] = (30.0, 30.0)#np.random.random((max_creatures, 2)).astype(np.float32)*20.0 - 10.0
        # ROTATION
        self.creature_parts[:, 2] = np.random.random(max_creatures)*2*np.pi - np.pi
        # SCALE
        self.creature_parts[:, 3] = np.random.random(max_creatures)*0.9 + 1.0
        # TEXTURE INDEX
        self.creature_parts[:, 4] = np.random.randint(0, 10, max_creatures)
        # COLOR ROTATION
        self.creature_parts[:, 5] = np.random.randint(0, 4, max_creatures)/4.0
        # SATURATION
        self.creature_parts[:, 6] = 1.0
        # ALPHA
        self.creature_parts[:, 7] = 1.0

        self.creature_data = np.zeros((max_creatures, 3))
        self.creature_data[:, 1] = 1.0 # max_age

        self.pm_space = pm.Space()
        self.pm_space.damping = 0.9
        # self.pm_space.gravity = 0.0, -1.0
        self.pm_body = []
        self.pm_target = []
        self.pm_constraints = []
        for i in range(max_creatures):
            body = pm.Body(1.0, 10.0)
            body.position = tuple(self.creature_parts[i, :2])
            self.pm_body.append(body)

            target = pm.Body(10.0, 10.0)
            target.position = tuple(self.creature_parts[i, :2] + (0.0, 5.0))
            self.pm_target.append(target)

            target_spring = pm.constraint.DampedSpring(body, target, (0.0, 0.8), (0.0, 0.0), 0.0, 5.0, 15.0)
            self.pm_constraints.append(target_spring)

            self.pm_space.add([body, target_spring])

        self.prev_update = time.perf_counter()
        self.ct = time.perf_counter()

        #self.dt = p0.0

    def add_creature(self):
        print('adding creature')
        ind = _find_first(self.creature_data[:, 0], 0.0)
        if ind != -1:
            new_pos = np.random.random(2)*20.0 - 10.0
            print('at position: ', new_pos)
            self.pm_body[ind].position = tuple(new_pos) #creature_data[ind, :2] = new_pos
            self.pm_body[ind].reset_forces()
            self.pm_body[ind].velocity = 0.0, 0.0
            self.pm_target[ind].position = tuple(new_pos + (0.0, 0.8))
            self.creature_data[ind, :] = [1.0, np.random.random(1)*10+1, 0.0]  # Alive, max_age, age
            self.creature_parts[ind, 6] = 1.0

    def update(self, dt):
        self.ct = time.perf_counter()
        if self.ct - self.prev_update > 5.0:
            self.add_creature()
            #i = np.random.randint(0, max_creatures)
            #self.pm_target[i].position = tuple(np.random.random(2)*20.0 - 10.0)
            self.prev_update = self.ct

        alive = self.creature_data[:, 0]
        max_age = self.creature_data[:, 1]
        cur_age = self.creature_data[:, 2]
        cur_age[:] += dt
        self.creature_parts[:, 6] = np.clip(1.0 - (cur_age / max_age), 0.0, 1.0)
        # dying_creatures = (alive == 1.0) & (cur_age > max_age)
        self.creature_parts[:, 7] = np.clip(1.0 - (cur_age - max_age)/5.0, 0.0, 1.0)
        dead_creatures = (alive == 1.0) & (cur_age > max_age + 5.0)
        self.creature_data[dead_creatures, 0] = 0.0


        self.pm_space.step(dt)

        for i in range(max_creatures):
            self.creature_parts[i, :2] = tuple(self.pm_body[i].position)
            self.creature_parts[i, 2] = self.pm_body[i].angle

        #self.creature_data[:, 2] += dt

    @staticmethod
    def cleanup():
        print('Cleaning up')
        sa.delete('creature_parts')


def main():
    culture = Culture()

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

    culture.cleanup()


if __name__ == "__main__":
    main()
