import signal
import sys
import time

import numpy as np
import SharedArray as sa

sys.path.append('./pymunk')
import pymunk as pm

max_creatures = 50


class Culture(object):
    def __init__(self):
        self.creature_data = sa.create('creature_data', (max_creatures, 7), dtype=np.float32)
        # X POSITION, Y POSITION
        self.creature_data[:, :2] = np.random.random((max_creatures, 2)).astype(np.float32)*20.0 - 10.0
        # ROTATION
        self.creature_data[:, 2] = np.random.random(max_creatures).astype(np.float32)*2*np.pi - np.pi
        # TEXTURE INDEX
        self.creature_data[:, 3] = np.random.randint(0, 10, max_creatures).astype(np.float32)
        # COLOR ROTATION
        self.creature_data[:, 4] = np.random.randint(0, 4, max_creatures).astype(np.float32)/4.0
        # SATURATION
        self.creature_data[:, 5] = np.random.randint(0, 5, max_creatures).astype(np.float32)/4.0
        # ALPHA
        self.creature_data[:, 6] = np.random.randint(1, 5, max_creatures).astype(np.float32)/4.0

        self.pm_space = pm.Space()
        # self.pm_space.gravity = 0.0, -1.0
        self.pm_body = []
        self.pm_target = []
        self.pm_constraints = []
        for i in range(max_creatures):
            body = pm.Body(10.0, 10.0)
            body.position = tuple(self.creature_data[i, :2])
            self.pm_body.append(body)

            target = pm.Body(10.0, 10.0)
            target.position = tuple(self.creature_data[i, :2] + (0.0, 5.0))
            self.pm_target.append(target)

            target_spring = pm.constraint.DampedSpring(body, target, (0.0, 0.8), (0.0, 0.0), 0.0, 10.0, 5.0)
            self.pm_constraints.append(target_spring)

            self.pm_space.add([body, target_spring])

        self.prev_update = time.perf_counter()
        self.ct = time.perf_counter()

        #self.dt = p0.0

    def update(self, dt):
        self.ct = time.perf_counter()
        if self.ct - self.prev_update > 2.0:
            i = np.random.randint(0, max_creatures)
            self.pm_target[i].position = tuple(np.random.random(2)*20.0 - 10.0)
            self.prev_update = self.ct

        self.pm_space.step(dt)

        for i in range(max_creatures):
            self.creature_data[i, :2] = tuple(self.pm_body[i].position)

        self.creature_data[:, 2] += dt

    @staticmethod
    def cleanup():
        print('Cleaning up')
        sa.delete('creature_data')


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
