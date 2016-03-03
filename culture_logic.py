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
        self.creature_data = sa.create('creature_data', (max_creatures, 5), dtype=np.float32)
        # X POSITION, Y POSITION
        self.creature_data[:, :2] = np.random.random((max_creatures, 2)).astype(np.float32)*20.0 - 10.0
        # ROTATION
        self.creature_data[:, 2] = np.random.random(max_creatures).astype(np.float32)*2*np.pi - np.pi
        # TEXTURE INDEX
        self.creature_data[:, 3] = np.random.randint(0, 10, max_creatures).astype(np.float32)
        # COLOR ROTATION
        self.creature_data[:, 4] = np.random.randint(0, 4, max_creatures).astype(np.float32)/4.0

        self.pm_space = pm.Space()
        #self.pm_space.gravity = 0.0, -1.0
        self.pm_body = []
        self.pm_target = []
        for i in range(max_creatures):
            body = pm.Body(10.0, 10.0)
            body.position = tuple(self.creature_data[i, :2])
            self.pm_body.append(body)

            target = pm.Body(10.0, 10.0)
            target.position = tuple(self.creature_data[i, :2] + (0.0, 2.0))
            self.pm_target.append(target)

            self.pm_space.add(body)

    def update(self, dt):
        self.pm_space.step(dt)

        for i in range(max_creatures):
            self.creature_data[i, :2] = tuple(self.pm_body[i].position)

        self.creature_data[:, 2] += dt

    def cleanup(self):
        print('Cleaning up')
        sa.delete('creature_data')


def main():
    culture = Culture()

    running = True

    def signal_handler(signal, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    print('Press Ctrl+C to quit')

    while running:
        culture.update(0.01)
        time.sleep(0.01)

    culture.cleanup()


    #pm_space = pm.Space()
    #pm_space.gravity = 0.0, -0.1
    #pm_creature = []
    #pm_shape = []
    #for i in range(max_creatures):
    #    creature = pm.Body(10.0, 10.0)
    #    creature.position = tuple(creature_data[i, :2])
    #    pm_creature.append(creature)
        #shape = pm.Circle(creature, 1.0)
        #shape.group = 1
        #pm_shape.append(shape)
        #pm_space.add(creature)
        #pm_space.add([creature, shape])

if __name__ == "__main__":
    main()