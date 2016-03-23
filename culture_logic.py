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
refractory_period = 10 #s

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

        self.creature_data = np.zeros((max_creatures, 11))
        # self.creature_data[:, 0] = 0 # alive
        self.creature_data[:, 1] = 1.0 # max_age
        # self.creature_data[:, 2] = 0 # cur_age
        self.creature_data[:, 3] = 0.5 # creature size

        self.creature_data[:, 4] = 1 # "mood" (state)
        # Mood can have these possible values:
        # -2 - Occupied in a sex
        # -1 - Occupied in a fight
        # 0 - Occupied in an interaction
        # 1 - Default: follow target aimlessly
        # 2 - ??

        # self.creature_data[:, 5] = 0 # ts last collided
        # self.creature_data[:, 6] = 0 # agility
        # self.creature_data[:, 7] = 0 # mojo
        # self.creature_data[:, 8] = 0 # aggressiveness
        # self.creature_data[:, 9] = 0 # power
        # self.creature_data[:, 10] = 0 # toughness

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

        self.refractory_period = refractory_period
        #self.dt = p0.0

    def add_creature(self, pos=None):
        # print('adding creature')
        ind = _find_first(self.creature_data[:, 0], 0.0)
        if ind != -1:
            if not pos:
                new_pos = pm.vec2d.Vec2d(tuple(np.random.random(2)*20.0 - 10.0))
            else:
                new_pos = pm.vec2d.Vec2d(pos)
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
                self.creature_parts[ind*3+i, 3] = 1 - i*0.2  # size/scale
                self.creature_parts[ind*3+i, 6] = 1.0

            self.creature_data[ind, :] = [1.0, # Alive
                                        np.random.random(1)*180+180, # max_age
                                        0.0, # age
                                        0.5, # size
                                        1, # mood
                                        # time.perf_counter(), # lastcollided
                                        0.0, # lastcollided
                                        np.random.random(), # agility
                                        np.random.random(), # mojo
                                        np.random.random(), # aggressiveness
                                        np.random.random(), # power
                                        np.random.random()]  # toughness


    def update(self, dt):
        self.ct = time.perf_counter()
        # add creatures every x seconds
        if self.ct - self.prev_update > 3.0:
            self.add_creature()
            #i = np.random.randint(0, max_creatures)
            #self.pm_target[i].position = tuple(np.random.random(2)*20.0 - 10.0)
            self.prev_update = self.ct

        alive = self.creature_data[:, 0]
        max_age = self.creature_data[:, 1]
        cur_age = self.creature_data[:, 2]
        cur_age[:] += dt
        # set saturation and alpha according to age
        self.creature_parts[:, 6] = np.clip(1.0 - (cur_age / max_age), 0.0, 1.0).repeat(3)
        self.creature_parts[:, 7] = np.clip(1.0 - (cur_age - max_age)/5.0, 0.0, 1.0).repeat(3)
        # dying_creatures = (alive == 1.0) & (cur_age > max_age)
        dead_creatures = (alive == 1.0) & (cur_age > max_age + 5.0)
        self.creature_data[dead_creatures, 0] = 0.0

        self.pm_space.step(dt)

        for i in range(max_creatures):
            # MOVE
            head_offset = pm.vec2d.Vec2d((0.0, 0.8)) * 0.5
            # head_offset = pm.vec2d.Vec2d((0.0, 0.8))
            # if alive[i] == 1.0 and \
            #    (self.pm_body[i][0].position - (self.pm_target[i].position - head_offset)).get_length() < 4.2:
            if alive[i] == 1.0 and self.pm_body[i][0].velocity.get_length() < 0.1:
                self.pm_target[i].position += random_circle_point()
            elif not alive[i]:
                self.pm_target[i].position = self.pm_body[i][0].position
            # if alive[i] == 1:
                # print((self.pm_body[i][0].position - (self.pm_target[i].position - head_offset)).get_length())
                # if ((self.pm_body[i][0].position - (self.pm_target[i].position - head_offset)).get_length() < 6):
                    # self.pm_target[i].position += random_circle_point()
            for j in range(3):
                self.creature_parts[3*i+j, :2] = tuple(self.pm_body[i][j].position)
                self.creature_parts[3*i+j, 2] = self.pm_body[i][j].angle

        #self.creature_data[:, 2] += dt

        collisions = self.get_collision_matrix()
        in_collision = np.nonzero([sum(i) for i in collisions])[0]
        if in_collision.any():
            # print(in_collision)
            #set moods: 0 for colliding and 1 for all others
            self.creature_data[in_collision, 4] = 0
            self.creature_data[np.setdiff1d(np.arange(max_creatures),in_collision), 4] = 1

            #modify colliders' alpha to see them
            self.creature_parts[3*in_collision, 7] = 0.2

        # interactions: check all creatures occupied in collision
        for ind in (self.creature_data[:, 4] == 0).nonzero()[0]:
            # who is everyone colliding with?
            other = collisions[ind].nonzero()[0]
            if not other.any():
                continue

            other = other[0] #FIXME only consider the first if many are colliding with this one

            # is either or both the belligerents occupied or resting?
            if self.is_occupied([ind, other]) or self.is_on_refractory_period([ind, other]):
                continue

            # does either belligerent want to fight?
            self.resolve_aggr_check(ind, other)

            # either belligerent(?) wants to sex?
            self.resolve_mating_check(ind, other)

    def is_occupied(self, arr_like):
        return (self.creature_data[arr_like, 4] < 0).any()

    def is_on_refractory_period(self, arr_like):
        return np.array([(self.ct - self.creature_data[r, 5] < self.refractory_period) for r in arr_like]).any()

    def resolve_mating_check(self, a, b):
        mojo = self.creature_data[:, 7]
        alive = self.creature_data[:, 0]
        if alive[[a,b]].all() and (mojo[[a,b]] > 0.4).any():
            print('these two had sex: {0} mojo={1:.4f}, {2} mojo={3:.4f}, and they produced a new one'.format(a, mojo[a], b, mojo[b]))
            self.add_creature(self.pm_body[a][0].position)
            self.creature_data[[a,b],5] = self.ct
        return

    def resolve_aggr_check(self, a, b):
        aggr = self.creature_data[:, 8]
        power = self.creature_data[:, 9]
        toughness = self.creature_data[:, 10]
        if max(aggr[[a,b]]) < 0.7: #neither wants to fight
            self.creature_data[[a,b],5] = self.ct
            print('neither {0} nor {1} wanted a fight'.format(a,b))

        else: #FIGHT
            print('at least one wants to fight: {0} aggr={1:.4f}, {2} aggr={3:.4f}'.format(a, aggr[a], b, aggr[b]))
            a_dmg = power[a] - toughness[b]
            b_dmg = power[b] - toughness[a]
            # someone has to die
            # we do it by setting max age to be current age
            if a_dmg < b_dmg:
                self.creature_data[a, 1] = self.creature_data[a, 2]
                self.creature_data[a, 0]  = 0 # blarg im dead
                self.creature_data[[a,b],5] = self.ct
                print('{0} killed {1}!'.format(b,a))

            else:
                self.creature_data[b, 1] = self.creature_data[b, 2]
                self.creature_data[b, 0] = 0 # blarg im dead
                self.creature_data[[a,b],5] = self.ct
                print('{0} killed {1}!'.format(a,b))

    def get_collision_matrix(self):
        collisions = np.zeros((max_creatures, max_creatures))
        alive = self.creature_data[:, 0]
        x = self.creature_parts[:, 0]
        y = self.creature_parts[:, 1]
        scale = self.creature_parts[:, 3]
        # we only need one half of the matrix separated by the diagonal, but
        # we still compute the whole thing D:
        # NVM RETURNING THE PART UNDER THE DIAGONAL
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
