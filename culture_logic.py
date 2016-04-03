import signal
import zmq
import subprocess
import sys
import time

import numba
import numpy as np
import SharedArray as sa

sys.path.append('./pymunk')
import pymunk as pm

zmq_port = '5556'
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

        self.creature_data = np.zeros((max_creatures, 12))
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
        # self.creature_data[:, 11] = 0 # hunger

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

            shape = pm.Circle(head, self.creature_parts[0,3]) # use the scale of the first creature part
            # shape.sensor = True
            shape.collision_type = 1
            self.pm_space.add(shape)

        # def printcolliders(space, arbiter):
        #     print([s.body for s in arbiter.shapes])
        #     return True
        #
        # self.pm_space.add_collision_handler(1, 2, pre_solve=printcolliders)

        # add some walls
        walls = [pm.Segment(self.pm_space.static_body, (-13, 13), (13, 13), 0.1)
                ,pm.Segment(self.pm_space.static_body, (13, 13), (13, -13), 0.1)
                ,pm.Segment(self.pm_space.static_body, (13, -13), (-13, -13), 0.1)
                ,pm.Segment(self.pm_space.static_body, (-13, -13), (-13, 13), 0.1)
                ]
        for wall in walls:
            wall.collision_type = 2
        self.pm_space.add(walls)

        self.prev_update = time.perf_counter()
        self.ct = time.perf_counter()

        self.refractory_period = refractory_period
        #self.dt = p0.0

    def add_creature(self, x=None, y=None, pos=None):
        # print('adding creature at ', x, y, pos)
        ind = _find_first(self.creature_data[:, 0], 0.0)
        if ind != -1:
            if pos:
                new_pos = pm.vec2d.Vec2d(pos)
            elif (x and y):
                new_pos = pm.vec2d.Vec2d((float(x), float(y)))
            else:
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
                                        np.random.random(), # base_aggressiveness
                                        np.random.random(), # power
                                        np.random.random(), # toughness
                                        np.random.random()] # hunger


    def update(self, dt):
        self.ct = time.perf_counter()
        # add creatures every x seconds
        # if self.ct - self.prev_update > 3.0:
        #     self.add_creature()
        #     #i = np.random.randint(0, max_creatures)
        #     #self.pm_target[i].position = tuple(np.random.random(2)*20.0 - 10.0)
        #     self.prev_update = self.ct

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

        #compute dynamic params at current ts
        agility = self.creature_data[:, 6] * (1 - (self.creature_data[:,2] / self.creature_data[:,1]))
        aggr = self.creature_data[:, 8] + self.creature_data[:, 11]
        power = self.creature_data[:, 9]

        # interactions: check all creatures occupied in collision
        for ind in (self.creature_data[:, 4] == 0).nonzero()[0]:
            # who am i colliding with?
            other = collisions[ind].nonzero()[0]
            if not other.any():
                continue

            # FIXME do something fancy here?
            other = other[0] # only consider the first if many are colliding with this one

            # is either or both the belligerents occupied or recovering?
            if self.is_occupied([ind, other]) or self.is_on_refractory_period([ind, other]):
                continue

            # does either belligerent want to fight?
            # self.resolve_aggr_check(ind, other)
            aggr_checks = self.aggr_check([ind, other], aggr)
            if aggr_checks.all(): #both want to fight, or [True, True]
                winner = ind if power[ind] > power[other] else other
                loser = other if winner == ind else ind
                print('{} killed {}'.format(winner, loser))

            elif aggr_checks.any(): #only 1 wants to fight, so give chase
                chaser = [ind, other][aggr_checks.argmax()]
                chased = [ind, other][aggr_checks.argmin()]

                did_escape = self.check_escape(chaser, chased, agility)
                print('escape check for {} against {} {}'.format(chased, chaser, {True:'succeeded', False:'failed'}[did_escape]))

            else: # neither wanted to fight so how about mating?
                print('neither {} nor {} wanted to fight'.format(ind, other))

            # either belligerent(?) wants to sex?
            # self.resolve_mating_check(ind, other)

            # finally set both as "recently checked" so they wont interact in a while
            self.creature_data[[ind, other],5] = self.ct

    def is_occupied(self, arr_like):
        return (self.creature_data[arr_like, 4] < 0).any()

    def is_on_refractory_period(self, arr_like):
        return np.array([(self.ct - self.creature_data[r, 5] < self.refractory_period) for r in arr_like]).any()

    def check_escape(self, chaser, chased, agility):
        # compares dynamic agilities, returns True if chased is more agile, i.e gets away
        return agility[chaser] < agility[chased]

    def aggr_check(self, creature_ind_array, aggr):
        # roll a die (random [0,1]): if the value is less than aggr-value ([0,1]), the check returns True
        aggro_array = [aggr[i] > np.random.random() for i in creature_ind_array]
        print('aggressiveness checks for {} / {}: {}'.format(creature_ind_array, aggr[creature_ind_array], aggro_array))
        return np.array(aggro_array)

    # def resolve_mating_check(self, a, b):
    #     alive = self.creature_data[:, 0]
    #     agility = self.creature_data[:, 6]
    #     mojo = self.creature_data[:, 7]
    #     if alive[[a,b]].all() and (mojo[[a,b]] > 0.6).any():
    #         print('these two had sex: {0} mojo={1:.4f}, {2} mojo={3:.4f}, and they produced a new one'.format(a, mojo[a], b, mojo[b]))
    #         self.add_creature(pos=self.pm_body[a][0].position)
    #         self.creature_data[[a,b],5] = self.ct
    #     return

    # def resolve_aggr_check(self, a, b):
    #     alive = self.creature_data[:, 0]
    #     agility = self.creature_data[:, 6]
    #     aggr = self.creature_data[:, 8]
    #     power = self.creature_data[:, 9]
    #     toughness = self.creature_data[:, 10]
    #     if not alive[[a,b]].all():
    #         return
    #     if max(aggr[[a,b]]) < 0.6: #neither wants to fight
    #         self.creature_data[[a,b],5] = self.ct
    #         print('neither {0} nor {1} wanted a fight'.format(a,b))
    #
    #     else: #FIGHT
    #         print('at least one wants to fight: {0} aggr={1:.4f}, {2} aggr={3:.4f}'.format(a, aggr[a], b, aggr[b]))
    #         a_dmg = power[a] - toughness[b]
    #         b_dmg = power[b] - toughness[a]
    #         # only one can walk away from this
    #         winner = a if a_dmg > b_dmg else b
    #         loser = b if winner == a else a
    #         # we do it by setting max age to be current age
    #         self.creature_data[loser, 1] = self.creature_data[loser, 2]
    #         self.creature_data[loser, 0]  = 0 # blarg im dead
    #
    #         # try to stop the dead thing from moving: set target pos = creature pos
    #         self.pm_target[loser].position = self.pm_body[loser][0].position
    #
    #         self.creature_data[[a,b],5] = self.ct
    #         print('{0} killed {1}!'.format(winner,loser))

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

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind('tcp://*:5556')

    while running:
        try:
            message = socket.recv_string(zmq.NOBLOCK)
            x,y = message.split()
            culture.add_creature(x,y)
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
