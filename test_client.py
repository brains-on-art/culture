import zmq
import numpy as np
import time

def start_client():
    port = '5556'
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect('tcp://localhost:{}'.format(port))

    try:
        while True:
            time.sleep(10)
            x,y = np.random.random(2)*20.0 - 10.0
            # req = "{} {}".format(x,y)
            req = " ".join([str(a) for a in [x,y]])
            print('REQ={}'.format(req))
            socket.send_string(req)
            rep = socket.recv()
            print('REQ={} REP={}'.format(req, rep))
    except KeyboardInterrupt:
        print('\nQuitting')


if __name__ == '__main__':
    start_client()
