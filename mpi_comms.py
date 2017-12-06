import pickle
import zlib
from mpi4py import MPI
import numpy as np
import blosc
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

max_bytes = {}

LEVEL = 2
METHOD = 'snappy'


def igather(obj, name=""):
    """
    Gathers a python object to a root node.

    Returns
    =======
    objs : list[obj]
        List of objects from each worker
    req : MPI.REQUEST
        Supports req.Wait function
    """
    global max_bytes
    t = [time.time()]
    pickled = pickle.dumps(obj)
    t += [time.time()]
    send = bytearray(pickled)
    send = blosc.compress(send, clevel=LEVEL, cname=METHOD)
    t += [time.time()]

    send += bytearray(255 for _ in range(15))

    max_bytes[name] = max(max_bytes.get(name, 0), (len(send) + 1) * 1)
    max_bytes[name] = max(max_bytes[name], 1024*5)
    recv = bytearray(max_bytes[name] * size)
    t += [time.time()]
    req = comm.Igatherv([send, MPI.BYTE], [recv, MPI.BYTE])
    t += [time.time()]
    return recv, req, {'pickle_time': t[1] - t[0], 'compress_time': t[2] - t[1],
                       'igather_time': t[4] - t[3]}


def trim_msg(msg):
    """
    msg : bytearray
        Somewhere in msg, 15 elements are 255. Returns the msg before that
    """
    i = msg.find(b'\xff'*15)
    return msg[:i]


def irecv(recv, req, name=""):
    global max_bytes
    if rank == 0:
        req.Wait()
        bytes_ = max_bytes[name]
        msgs = [recv[bytes_*n:bytes_*(n+1)] for n in range(size)]
        msgs = map(trim_msg, msgs)
        msgs = map(blosc.decompress, msgs)
        objs = map(pickle.loads, msgs)
        return list(objs)


def irecv1(recv, req):
    req.Wait()
    recv = blosc.decompress(recv)
    return pickle.loads(recv)


def ibroadcast(obj):
    pickled = pickle.dumps(obj)
    send = bytearray(pickled)
    send = blosc.compress(send, clevel=LEVEL, cname=METHOD)
    req = comm.Ibcast([send, MPI.BYTE])
    return send, req


if __name__ == "__main__":
    obj = {rank: rank, 'str': str(rank)}

    r = ibroadcast(obj)
    obj_hat = irecv1(*r)
    assert obj_hat == obj

    r = igather(obj, name="test")
    if rank == 0:
        objs = irecv(*r, name="test")
        for obj_hat in objs:
            assert obj == obj_hat
