import time
import pickle
import functools
import zlib
import blosc
from mpi4py import MPI
import torch
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

max_bytes = {}

LEVEL = 2
METHOD = 'snappy'


def to_np(d):
    if isinstance(d, torch.Tensor):
        if d.is_cuda:
            d.cpu()
        return d.numpy()
    if isinstance(d, dict):
        return {k: to_np(v) for k, v in d.items()}
    if isinstance(d, list):
        return list(map(to_np, d))
    if isinstance(d, map):
        return map(to_np, d)
    return d


def to_torch(d, cuda=False):
    if isinstance(d, np.ndarray):
        d = torch.Tensor(d)
        if cuda:
            d = d.cuda()
        return d
    if isinstance(d, dict):
        return {k: to_torch(v) for k, v in d.items()}
    if isinstance(d, list):
        return list(map(to_torch, d))
    if isinstance(d, map):
        return map(to_torch, d)
    return d

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
    obj = to_np(obj)
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
                       'alloc_time': t[3] - t[2],
                       'igather_time': t[4] - t[3], 'level': LEVEL,
                       'method': METHOD, 'alloc_bytes': max_bytes[name]}


def trim_msg(msg):
    """
    msg : bytearray
        Somewhere in msg, 15 elements are 255. Returns the msg before that
    """
    i = msg.find(b'\xff'*15)
    return msg[:i]


def irecv(recv, req, name="", cuda=False):
    global max_bytes
    if rank == 0:
        req.Wait()
        bytes_ = max_bytes[name]
        msgs = [recv[bytes_*n:bytes_*(n+1)] for n in range(size)]
        msgs = map(trim_msg, msgs)
        msgs = map(blosc.decompress, msgs)
        objs = map(pickle.loads, msgs)
        objs = map(functools.partial(to_torch, cuda=cuda), objs)
        return list(objs)


def irecv1(recv, req, cuda=False):
    req.Wait()
    recv = blosc.decompress(recv)
    obj = pickle.loads(recv)
    return to_torch(obj, cuda=cuda)


def ibroadcast(obj):
    obj = to_np(obj)
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

    *r, data = igather(obj, name="test")
    if rank == 0:
        objs = irecv(*r, name="test")
        for obj_hat in objs:
            assert obj == obj_hat
