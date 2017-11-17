import pickle
import zlib
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

max_bytes = {}


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
    pickled = pickle.dumps(obj)
    send = bytearray(pickled)
    max_bytes[name] = max(max_bytes.get(name, 0), len(send) * 1000)
    max_bytes[name] = max(max_bytes[name], 1024*1024)
    recv = bytearray(max_bytes[name] * size)
    req = comm.Igatherv([send, MPI.BYTE], [recv, MPI.BYTE])
    return recv, req


def irecv(recv, req, name=""):
    global max_bytes
    if rank == 0:
        req.Wait()
        bytes_ = max_bytes[name]
        msgs = [recv[bytes_*n:bytes_*(n+1)] for n in range(size)]
        objs = [pickle.loads(msg) for msg in msgs]
        return objs


def irecv1(recv, req):
    req.Wait()
    return pickle.loads(recv)


def ibroadcast(obj):
    pickled = pickle.dumps(obj)
    send = bytearray(pickled)
    req = comm.Ibcast([send, MPI.BYTE])
    return send, req


if __name__ == "__main__":
    pass
