import pickle
import zlib
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
dtype = np.dtype('int16')

name = 1
obj = {'str': 'str', 'int': 1}
max_bytes = {}
pickled = pickle.dumps(obj)
send = bytearray(pickled)
max_bytes[name] = max(max_bytes.get(name, 0), len(send)) * 100
print(max_bytes, len(send))
recv = bytearray(max_bytes[name] * size)
#  req = comm.Ialltoallv([send, MPI.BYTE], [recv, MPI.BYTE])
req = comm.Ialltoallv([send, MPI.BYTE], [recv, MPI.BYTE])

#  req.Wait()
#  bytes_ = max_bytes[name]
#  msgs = [recv[bytes_*n:bytes_*(n+1)] for n in range(size)]
#  objs = [pickle.loads(msg) for msg in msgs]








def test_async_send_bytearray():
    obj = {'size': size, 'rank': [rank] * (rank+1)}
    pickled = pickle.dumps(obj)
    send = bytearray(pickled)
    recv = None
    if rank == 0:
        max_bytes = 1024 * 1024
        recv = bytearray(max_bytes * size)

    dtype = MPI.BYTE
    req = comm.Igatherv([send, dtype], [recv, dtype])
    req.Wait()

    if rank == 0:
        msgs = [recv[max_bytes*n:max_bytes*(n+1)] for n in range(size)]
        objs = [pickle.loads(msg) for msg in msgs]
        sent = [{'size': size, 'rank': [rank]*(rank+1)} for rank in range(size)]
        assert objs == sent


def test_async_diff_sizes():
    data = {'a': 'a', 'async': [rank] * (rank + 1)}
    pickled = pickle.dumps(data)
    while len(pickled) % dtype.itemsize != 0:
        pickled += b' '
    send = np.fromstring(pickled, dtype=dtype)

    recv = None
    if rank == 0:
        max_bytes = 1024 * 1024
        recv = np.empty([size, max_bytes // 2], dtype=dtype)

    req = comm.Igatherv(send, recv)
    req.Wait()

    if rank == 0:
        msgs = np.array_split(recv, size)
        objs = [pickle.loads(msg) for msg in msgs]

        sent = [{'a': 'a', 'async': [rank] * (rank + 1)}
                for rank in range(size)]
        assert objs == sent


def test_sync_same_size():
    data = {'a': 'a', 'sync': [rank] * (rank + 1)}
    pickled = pickle.dumps(data)

    send = np.fromstring(pickled, dtype=np.uint8)
    recv = None
    if rank == 0:
        recv = np.empty([size, size*len(pickled)], dtype=np.uint8)

    comm.Gatherv(send, recv)

    if rank == 0:
        msgs = np.array_split(recv, size)
        elems = [msg.tobytes() for msg in msgs]
        objs = [pickle.loads(x) for x in elems]

        sent = [{'a': 'a', 'sync': [rank] * (rank + 1)}
                for rank in range(size)]
        assert objs == sent


