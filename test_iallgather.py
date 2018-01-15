from mpi4py import MPI
import pickle
import numpy as np
import random


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def to_mpi_v(v, counts, dtype=MPI.BYTE):
    displacements = [sum(counts[:i]) for i in range(len(counts))]
    return (v, (counts, displacements), dtype)


def to_mpi(v, dtype=MPI.BYTE):
    return (v, dtype)


def collect_sizes(rank_size, size=size):
    rank_size = np.array(rank_size, dtype=np.int16)
    counts = np.zeros(size, dtype=np.int16)
    req = comm.Iallgather(rank_size, counts)
    req.Wait()
    return counts


def _make_obj(rank, size):
    obj = {'rank': rank, 'list': [rank] * (rank + 1)}
    return obj

def format_for_send(obj):
    send = pickle.dumps(obj)
    return send

if __name__ == "__main__":
    obj = _make_obj(rank, size)
    send = format_for_send(obj)

    counts = collect_sizes(len(send))

    recv = bytearray(sum(counts))
    req = comm.Iallgatherv(to_mpi(send), to_mpi_v(recv, counts))
    req.Wait()

    displacements = [sum(counts[:i]) for i in range(len(counts))]
    pickles = [recv[displacements[i]:displacements[i+1]]
               for i in range(len(displacements) - 1)]
    pickles += [recv[displacements[-1]:]]
    jar = [pickle.loads(p) for p in pickles]
    objs_true = [_make_obj(rank, size) for rank in range(size)]

    assert jar == objs_true
