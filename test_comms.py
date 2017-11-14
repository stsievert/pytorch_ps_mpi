from mpi4py import MPI
import comms

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def test_gather():
    obj = {'str': 'str', 'rank': rank, 'list': [rank] * (rank + 1)}
    msg = comms.igather(obj, name=1)
    objs = comms.irecv(*msg, name=1)
    sent = [{'str': 'str', 'rank': rank, 'list': [rank] * (rank + 1)}
            for rank in range(size)]
    if rank == 0:
        assert objs == sent


def test_bcast():
    obj = {'x': 'x', 'list': [1]}
    if rank == 0:
        obj = {'a': 'a', 'list': [0]}
    tmp = comms.ibroadcast(obj)
    recv = comms.irecv1(*tmp)
    sent = {'a': 'a', 'list': [0]}
    assert recv == sent

if __name__ == "__main__":
    test_bcast()
    test_gather()
