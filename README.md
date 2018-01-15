## Why?

### Features
It fits for our problem constraints:

* models fit on one device
* communication is reliable
* only one PS required
* have MPI available.

Additionally, this implementation of a PS

* uses MPI, a declarative standard common in clusters.
* can allow compression if concerned about bandwidth.

### Limitations
or, why use other PS systems? You may...

* want multiple PSs.
* have models that may not fit on one machine.
* not want to use MPI.

### Why not torch.distributed?
* torch.distributed only sends tensors. We may want to send generic Python
  objects.
* torch.distributed does not have any `Igatherv`, which is what we want for a
  PS.

## Notes
* `MPI_Gather` requires the recv count of any particular message.
    1. Send sizes before, allocate largest size (and then trim down per size)
    2. (when recieinv from any source)
* Tensorflow does... (see tensorflow's docs for ConditionalAccumulator)
    1. make sure gradients are current
    2. waits until $n$ gradients have been received from any number of workers

## plan
Implement a parameter server with mpi4py

0. Call encode
1. Convert to pickle object
2. (optional) compress
4. Allocate largest size expected + some padding (to see if overflowed)
5. Convert to NumPy buffer, use Igather
6. Convert back to pickle object when needed
7. Send dictionary to decode

## Notes
* An MPI enabled parameter server will not fit models that cannot fit on every
  machine.
* Though we can safely (?) assume that data for every machine is costant

## Resources
* Good summary of parameter servers at http://hunch.net/?p=151364

## Async PS psuedo-code
* This is algorithm AsySG-InCon (for inconsistent reads) in [1]
* [1]:Asynchronous parallel stochastic gradient for nonconvex optimization,
  https://arxiv.org/abs/1506.08272

``` python
# optimizaiton step function
irequest_params()
for p in params:
    if rank == 0:
        comms = []
        while True:
            comms += [recv(MPI.ANY_SOURCE)]
            if len(comms) == 32:
                break
        params = [receive(comm) for comm in comms]
        p = sum(params)
        step()
    else:
        send(param, 0)
    req = ibcast(p.data)
```

It's easy to impelment inconsistent reads with `ibcast`. It's harder to do
consistent reads; we need to do a buffered broadcast to allow workers to
continue to compute gradients.

## Iallgatherv
I tried playing with `Iallgatherv`. Some problems:

* Does not support sending objects of *unknown* sizes. Every rank needs to know
  the size of every object it's receiving.
* Sending the objects of these types involves sending a tuple as shown in [the
  implementation][allgather-impl] and [the tests][allgather-tests]

[allgather-tests]:https://github.com/mpi4py/mpi4py/blob/bd5278b232bde9f40247c3af1a8aed6166e7cbcf/test/test_cco_nb_vec.py#L192
[allgather-impl]:https://github.com/mpi4py/mpi4py/blob/3fd4dbd57b54f412e28b84aa6f77fb440c120f7d/test/arrayimpl.py#L99
