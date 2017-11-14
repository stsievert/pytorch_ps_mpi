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
