import torch
import time
from functools import partial
from toolz import reduce
import os
import sys
__dir__ = "/".join(__file__.split('/')[:-1])
sys.path.append(__dir__)
import mpi_comms as comms
from mpi4py import MPI
import pickle
from distributed import Client, LocalCluster
from pprint import pprint
sys.path.append('..')
sys.path.append('.')
import svd_comms
import qsgd
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

def _bytes_of(obj):
    # BUG: for 2D arrays doesn't return the number of bytes
    # that is, when sizes printed, only 1D sizes printed
    if isinstance(obj, torch.autograd.Variable):
        print('autograd variable')
        return _bytes_of(obj.grad) + obj.element_size()*obj.numel()
    cuda_tensor = getattr(obj, 'cuda', False)
    if isinstance(obj, torch.Tensor) or cuda_tensor:
        # t_size is a lower bound; only the number of elements
        t_size = obj.element_size() * obj.numel()
        #  py_size = sys.getsizeof(obj)
        return t_size

    if isinstance(obj, dict):
        return sum([_bytes_of(v) for k, v in obj.items()])
    if isinstance(obj, tuple) or isinstance(obj, list):
        return sum([_bytes_of(v) for v in obj])

    return sys.getsizeof(obj)  # only counting tensors as stores


def find_param(params, name):
    matches = [p for p in params if p.name == name]
    if len(matches) > 1:
        raise ValueError('More than one name found')
    return matches[0]


class MPI_PS(torch.optim.SGD):
    def __init__(self, named_params, *args,
                 encode=None, decode=None, names=[],
                 encode_kwargs={}, use_mpi=True, cuda=False,
                 **kwargs):
        self.encode = encode
        self.decode = decode
        for i, (name, param) in enumerate(named_params):
            param.name = name
            param.register_hook(partial(self.prepare_for_send, name=name,
                                        encode=self.encode,
                                        format=comms.format_for_send, i=i))
        self.use_mpi = use_mpi
        #  self.compress = kwargs.pop('compress', False)
        #  self.rescale = rescale
        #  self.svd_rank = svd_rank
        self.encode_kwargs = encode_kwargs
        self.names = names
        self.cuda = cuda

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.steps = 0
        self.iallgather = comms.Iallgather()
        super(MPI_PS, self).__init__(*args, **kwargs)

        self.recv_msgs = {}
        self.msgs = {}
        self.timings = []
        self.futures = []
        self.msgs = {}
        self.pool = ThreadPoolExecutor(max_workers=100)

    def __exit(self):
        self.pool.shutdown()

    def format_for_send(self, grad, name="", **kwargs):
        code = self.encode(grad.cpu(), **kwargs)
        msg = comms.format_for_send(code)
        return name, msg

    def prepare_for_send(self, grad, name=None, encode=None, i=0, format=None):
        def format_for_send(grad, name=None, i=0, encode=None, format=None,
                            **kwargs):
            code = encode(grad.cpu(), **kwargs)
            msg = format(code)
            self.msgs[name] = msg

        future = self.pool.submit(format_for_send, grad.data, name=name,
                                  cuda=self.cuda, i=i,
                                  encode=encode, format=format,
                                  **self.encode_kwargs)
        self.futures += [future]

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.steps += 1

        data = {'comm_wait': 0}
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            if len(set(self.names)) != len(group['params']):
                raise ValueError('len(set(names)) != len(params)')

            # Order in order of computation
            ordered_params = {p.name: p for p in group['params'][::-1]}

            # Wait for all keys to be in self.msgs
            start = time.time()
            concurrent.futures.wait(self.futures)
            data['encode_wait'] = time.time() - start

            # send gradients
            start = time.time()
            responses = {}
            for name in ordered_params:
                responses[name] = self.iallgather.send(self.msgs[name])
            self.msgs = {}
            data['isend_time'] = time.time() - start

            names = [p.name for p in group['params'][::-1]]
            if len(names) != len(set(names)):
                repeated = set([x for x in names if names.count(x) > 1])
                raise ValueError(f'names not unique. Repeated names = {repeated}')

            for name, p in ordered_params.items():
                start = time.time()
                codes = self.iallgather.recv(*responses[p.name], cuda=self.cuda)
                data['comm_wait'] += time.time() - start
                grads = list(map(partial(self.decode, cuda=self.cuda), codes))

                cond = all([g.shape == grads[0].shape for g in grads])
                if not cond:
                    print("  !!", self.rank, p.name, [g.shape for g in grads])
                    raise ValueError('shapes not the same')
                d_p = sum(grads)

                if p.grad is None:
                    continue
                #  d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss, data
