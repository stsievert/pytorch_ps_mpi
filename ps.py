import torch
import time
from functools import partial
from toolz import reduce
import os
import sys
from collections import OrderedDict
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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import concurrent.futures
import cProfile, pstats, io
#  import multiprocessing as mp
#  mp.set_start_method('forkserver')

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
        self.call_pre_decode = encode_kwargs.pop('call_pre_decode', False)
        self.encode = partial(encode, **encode_kwargs)
        self.decode = decode

        for i, (name, param) in enumerate(named_params):
            param.name = name
            param.register_hook(partial(self.async_code, name=name,
                                        encode=self.encode))
        self.use_mpi = use_mpi
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
        self.names = []
        #  self.encode_timings = []
        self.pool = ThreadPoolExecutor(max_workers=200)
        #  self.pool = ProcessPoolExecutor()
        # TODO: look into this, chanage to processes

    def __exit(self):
        self.pool.shutdown()

    def format_for_send(self, grad, encode=None, format=comms.format_for_send,
                        **kwargs):
        code = encode(grad.data.cpu(), **kwargs)
        msg = format(code)
        return msg

    def async_code(self, grad, *args, name=None, **kwargs):
        future = self.pool.submit(self.format_for_send, grad, *args, **kwargs)
        self.futures += [future]
        self.names += [name]

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

        data = {'comm_wait': 0, 'optim_step_time': 0, 'decode_time': 0}
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            if len(set(self.names)) != len(group['params']):
                raise ValueError('len(set(names)) != len(params)')

            # Order in order of computation
            ordered_params = OrderedDict([(p.name, p)
                                          for p in group['params'][::-1]])

            # Wait for all keys to be in self.msgs
            #  grads = (p.grad.data.cpu() for name, p in ordered_params)
            #  start = time.time()
            #  codes = self.pool.map(partial(self.encode, **self.encode_kwargs),
                                    #  grads)
            #  codes = list(codes)
            #  data['code_wait'] = time.time() - start

            #  start = time.time()
            #  msgs = self.pool.map(comms.format_for_send, codes)
            #  msgs = list(msgs)
            #  data['format_for_send_time'] = time.time() - start

            # iallgather.send takes a long time because of stragglers (and it
            # has to send the counts to each machine).
            # To fix this, send all sizes async
            start = time.time()
            msgs = (future.result() for future in self.futures)
            #  msgs = concurrent.futures.wait(self.futures)
            #  print(msgs, type(msgs[0]))
            msgs = list(msgs)
            data['code_wait'] = time.time() - start

            start = time.time()
            sizes = self.iallgather.prepare(list(map(len, msgs)))
            data['iallgather_prepare_time'] = time.time() - start

            start = time.time()
            responses = []
            data['msg_bytes'] = 0
            for (req, count), msg in zip(sizes, msgs):
                req.Wait()
                data['msg_bytes'] += sum(count) / len(count)
                responses += [self.iallgather.send(msg, count)]
            data['isend_time'] = time.time() - start

            names = [p.name for p in group['params'][::-1]]
            if len(names) != len(set(names)):
                repeated = set([x for x in names if names.count(x) > 1])
                raise ValueError(f'names not unique. Repeated names = {repeated}')

            paired_info = [(name, ordered_params[name], msg, r)
                           for name, msg, r in zip(self.names, msgs, responses)]
            self.names = []
            self.futures = []
            for name, p, msg, response in paired_info:
                start = time.time()
                codes = self.iallgather.recv(*response, cuda=self.cuda)
                data['comm_wait'] += time.time() - start

                start = time.time()
                decode = partial(self.decode, cuda=self.cuda)
                if self.call_pre_decode:
                    result = self.pre_decode(codes)
                    decode = partial(decode, **result)
                grads = map(decode, codes)
                grads = list(map(comms.to_torch, grads))
                data['decode_time'] += time.time() - start

                start = time.time()

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
                data['optim_step_time'] += time.time() - start

        return loss, data
