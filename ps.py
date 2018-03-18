import torch
import time
from functools import partial
from toolz import reduce
import os
import sys
import math
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
import codings
import numpy as np
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


def decode_grads(codes, decode=None, cuda=False):
    grads = [decode(x, cuda=cuda) for x in codes]
    grads = [comms.to_torch(x, cuda=cuda) for x in grads]
    return grads

class MPI_PS(torch.optim.Optimizer):
    def __init__(self, named_params, *args,
                 names=[],
                 optim='sgd',
                 code=None,
                 use_mpi=True, cuda=False,
                 encode_kwargs={},
                 **kwargs):
        self.code = code
        self.optim = optim

        for i, (name, param) in enumerate(named_params):
            param.name = name
            param.register_hook(partial(self.async_code, name=name,
                                        encode=code.encode))
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
        code = encode(grad.data, **kwargs)
        code.update({'_norm(grad)**2': torch.norm(grad.data)**2})
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
            if len(set(self.names)) != len(group['params']):
                raise ValueError('len(set(names)) != len(params)')

            # Order in order of computation
            ordered_params = OrderedDict([(p.name, p)
                                          for p in group['params'][::-1]])

            # iallgather.send takes a long time because of stragglers (and it
            # has to send the counts to each machine).
            # To fix this, send all sizes async
            start = time.time()
            msgs = (future.result() for future in self.futures)
            #  msgs = concurrent.futures.wait(self.futures)
            #  print(msgs, type(msgs[0]))
            msgs = list(msgs)
            data['svd_rank'] = sum(msg.get('rank', -1) for msg in msgs) / len(msgs)
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
            _truth_var = 0
            _est_var = 0
            decode_futures = {}
            _decode_grads = partial(decode_grads, decode=self.code.decode,
                                    cuda=self.cuda)
            data['decode_send_time'] = 0
            for name, p, msg, response in paired_info:
                start = time.time()
                codes = self.iallgather.recv(*response, cuda=self.cuda)
                data['comm_wait'] += time.time() - start

                self.code.codes = codes
                start = time.time()
                decode_futures[name] = self.pool.submit(_decode_grads, codes)
                data['decode_send_time'] += time.time() - start

            for name, p, msg, future in paired_info:
                start = time.time()
                grads = decode_futures[name].result()
                data['decode_time'] += time.time() - start
                d_p = sum(grads)

                start = time.time()
                _truth_var += np.mean([code['_norm(grad)**2'] for code in codes])
                _est_var += np.mean([torch.norm(g)**2 for g in grads])

                cond = all([g.shape == grads[0].shape for g in grads])
                if not cond:
                    print("  !!", self.rank, p.name, [g.shape for g in grads])
                    raise ValueError('shapes not the same')
                d_p = sum(grads)

                if p.grad is None:
                    continue
                #  d_p = p.grad.data
                if self.optim == 'sgd':
                    kwargs = {k: group[k] for k in ['weight_decay', 'momentum',
                                                    'dampening', 'nesterov', 'lr']}
                elif self.optim == 'adam':
                    kwargs = {k: group[k] for k in ['betas', 'weight_decay',
                                                    'eps', 'lr']}
                else:
                    raise ValueError('self.optim not in [sgd, adam]')

                self.optim_step(p, d_p, **kwargs)
                data['optim_step_time'] += time.time() - start

        data['grad_variance_increase'] = _est_var / _truth_var
        data['steps'] = self.steps
        return loss, data

class SGD(MPI_PS, torch.optim.SGD):

    def optim_step(self, p, d_p, weight_decay=0, momentum=0, dampening=0,
                   nesterov=0, lr=0):
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

        p.data.add_(-lr, d_p)


class Adam(MPI_PS, torch.optim.Adam):
    def optim_step(self, p, grad, amsgrad=False, betas=[0.9, 0.999], weight_decay=0,
                  eps=1e-8, lr=1e-3):
        if grad.is_sparse:
            raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

        state = self.state[p]

        # State initialization
        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(p.data)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(p.data)
            if amsgrad:
                # Maintains max of all exp. moving avg. of sq. grad. values
                state['max_exp_avg_sq'] = torch.zeros_like(p.data)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        if amsgrad:
            max_exp_avg_sq = state['max_exp_avg_sq']
        beta1, beta2 = betas

        state['step'] += 1

        if weight_decay != 0:
            grad = grad.add(weight_decay, p.data)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = max_exp_avg_sq.sqrt().add_(eps)
        else:
            denom = exp_avg_sq.sqrt().add_(eps)

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        step_size = lr * math.sqrt(bias_correction2) / bias_correction1

        p.data.addcdiv_(-step_size, exp_avg, denom)
