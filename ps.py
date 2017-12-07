import torch
import time
import sys
__dir__ = "/".join(__file__.split('/')[:-1])
sys.path.append(__dir__)
import mpi_comms as comms
from mpi4py import MPI


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


class MPI_PS(torch.optim.SGD):
    def __init__(self, *args,
                 encode=None, decode=None, names=[],
                 encode_kwargs={}, use_mpi=True, cuda=False,
                 **kwargs):
        self.encode = encode
        self.decode = decode
        self.use_mpi = use_mpi
        #  self.compress = kwargs.pop('compress', False)
        #  self.rescale = rescale
        #  self.svd_rank = svd_rank
        self.encode_kwargs = encode_kwargs
        self.names = names
        self.cuda = cuda

        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.steps = 0
        super(MPI_PS, self).__init__(*args, **kwargs)

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
        data = {'grad_comm_time': 0, 'msg_size': 0, 'step': self.steps,
                'encode_time': 0, 'decode_time': 0, 'param_compute_time': 0,
                'grad_sum_time': 0, 'rank': self.rank, 'world_size': self.size,
                'igather_time': 0, 'ibroadcast_time': 0}
        for_loop_start = time.time()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            assert len(self.names) == len(group['params'])

            recv_msgs = {}
            igather_data = []
            for name, param in zip(self.names, group['params']):
                start = time.time()
                msg = self.encode(param.grad.data, cuda=self.cuda, **self.encode_kwargs)
                data['encode_time'] += time.time() - start
                start = time.time()
                if self.use_mpi:
                    *r, igather_datum = comms.igather(msg, name=name)
                    recv_msgs[name] = r
                    igather_data += [{'name': name, **igather_datum}]
                else:
                    recv_msgs[name] = msg
                    igather_data = []
                data['igather_time'] += time.time() - start
            data['igather_data'] = igather_data

            sent_msgs = []
            for ((name, recv_msg), p) in zip(recv_msgs.items(), group['params']):
                if self.rank == 0:
                    if self.use_mpi:
                        start = time.time()
                        codes = comms.irecv(*recv_msg, name=name, cuda=self.cuda)
                        data['grad_comm_time'] += time.time() - start
                        data['msg_size'] += _bytes_of(codes)

                        start = time.time()
                        grad = [self.decode(code, cuda=self.cuda) for code in codes]
                        data['decode_time'] += time.time() - start
                    else:
                        data['grad_comm_time'] += 1e-6
                        data['msg_size'] += _bytes_of(recv_msg)
                        start = time.time()
                        grad = [self.decode(recv_msg, cuda=self.cuda)]
                        data['decode_time'] += time.time() - start

                    start = time.time()
                    d_p = sum(grad)
                    data['grad_sum_time'] += time.time() - start

                    start = time.time()
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
                    data['param_compute_time'] += time.time() - start
                start = time.time()
                if self.use_mpi:
                    sent_msgs += [comms.ibroadcast(p.data)]
                data['ibroadcast_time'] += time.time() - start
            data['param_comm_time'] = 0
            if self.use_mpi:
                for sent_msg, p in zip(sent_msgs, group['params']):
                    start = time.time()
                    p.data = comms.irecv1(*sent_msg, cuda=self.cuda)
            data['param_comm_time'] += time.time() - start

        data['opt_time'] = time.time() - for_loop_start
        return loss, data
