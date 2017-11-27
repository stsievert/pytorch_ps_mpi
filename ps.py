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
                 encode=None, decode=None, rescale=True, svd_rank=0,
                 **kwargs):
        self.encode = encode
        self.decode = decode
        self.compress = kwargs.pop('compress', False)
        self.rescale = rescale
        self.svd_rank = svd_rank

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
                'encode_time': 0, 'decode_time': 0, 'param_compute_time': 0}
        for_loop_start = time.time()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            recv_msgs = []
            for i, param in enumerate(group['params']):
                start = time.time()
                msg = self.encode(param.grad.data, compress=self.compress,
                                  svd_rank=self.svd_rank,
                                  random_sample=self.rescale)
                data['encode_time'] += time.time() - start

                recv_msgs += [comms.igather(msg, name=i)]

            sent_msgs = []
            for i, (recv_msg, p) in enumerate(zip(recv_msgs, group['params'])):
                if self.rank == 0:
                    start = time.time()
                    codes = comms.irecv(*recv_msg, name=i)
                    data['grad_comm_time'] += time.time() - start
                    data['msg_size'] += _bytes_of(codes)
                    start = time.time()
                    grad = [self.decode(code, rescale=self.rescale)
                            for code in codes]
                    data['decode_time'] += time.time() - start
                    start = time.time()
                    d_p = sum(grad)

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
                sent_msgs += [comms.ibroadcast(p.data)]
            data['param_comm_time'] = 0
            for sent_msg, p in zip(sent_msgs, group['params']):
                start = time.time()
                p.data = comms.irecv1(*sent_msg)
                data['param_comm_time'] += time.time() - start

        data['total_time'] = time.time() - for_loop_start
        return loss, data
