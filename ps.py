import torch
import comms

class MPI_PS(torch.optim.SGD):
    def __init__(self, *args, encode=None, decode=None, **kwargs):
        self.encode = encode
        self.decode = decode
        self.rank = comms.rank
        super(torch.optim.SGD, MPI_PS).__init__(*args, **kwargs)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            recv_msgs = [comms.igather(self.encode(param.grad.data, name=param.name))
                         for param in group['params']]

            sent_msgs = []
            for recv_msg, p in zip(recv_msgs, group['params']):
                if self.rank == 0:
                    codes = comms.irecv(*recv_msg)
                    grad = [self.decode(code) for code in codes]
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
                sent_msgs += [comms.ibroadcast(p.data)]
            for p, sent_msg in zip(sent_msgs, group['params']):
                p.data = comms.irecv(*sent_msg)

        return loss
