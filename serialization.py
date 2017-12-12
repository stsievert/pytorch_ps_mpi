import pickle
import cloudpickle
import blosc
import numpy as np
import torch


def _tensor_info(v, k):
    ret = {f: getattr(v, f)() for f in ['numel', 'data_ptr', 'element_size']}
    ret.update(k=k)
    return ret


def _predump(obj):
    tensors = {k: v for k, v in obj.items()
               if isinstance(v, (torch.Tensor, torch.Storage))}
    info = [_tensor_info(v, k) for k, v in tensors.items()]
    d = {k: v for k, v in obj.items() if k not in tensors}
    return cloudpickle.dumps(d), info


def _tensor_dump(info):
    return blosc.compress_ptr(info['data_ptr'], info['numel'], info['element_size'])

def compress(obj):
    d, tensor_info = _predump(obj)
    msgs = list(map(_tensor_dump, tensor_info))


    return msg


def decompress(msg):
    b = blosc.decompress(msg)
    y = torch.ByteStorage().from_buffer(b, 'native')
    return bytes(y)
    #  x_hat = torch.ByteTensor(y)
    #  return x_hat

if __name__ == "__main__":
    n = int(1e3)
    x = torch.linspace(0, 6.28, n)
    y = torch.sin(x) + torch.randn(n) / 4
    obj = {'x': x, 'y': y, 'n': n}
    obj_np = {'x': x.numpy(), 'y': y.numpy(), 'n': n}

    o = dumps(obj)
    msg = compress(o)
    oo = decompress(msg)
    obj_hat = loads(oo)
