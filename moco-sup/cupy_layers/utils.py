import torch
from string import Template
from collections import namedtuple
import cupy

Stream = namedtuple('Stream', ['ptr'])


def Dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'
    elif isinstance(t, torch.cuda.HalfTensor):
        return '__half'


@cupy.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    print(code)
    kernel_code = cupy.cuda.compile_with_cache(code, options=(
        "-I/home/caiqi/anaconda3/lib/python3.7/site-packages/cupy/_core/include",))
    return kernel_code.get_function(kernel_name)
