from . import _C
def add(a, b):
    return _C.add(a, b)
def mul(a, b):
    return _C.mul(a, b)
def rms_norm(x, weight, eps=1e-6):
    return _C.rms_norm(x, weight, eps)
def softmax(x):
    return _C.softmax(x)
def rope(x, cos, sin):
    return _C.rope(x, cos, sin)
def linear(x,weight):
    return _C.gemm(x,weight)