import torch
from torch import nn


def length(x):
    return torch.norm(x, dim=-1, keepdim=True)


def maximum(x, y, *z):
    if not isinstance(y, torch.Tensor):
        y = torch.full_like(x, y)
    m = torch.max(x, y)
    return maximum(m, *z) if z else m


def minimum(x, y, *z):
    if not isinstance(y, torch.Tensor):
        y = torch.full_like(x, y)
    m = torch.min(x, y)
    return minimum(m, *z) if z else m


def clamp(x, a, b):
    return minimum(maximum(x, a), b)


def mix(x, y, a):
    return x * (1 - a) + y * a


def stack(*x):
    return torch.stack(x, dim=-2)


def unstack(x):
    return torch.unbind(x, dim=-2)


def concat(*x):
    return torch.cat(x, dim=-1)


def unconcat(x):
    return torch.unbind(unsqueeze(x), dim=-2)


def squeeze(x):
    return torch.squeeze(x, dim=-1)


def unsqueeze(x):
    return torch.unsqueeze(x, dim=-1)


def abs(x):
    return torch.abs(x)


def cos(x):
    return torch.cos(x)


def sin(x):
    return torch.sin(x)


def relu(x):
    return nn.functional.relu(x)


def mod(x, y):
    return torch.fmod(x, y)


def sphere(p, r):
    return length(p) - r


def box(p, b):
    q = torch.abs(p) - b
    return length(relu(q)) - relu(-maximum(*unconcat(q)))


def torus(p, t):
    px, py, pz = unconcat(p)
    tx, ty = unconcat(t)
    q = concat(length(concat(px, pz)) - tx, py)
    return length(q) - ty


def round(d, r):
    return d - r


def twist(p, k):
    px, py, pz = unconcat(p)
    c = cos(k * py)
    s = sin(k * py)
    m = stack(concat(c, -s), concat(s, c))
    p = stack(px, pz)
    q = concat(squeeze(m @ p), py)
    return q


def bend(p, k):
    px, py, pz = unconcat(p)
    c = cos(k * px)
    s = sin(k * px)
    m = stack(concat(c, -s), concat(s, c))
    p = stack(px, py)
    q = concat(squeeze(m @ p), pz)
    return q


def repetition(p, c):
    return mod(p + 0.5 * c, c) - 0.5 * c


def union(d1, d2):
    return minimum(d1, d2)


def union(d1, d2, k):
    h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0., 1.)
    return mix(d2, d1, h) - k * h * (1. - h)


def intersection(d1, d2):
    return maximum(d1, d2)


def intersection(d1, d2, k):
    h = clamp(0.5 - 0.5 * (d2 - d1) / k, 0., 1.)
    return mix(d2, d1, h) + k * h * (1. - h)


def subtraction(d1, d2):
    return maximum(-d1, d2)


def subtraction(d1, d2, k):
    h = clamp(0.5 - 0.5 * (d2 + d1) / k, 0., 1.)
    return mix(d2, -d1, h) + k * h * (1. - h)
