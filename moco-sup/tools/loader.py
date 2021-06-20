# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import random


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

class TwoCropsMultiTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self,
                 base_transform_q1,
                 base_transform_q2,
                 base_transform_k,
                 prob_q):
        self.base_transform_q1 = base_transform_q1
        self.base_transform_q2 = base_transform_q2
        self.base_transform_k = base_transform_k
        self.prob_q = prob_q

    def __call__(self, x):
        if random.uniform(0., 1.) <= self.prob_q:
            q = self.base_transform_q1(x)
        else:
            q = self.base_transform_q2(x)
        k = self.base_transform_k(x)
        return [q, k]

class TwoScaleCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self,
                 base_transform_l,
                 base_transform_s):
        self.base_transform_l = base_transform_l
        self.base_transform_s = base_transform_s

    def __call__(self, x):
        q_l = self.base_transform_l(x)
        q_s = self.base_transform_s(x)
        k = self.base_transform_l(x)
        return [q_l, q_s, k]

class MultiCropsTransform:
    """Take multiple random crops of one image as different views."""

    def __init__(self, base_transform, times):
        self.base_transform = base_transform
        self.times = times
        assert self.times > 1

    def __call__(self, x):
        return [self.base_transform(x) for _ in range(self.times)]


class MultiCropsQKTransform:
    """Take multiple random crops of one image as different views."""

    def __init__(self, base_transform_q, base_transform_k, times):
        self.base_transform_q = base_transform_q
        self.base_transform_k = base_transform_k
        self.times = times
        assert self.times > 1

    def __call__(self, x):
        return [self.base_transform_q(x) for _ in range(self.times - 1)] + [self.base_transform_k(x)]


class MultiCropsTransform2:
    """Take multiple random crops of one image as different views."""

    def __init__(self, base_transform, non_transform, times):
        self.base_transform = base_transform
        self.non_transform = non_transform
        self.times = times
        assert self.times > 1

    def __call__(self, x):
        return [self.base_transform(x) for _ in range(self.times)] + \
               [self.non_transform(x)]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
