from torch.nn import Parameter

class Manifold(object):

    def __init__(self):
        super().__init__()
        self.eps = 10e-8

    def sqdist(self, p1, p2, c):
        raise NotImplementedError

    def proj(self, p, c):
        raise NotImplementedError

    def proj_tan(self, u, p, c):
        raise NotImplementedError

    def proj_tan0(self, u, c):
        raise NotImplementedError

    def expmap(self, u, p, c):
        raise NotImplementedError

    def logmap(self, p1, p2, c):
        raise NotImplementedError

    def expmap0(self, u, c):
        raise NotImplementedError

    def logmap0(self, p, c):
        raise NotImplementedError

    def mobius_add(self, x, y, c, dim=-1):
        raise NotImplementedError

    def mobius_matvec(self, m, x, c):
        raise NotImplementedError

    def init_weights(self, w, c, irange=1e-5):
        raise NotImplementedError

    def inner(self, p, c, u, v=None, keepdim=False):
        raise NotImplementedError

    def ptransp(self, x, y, u, c):
        raise NotImplementedError

    def ptransp0(self, x, u, c):
        raise NotImplementedError


class ManifoldParameter(Parameter):
    """
    Subclass of torch.nn.Parameter for Riemannian optimization.
    """
    def __new__(cls, data, requires_grad, manifold, c):
        return Parameter.__new__(cls, data, requires_grad)

    def __init__(self, data, requires_grad, manifold, c):
        self.c = c
        self.manifold = manifold

    def __repr__(self):
        return '{} Parameter containing:\n'.format(self.manifold.name) + super(Parameter, self).__repr__()