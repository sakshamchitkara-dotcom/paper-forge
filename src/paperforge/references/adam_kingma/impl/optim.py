"""Adam (Kingma & Ba, arXiv:1412.6980, Algorithm 1) plus the Sec 6.1 baselines.

Hand-written reference implementation for paper-forge.
"""

import numpy as np


class Adam:
    def __init__(self, alpha=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, decay=False):
        self.alpha, self.b1, self.b2, self.eps, self.decay = alpha, beta1, beta2, eps, decay
        self.m = self.v = None
        self.t = 0

    def step(self, theta, g):
        if self.m is None:
            self.m, self.v = np.zeros_like(theta), np.zeros_like(theta)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        m_hat = self.m / (1 - self.b1 ** self.t)  # bias-corrected first moment
        v_hat = self.v / (1 - self.b2 ** self.t)  # bias-corrected second moment
        alpha = self.alpha / np.sqrt(self.t) if self.decay else self.alpha
        return theta - alpha * m_hat / (np.sqrt(v_hat) + self.eps)


class Adagrad:
    def __init__(self, alpha=1e-2, eps=1e-8, decay=False):
        self.alpha, self.eps, self.decay, self.G, self.t = alpha, eps, decay, None, 0

    def step(self, theta, g):
        if self.G is None:
            self.G = np.zeros_like(theta)
        self.t += 1
        self.G += g * g
        alpha = self.alpha / np.sqrt(self.t) if self.decay else self.alpha
        return theta - alpha * g / (np.sqrt(self.G) + self.eps)


class NesterovSGD:
    """SGD with Nesterov momentum (Sutskever et al. 2013 formulation)."""

    def __init__(self, alpha=1e-2, mu=0.9, decay=False):
        self.alpha, self.mu, self.decay, self.v, self.t = alpha, mu, decay, None, 0

    def step(self, theta, g):
        # g is evaluated at the look-ahead point by the caller via lookahead()
        if self.v is None:
            self.v = np.zeros_like(theta)
        self.t += 1
        alpha = self.alpha / np.sqrt(self.t) if self.decay else self.alpha
        self.v = self.mu * self.v - alpha * g
        return theta + self.v

    def lookahead(self, theta):
        return theta if self.v is None else theta + self.mu * self.v
