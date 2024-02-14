import numpy as np

eps = 1e-12
def sigmoid(x):
    return 1 / (1 + np.exp(-x) + eps)

class param:
    def __init__(self, value):
        self.value = value
        self.grad = 0

    def zero_grad(self):
        self.grad = 0

    def __repr__(self):
        return str(self.value)

    def forward(self):
        return self.value

    def __call__(self):
        return self.forward()


class LEAF:
    def __init__(self, init='relu', lr_p1=0.001, lr_p2=0.001, lr_p3=0.001, lr_p4=0.001, size=1):
        super().__init__()
        self.lr_p1 = lr_p1
        self.lr_p2 = lr_p2
        self.lr_p3 = lr_p3
        self.lr_p4 = lr_p4
        if init == 'relu':
            self.p1 = param(np.ones(size) * 1)
            self.p2 = param(np.ones(size) * 0)
            self.p3 = param(np.ones(size) * 2**16)
            self.p4 = param(np.ones(size) * 0)
        elif init == 'tanh':
            self.p1 = param(0)
            self.p2 = param(2)
            self.p3 = param(2)
            self.p4 = param(-1)
        else:
            self.p1 = param(np.random.normal(0, 1, size))
            self.p2 = param(np.random.normal(0, 1, size))
            self.p3 = param(np.random.normal(0, 1, size))
            self.p4 = param(np.random.normal(0, 1, size))

    def forward(self, x):
        return (self.p1() * x + self.p2()) * sigmoid(self.p3() * x) + self.p4()

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        return [self.p1(), self.p2(), self.p3(), self.p4()]

    def p1_grad(self, u):
        return sigmoid(self.p3() * u) * u

    def p2_grad(self, u):
        return sigmoid(self.p3() * u)

    def p3_grad(self, u):
        return (self.p1() * u + self.p2()) * sigmoid(self.p3() * u) * u * sigmoid(- self.p3() * u)

    def p4_grad(self):
        return 1

    def grad(self, x, delta):
        return [self.p1_grad(x) * delta, self.p2_grad(x) * delta, self.p3_grad(x) * delta, self.p4_grad() * delta]

    def backward(self, x, delta):
        return self.grad(x, delta)

    def __repr__(self):
        return f'LEAF(p1={self.p1}, p2={self.p2}, p3={self.p3}, p4={self.p4})'

    def update(self, x, delta):
        grads = self.backward(x, delta)

        self.p1.value -= self.lr_p1 * np.mean(grads[0], axis=0)
        self.p2.value -= self.lr_p2 * np.mean(grads[1], axis=0)
        self.p3.value -= self.lr_p3 * np.mean(grads[2], axis=0)
        self.p4.value -= self.lr_p4 * np.mean(grads[3], axis=0)

    def derivative(self, x):
        # return self.p2() * self.p3() * sigmoid(self.p3() * x) * (1 - sigmoid(self.p3() * x)) + self.p1() * (
        #         sigmoid(self.p3() * x) + x * self.p3() * sigmoid(self.p3() * x) * (1 - sigmoid(self.p3() * x)))
        u = self.p3() * x
        sig_u = sigmoid(u)
        return sig_u * (self.p2() + self.p3() * (self.p1() + x * self.p2()) * (1 - sig_u))



p1 = 0.0253
p2 = -0.10142
p3 = -0.0344
p4 = -0.0699

x = np.arange(-10, 10, 0.1)

y = (p1 * x + p2) * sigmoid(p3 * x) + p4

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

plt.figure()
plt.plot(x, y)
plt.show()
