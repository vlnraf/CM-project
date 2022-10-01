import numpy as np


class Norm():
    def __init__(self, A):
        self.A = A
        self.Q = np.matmul(np.transpose(A), A)
        self.dim = self.Q.shape[0]


    def func_value(self,x):
        self.xT = x.T
        self.xTx = np.matmul(self.xT, x)
        self.Qx = np.matmul(self.Q, x)
        self.xQx = np.matmul(self.xT, self.Qx)
        fx = self.xQx / self.xTx
        return -fx

    def func_grad(self, x):
        self.xT = x.T
        self.xTx = np.matmul(self.xT, x)
        self.Qx = np.matmul(self.Q, x)
        self.xQx = np.matmul(self.xT, self.Qx)
        f_x = self.xQx / self.xTx
        nabla_f = (2 * x * f_x) / self.xTx - (2 * self.Qx) / self.xTx

        return nabla_f


    def compute(self, x):
        self.x = x
        self.f_x = self.func_value(self.x) 
        nabla_f = self.func_grad(self.x)
        return -self.f_x, nabla_f

    def init_x(self):
        return np.random.rand(self.dim, 1)
