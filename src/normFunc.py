import numpy as np


class Norm():
    def __init__(self, A):
        '''
        Initialize the 2-Norm function on matrix A

        Parameters

            A : Matrix
        '''
        self.A = A
        self.Q = np.matmul(np.transpose(A), A)
        self.dim = self.Q.shape[0]


    def func_value(self,x):
        '''
        Function value to minimize

        Parameters

            x : vector x_k


        Return

            -self.fx : negate function value [f(x) = (x^T M x) / x^T x , where M = A^T A] to deal with a minimization problem
        '''
        self.xT = x.T
        self.xTx = np.matmul(self.xT, x)
        self.Qx = np.matmul(self.Q, x)
        self.xQx = np.matmul(self.xT, self.Qx)
        self.fx = self.xQx / self.xTx
        return -self.fx

    def func_grad(self, x):
        '''
        Gradient of the function

        Parameters

            x : vector x_k


        Return

            self.nabla_f : gradient of -fx
        '''
        self.nabla_f = (2 * x * self.fx) / self.xTx - (2 * self.Qx) / self.xTx
        return self.nabla_f
