import numpy as np
import logging
from numpy import linalg as la

class NoUpdate(Exception):
    pass
    # def print(self):
    #     log.info("optimal value reached")

class MaxIterations(Exception):
    pass
    # def print(self):
    #     log.info("Stopped for iterations")

class InvalidAlpha(Exception):
    pass
    # def print(self):
    #     log.info("Alpha too much small")

class UnboundedFunction(Exception):
    pass
    # def print(self):
    #     log.info("The function is unbounded")
class ValueError(Exception):
    pass


class conjugateGradient():
    def __init__(self, function, x, eps, fstar, method='FR', verbose = True):
        #arrays to store history
        self.norm_history = []
        self.function_value_history = []
        self.error_history = []

        self.verbose = verbose
        self.function = function
        self.feval = 1
        self.eps = eps
        self.fstar = fstar
        self.x = x
        self.Q = function.Q
        self.prev_value = 0
        self.ratek = 0
        self.method = method

        self.v = -self.function.func_value(self.x)
        self.g = self.function.func_grad(self.x)
        self.pOld = -1 # old value of p
        self.p = -self.g
        self.B = 0 #initial value of Beta
        self.gTg = np.dot(self.g.T, self.g)
        self.ng = la.norm(self.gTg) 
        if self.eps < 0: # stopping criteria
            self.ng0 = - np.sqrt(self.gTg)
        else:
            self.ng0 = 1

        

    def step(self):
        # self.v = self.function.func_value(self.x)
        # p = -g
        self.norm_history.append(float(np.sqrt(self.gTg)))
        self.function_value_history.append(float(self.v))
        self.error_history.append(float(abs(self.v - self.fstar) / abs(self.fstar)))

        g = self.function.func_grad(self.x)
        #     # Da aggiustare un po' il rate di convergenza
        #     #==============================================
        if self.prev_value != 0:
            self.ratek = (self.v - self.fstar) / (self.prev_value - self.fstar)

        self.prev_value = self.v
        #     #==============================================

        alpha = self.exactLineSearch(-self.p)

        if self.ng <= self.eps * self.ng0:
            raise NoUpdate()

        # if we iterate more times then maxIter we stop
        if self.feval >= self.maxIter:
            raise MaxIterations()

        # step too short
        # if alpha <= conf.mina:
        if alpha <= 1e-16:
            raise InvalidAlpha()

        # now we will update all the CG variables
        g = self.g
        self.oldgTg  = self.gTg
        self.pOld = self.p

        self.x = self.x + alpha * self.p
        self.v = - self.function.func_value(self.x)
        self.g = self.function.func_grad(self.x)
        self.feval = self.feval + 1
        self.gTg = np.dot(self.g.T, self.g)
        self.ng = la.norm(self.gTg) 

        # calculate beta.
        if self.method == 'FR':
            self.B = self.gTg / self.oldgTg
        elif self.method == 'PR':
            y_hat = self.g - g
            self.B  = np.dot(self.g.T, y_hat) / self.oldgTg 
        elif self.method == 'HS':
            y_hat = self.g - g
            self.B  = np.dot(self.g.T, y_hat) / np.dot(self.p.T, y_hat)
        else:
            raise ValueError('Method not implemented')
        
        self.p = -self.g + ((self.pOld)*self.B)

        # Unbounded function controll
        if self.v <= - float("inf"):
            raise UnboundedFunction()

        if self.verbose == True:
            print("iteration %d, f(x) = %0.4f, ||gradient(f(x))|| = %f, alpha=%0.4f, rate=%0.4f" %(self.feval, self.v, self.gTg, alpha, self.ratek)) 



    def run(self, maxIter=500):
        assert maxIter > 1

        self.maxIter = maxIter

        if self.verbose == True:
            print("[start]")

        for self.feval in range(0,maxIter+1):
            try:
                self.step()
            except UnboundedFunction:
                if self.verbose == True:
                    print("The function is unbounded")
            except MaxIterations:
                if self.verbose == True:
                    print("Stopped for iterations")
            except InvalidAlpha:
                if self.verbose == True:
                    print("Alpha too much small")
            except NoUpdate:
                if self.verbose == True:
                    print("optimal value reached")
                break
            
            self.feval = self.feval + 1

        if self.verbose == True:
            print("[end]")
            

        return self.norm_history, self.function_value_history, self.error_history


    def exactLineSearch(self, d):
        self.d = d

        dTd = np.matmul(self.d.T, self.d)
        xTd = np.matmul(self.x.T, self.d)
        self.xTx = np.matmul(self.x.T, self.x)
        Qd = np.matmul(self.Q, self.d)
        xQd = np.matmul(self.x.T, Qd)
        dQd = np.matmul(self.d.T, Qd)
        a = float(dTd * xQd - dQd * xTd)
        b = float(self.xTx * dQd - dTd * self.function.xQx)
        c = float(xTd * self.function.xQx - self.xTx * xQd)
        coef = np.array([a, b, c])
        roots = np.roots(coef)
        if roots[0] < 0 and roots[1] < 0:
            return 0
        elif roots[0] < 0:
            return roots[1]
        elif roots[1] < 0:
            return roots[0]
        return np.min([roots])
