from numpy import linalg as la
import numpy as np
import logging


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


class GradientDescent():
    def __init__(self, function, x, eps=1e-5, fstar=0, verbose = True):
        self.verbose = verbose
        self.function = function
        self.feval = 1
        self.eps = eps
        self.fstar = fstar
        self.x = x
        self.Q = function.Q
        self.prev_value = 0
        self.ratek = 0

        # Logging
        handlerPrint = logging.StreamHandler()
        handlerPrint.setLevel(logging.DEBUG)
        self.log = logging.getLogger("gradient-descent")
        self.log.addHandler(handlerPrint)
        self.log.setLevel(logging.DEBUG)


        self.v = -self.function.func_value(self.x)
        self.g = self.function.func_grad(self.x)
        self.ng = la.norm(self.g)
        if self.eps < 0:
            self.ng0 = - self.ng
        else:
            self.ng0 = 1


        #arrays to store history
        self.norm_history = []
        self.function_value_history = []
        self.error_history = []

    def step(self):

        self.norm_history.append(float(self.ng))
        self.function_value_history.append(float(self.v))
        self.error_history.append(float(abs(self.v - self.fstar)/abs(self.fstar)))


        #     # Da aggiustare un po' il rate di convergenza
        #     #==============================================
        if self.prev_value != 0:
            self.ratek = (self.v - self.fstar) / (self.prev_value - self.fstar)

        self.prev_value = self.v
        #     #==============================================
        

        # run the exact line search
        alpha = self.exactLineSearch()

        # Norm of the gradient lower or equal of the epsilon
        if self.ng <= self.eps * self.ng0:
            raise NoUpdate()

        # if we iterate more times then maxIter we stop
        if self.feval >= self.maxIter:
            raise MaxIterations()

        # step too short
        # if alpha <= conf.mina:
        if alpha <= 1e-16:
            raise InvalidAlpha()

        self.x = self.x - alpha * self.g
        self.v = -self.function.func_value(self.x)
        self.g = self.function.func_grad(self.x)

        # Unbounded function controll
        if self.v <= - float("inf"):
            raise UnboundedFunction()

        self.ng = la.norm(self.g)

        if self.verbose == True:
            self.log.debug("iteration %d, f(x) = %0.4f, ||gradient(f(x))|| = %f, alpha=%0.4f, rate=%0.4f" %(self.feval, self.v, self.ng, alpha, self.ratek)) 

    def run(self, maxIter=500):

        assert maxIter > 1

        self.maxIter = maxIter

        if self.verbose == True:
            self.log.debug("[start]")

        for self.feval in range(0,maxIter+1):
            try:
                self.step()
            except UnboundedFunction:
                if self.verbose == True:
                    self.log.info("The function is unbounded")
            except MaxIterations:
                if self.verbose == True:
                    self.log.info("Stopped for iterations")
            except InvalidAlpha:
                if self.verbose == True:
                    self.log.info("Alpha too much small")
            except NoUpdate:
                if self.verbose == True:
                    self.log.info("The function is unbounded")
                break
            
            self.feval = self.feval + 1

        if self.verbose == True:
            self.log.debug("[end]")
            

        return self.norm_history, self.function_value_history, self.error_history


    def exactLineSearch(self):
        self.d = self.g

        dTd = np.dot(self.d.T, self.d)
        xTd = np.dot(self.x.T, self.d)
        self.xTx = np.dot(self.x.T, self.x)
        Qd = np.dot(self.Q, self.d)
        xQd = np.dot(self.x.T, Qd)
        dQd = np.dot(self.d.T, Qd)
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
