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


class conjugateGradient():
    def __init__(self, function, x, eps, fstar, method='FR', verbose = True):
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

        # Logging
        handlerPrint = logging.StreamHandler()
        handlerPrint.setLevel(logging.DEBUG)
        self.log = logging.getLogger("gradient-descent")
        self.log.addHandler(handlerPrint)
        self.log.setLevel(logging.DEBUG)
        
        self.v = -self.function.func_value(self.x)
        self.g = self.function.func_grad(self.x)
        self.pOld = -1 # old value of p
        self.p = -self.g
        self.B = 0 #initial value of Beta
        self.gTg = np.dot(self.g.T, self.g)
        if self.eps < 0: # stopping criteria
            self.ng0 = - np.sqrt(self.gTg)
        else:
            self.ng0 = 1

        #arrays to store history
        self.norm_history = []
        self.function_value_history = []
        self.error_history = []

        def step(self):
            v = self.function.func_value(self.x)
            g = self.function.func_grad(self.x)
            p = -g
            self.norm_history.append(float(np.sqrt(self.gTg)))
            self.hiostyrValue.append(float(self.v))

        def run(self, maxIter):
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
                        self.log.info("optimal value reached")
                    break
                
                self.feval = self.feval + 1

            if self.verbose == True:
                self.log.debug("[end]")
                

            return self.norm_history, self.function_value_history, self.error_history

    def ConjugateGradient(self):
        while True:
            if self.verbose:
                print("Iteration number %d, -f(x) = %0.4f, gradientNorm = %f"%( self.feval, self.v, np.sqrt(self.gTg)))
            # If the norm of the gradient is lower or equal of eps then we stop
            if np.sqrt(self.gTg) <= self.eps * self.ng0:
                self.status = 'optimal'
                return self.historyNorm, self.historyValue

            # If we reach the maximum number of iteration we stop
            if self.feval > self.iterations:
                self.status = 'stopped'
                return self.historyNorm, self.historyValue

            # calculate step along direction
            # -direction because model calculate the derivative of 
            # phi' = f'(x-aplha*d)
            alpha = self.function.stepsizeAlongDirection(-self.p)

            # if the stop is too short we stop
            if alpha <= 1e-16:
                self.status = 'error'
                return self.historyNorm, self.historyValue
            
            # now we will update all the CG variables
            g = self.g
            self.oldgTg  = self.gTg
            self.pOld = self.p
            # lastx = self.x
            # update x 
            self.x = self.x + alpha * self.p
            self.v, self.g = self.function.calculate(self.x)
            self.feval = self.feval + 1
            self.gTg = np.dot(self.g.T, self.g)

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
            
            #self.pOld = self.p
            self.p = -self.g + ((self.pOld)*self.B)

            if self.v <= - float("inf"):
                self.status = 'unbounded'
                return self.historyNorm, self.historyValue



    def exactLineSearch(self, d=None):
        if d is not None:
            self.d = d
            self.dT = d.T

        dTd = np.dot(self.dT, self.d)
        xTd = np.dot(self.xT, self.d)
        self.xTx = np.dot(self.xT, self.x)
        Qd = np.dot(self.Q, self.d)
        xQd = np.dot(self.xT, Qd)
        dQd = np.dot(self.dT, Qd)
        # a = (d.T*d)(x*Q*d) - (d.T*Q*d)*(x.T*d)  
        a = float(dTd * xQd - dQd * xTd)
        # b = (xTx)(dQd) - (dTd)(xQx)
        b = float(self.xTx * dQd - dTd * self.function.xQx)
        # c = (xTd)(xQx) - (xTx)(xQd)
        c = float(xTd * self.function.xQx - self.xTx * xQd)
        # now alpha is the solution of ax^2+bx+c : x > 0 
        coef = np.array([a, b, c])
        roots = np.roots(coef)
        if roots[0] < 0 and roots[1] < 0:
            return 0
        elif roots[0] < 0:
            return roots[1]
        elif roots[1] < 0:
            return roots[0]
        return np.min([roots])
