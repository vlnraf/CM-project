import numpy as np

class conjugateGradient():
    def __init__(self, function, eps, iterations, method='FR', x = None, verbose = True):
        self.verbose = verbose
        self.function = function
        self.feval = 1
        self.x = x if x is not None else self.function.init_x()
        self.eps = eps
        self.iterations = iterations
        self.method = method
        
        self.v, self.g = function.calculate(self.x)
        # self.pOld = old value of p (see CG if you want to know what is p)
        # self.p = new value of p
        self.pOld = -1 # it's the old value of p take a look at the theory of the conjugate gradient to understand what is p
        self.p = -self.g
        self.B = 0 #initial value of Beta
        self.gTg = np.matmul(self.g.T, self.g)
        if self.eps < 0: # stopping criteria
            self.ng0 = - np.sqrt(self.gTg)
        else:
            self.ng0 = 1

    def ConjugateGradient(self):
        self.historyNorm = []
        self.historyValue = []
        v, g = self.function.calculate(self.x)
        p = -g
        while True:
            self.historyNorm.append(float(np.sqrt(self.gTg)))
            self.historyValue.append(float(self.v))
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
            self.oldgTg  = self.gTg
            lastx = self.x
            # update x 
            self.x = self.x + alpha * self.p
            self.v, self.g = self.function.calculate(self.x)
            self.feval = self.feval + 1
            self.gTg = np.matmul(self.g.T, self.g)

            # calculate beta.
            if self.method == 'FR':
                self.B = self.gTg / self.oldgTg
            elif self.method == 'PR':
                y_hat = self.g - g
                self.B  = np.dot(self.g.T, y_hat) / np.dot(g.T, g)
            elif self.method == 'HS':
                y_hat = self.g - g
                self.B  = np.dot(self.g.T, y_hat) / np.dot(p.T, y_hat)
            else:
                raise ValueError('Method not implemented')
            
            self.pOld = self.p
            self.p = -self.g + ((self.pOld)*self.B)

            if self.v <= - float("inf"):
                self.status = 'unbounded'
                return self.historyNorm, self.historyValue
