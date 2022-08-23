from numpy import linalg as la

class steepestGradientDescent():
    def __init__(self, function, eps, maxIter, x=None, verbose = True):
        self.verbose = verbose
        self.function = function
        self.status = ''
        self.feval = 1
        self.eps = eps
        self.maxIter = maxIter
        self.x = x if x is not None else self.function.init_x()

        self.v, self.g = function.calculate(self.x)
        self.ng = la.norm(self.g)
        # Absolute error or relative error?
        if self.eps < 0:
            self.ng0 = - self.ng
        else:
            self.ng0 = 1

    def steepestGradientDescent(self):
        self.historyNorm = []
        self.historyValue = []
        while True:
            self.historyNorm.append(float(self.ng))
            self.historyValue.append(float(self.v))
            if self.verbose: 
                self.print()

            # Norm of the gradient lower or equal of the epsilon
            if self.ng <= self.eps * self.ng0:
                self.status = 'optimal'
                if self.verbose:
                    self.print()
                return self.historyNorm, self.historyValue


            # Man number of iteration?
            if self.feval > self.maxIter:
                self.status = 'stopped'
                if self.verbose:
                    self.print()
                return self.historyNorm, self.historyValue

            # calculate step along direction
            alpha = self.function.stepsizeAlongDirection()

            # step too short
            # if alpha <= conf.mina:
            if alpha <= 1e-16:
                self.status = 'error'
                if self.verbose:
                    self.print()
                return self.historyNorm, self.historyValue

            # lastx = self.x
            self.x = self.x - alpha * self.g
            self.v, self.g = self.function.calculate(self.x)
            self.feval = self.feval + 1

            # if self.v <= conf.MInf:
            if self.v <= - float("inf"):
                self.status = 'unbounded'
                if self.verbose:
                    self.print()
                return self.historyNorm, self.historyValue

            self.ng = la.norm(self.g)

    def print(self):
        print("Iteration %d, f(x) = %0.4f, Norm of the gradient = %f " % (self.feval, self.v, self.ng) + self.status)
