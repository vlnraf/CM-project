import numpy as np
from numpy import linalg as la

class NoUpdate(Exception):
    pass

class MaxIterations(Exception):
    pass

class InvalidAlpha(Exception):
    pass

class UnboundedFunction(Exception):
    pass

class ValueError(Exception):
    pass


class conjugateGradient():
    def __init__(self, function, x, eps, fstar, method='FR', verbose = True):
        '''
        Initialize the conjugate gradient class

        Parameters

            function : function to minimize [f(x) = (x^T M x) / x^T x , where M = A^T A]
            x : starting point
            eps : stopping criteria
            fstar : optimal value of the function
            method : "PR"
                     "FR"
            verbose : to print values
        '''
        self.norm_history = []
        self.function_value_history = []
        self.error_history = []

        self.verbose = verbose
        self.function = function
        self.iter = 1
        self.eps = eps
        self.fstar = fstar
        self.x = x
        self.Q = function.Q
        self.prev_value = 0
        self.ratek = 0
        self.method = method

        self.f_value = -self.function.func_value(self.x)
        self.f_gradient = self.function.func_grad(self.x)
        self.prev_p = -1
        self.p = -self.f_gradient
        self.B = 0
        self.gTg = np.dot(self.f_gradient.T, self.f_gradient)
        self.gradient_norm = la.norm(self.f_gradient) 
        
        if self.eps < 0:
            self.initial_gradient_norm = - self.gradient_norm
        else:
            self.initial_gradient_norm = 1

        

    def step(self):
        '''
        Conjugate gradient step to do at every iteration
        '''
        self.norm_history.append(float(self.gradient_norm))
        self.function_value_history.append(float(self.f_value))
        self.error_history.append(float(abs(self.f_value - self.fstar) / abs(self.fstar)))

        prev_g = self.function.func_grad(self.x)
        if self.prev_value != 0:
            self.ratek = (self.f_value - self.fstar) / (self.prev_value - self.fstar)

        self.prev_value = self.f_value

        alpha = self.exactLineSearch(-self.p)
        self.gradient_norm = la.norm(self.f_gradient) 

        if self.gradient_norm <= self.eps * self.initial_gradient_norm:
            raise NoUpdate()

        if self.iter >= self.maxIter:
            raise MaxIterations()

        if alpha <= 1e-16:
            raise InvalidAlpha()

        prev_g = self.f_gradient
        self.prev_gTg  = self.gTg
        self.prev_p = self.p

        self.x = self.x + alpha * self.p
        self.f_value = - self.function.func_value(self.x)
        self.f_gradient = self.function.func_grad(self.x)
        self.iter = self.iter + 1
        self.gTg = np.dot(self.f_gradient.T, self.f_gradient)

        if self.method == 'FR':
            self.B = self.gTg / self.prev_gTg
        elif self.method == 'PR':
            y_hat = self.f_gradient - prev_g
            self.B  = np.dot(self.f_gradient.T, y_hat) / self.prev_gTg 
        else:
            raise ValueError('Method not implemented')
        
        self.p = -self.f_gradient + ((self.prev_p)*self.B)

        if self.f_value <= - float("inf"):
            raise UnboundedFunction()

        if self.verbose == True:
            print("iteration %d, f(x) = %0.4f, ||gradient(f(x))|| = %f, alpha=%0.4f, rate=%0.4f" %(self.iter, self.f_value, self.gTg, alpha, self.ratek)) 



    def run(self, maxIter=500):
        '''
        Function to run the algorithm

        Parameters

            maxIter : max number of iterations before stopping the algorithm


        Return

            self.norm_history : array containing the values of the norm of the gradient until stop
            self.function_value_history : array containing the values of the function norm until stop
            self.error_history : array containing the values of errors between the function value at iteration k and the optimal value until stop
        '''
        assert maxIter > 1

        self.maxIter = maxIter

        if self.verbose == True:
            print("[start]")

        for self.iter in range(0,maxIter+1):
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
            
            self.iter = self.iter + 1

        if self.verbose == True:
            print("[end]")
            

        return self.norm_history, self.function_value_history, self.error_history


    def exactLineSearch(self, d):
        '''
        Exact line serach to minimize the function phi(alpha)

        Parameters

            d : direction


        Return

            alpha : return the value of alpha
        '''
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
