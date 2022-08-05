import numpy as np
import numpy.linalg as la

def generate_matrix(n, cond_p=10**2):
    # cond_P = 10**2     # Condition number
    log_cond_P = np.log(cond_p)
    exp_vec = np.arange(-log_cond_P/4., log_cond_P * (n)/(4 * (n - 1)), log_cond_P/(2.*(n-1)))
    s = np.exp(exp_vec)
    S = np.diag(s)
    U, _ = la.qr((np.random.rand(n, n) - 5.) * 200)
    V, _ = la.qr((np.random.rand(n, n) - 5.) * 200)
    P = U.dot(S).dot(V.T)
    P = P.dot(P.T)

    return P
