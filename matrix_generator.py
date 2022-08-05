import numpy as np
import numpy.linalg as la

def generate_matrix(n, m, matrix_name):
    PATH = './matrix/'
    M1 = np.random.randn(n, m)
    np.savetxt(PATH + "" + matrix_name + ".txt", M1)
    print(" Matrix generated and saved.")


def generate_cond_matrix(n, cond_p=10**2):
    # cond_P = 10**2     # Condition number
    log_cond_P = np.log(cond_p)
    exp_vec = np.arange(-log_cond_P/4., log_cond_P * (n)/(4 * (n - 1)), log_cond_P/(2.*(n-1)))
    s = np.exp(exp_vec)
    S = np.diag(s)
    U, _ = la.qr((np.random.rand(n, n) - 5.) * 200)
    V, _ = la.qr((np.random.rand(n, n) - 5.) * 200)
    P = U.dot(S).dot(V.T)
    P = P.dot(P.T)


    PATH = './matrix/'
    np.savetxt(PATH + "Matrix_cond.txt", P)
    print(" Matrix generated and saved.")

    return P
