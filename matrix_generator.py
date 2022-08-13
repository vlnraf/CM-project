import numpy as np
import numpy.linalg as la
import scipy.sparse as sparse

np.random.seed(42)

def generate_matrix():
    PATH = './matrix/'
    M1 = sparse.random(500, 100, density=1, data_rvs=np.random.randn)
    M1 = M1.toarray()
    np.savetxt(PATH + "M1.txt", M1)
    print(" Matrix generated and saved.")

    M2 = sparse.random(2000, 50, density=1, data_rvs=np.random.randn)
    M2 = M2.toarray()
    np.savetxt(PATH + "M2.txt", M2)
    print(" Matrix generated and saved.")

    M3 = sparse.random(2000, 50, density=0.25, data_rvs=np.random.randn)
    M3 = M3.toarray()
    np.savetxt(PATH + "M3.txt", M3)
    print(" Matrix generated and saved.")

    M4 = sparse.random(2000, 1000, density=1, data_rvs=np.random.randn)
    M4 = M4.toarray()
    np.savetxt(PATH + "M4.txt", M4)
    print(" Matrix generated and saved.")

    return 0

def generate_cond_matrix(n, cond_p=1e5):
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
    np.savetxt(PATH + "M5.txt", P)
    print(" Matrix generated and saved.")

    return P

def main():
    generate_matrix()
    generate_cond_matrix(1000)

if __name__ == "__main__":
    main()
