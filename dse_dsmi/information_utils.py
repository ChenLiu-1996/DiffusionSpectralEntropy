import numpy as np
import scipy.sparse as ss
import matplotlib.pyplot as plt


def exact_eigvals(A: np.array):
    '''
    Compute the exact eigenvalues.
    '''
    if np.allclose(A, A.T, rtol=1e-5, atol=1e-8):
        # Symmetric matrix.
        eigenvalues = np.linalg.eigvalsh(A)
    else:
        eigenvalues = np.linalg.eigvals(A)

    return eigenvalues


def exact_eig(A: np.array):
    '''
    Compute the exact eigenvalues & vecs.
    '''

    #return np.ones(A.shape[0]), np.ones((A.shape[0],A.shape[0]))
    if np.allclose(A, A.T, rtol=1e-5, atol=1e-8):
        # Symmetric matrix.
        eigenvalues_P, eigenvectors_P = np.linalg.eigh(A)
    else:
        eigenvalues_P, eigenvectors_P = np.linalg.eig(A)

    # Sort eigenvalues
    sorted_idx = np.argsort(eigenvalues_P)[::-1]
    eigenvalues_P = eigenvalues_P[sorted_idx]
    eigenvectors_P = eigenvectors_P[:, sorted_idx]

    return eigenvalues_P, eigenvectors_P

def approx_eigvals(A: np.array, filter_thr: float = 1e-3):
    '''
    Estimate the eigenvalues of a matrix `A` using
    Chebyshev approximation of the eigenspectrum.

    Assuming the eigenvalues of `A` are within [-1, 1].

    There is no guarantee the set of eigenvalues are accurate.
    '''

    matrix = A.copy()
    N = matrix.shape[0]

    if filter_thr is not None:
        matrix[np.abs(matrix) < filter_thr] = 0

    # Chebyshev approximation of eigenspectrum.
    eigs, cdf = estimate_dos(matrix)

    # CDF to PDF conversion.
    pdf = np.zeros_like(cdf)
    for i in range(len(cdf) - 1):
        pdf[i] = cdf[i + 1] - cdf[i]

    # Estimate the set of eigenvalues.
    counts = N * pdf / np.sum(pdf)
    eigenvalues = []
    for i, count in enumerate(counts):
        if np.round(count) > 0:
            eigenvalues += [eigs[i]] * int(np.round(count))

    eigenvalues = np.array(eigenvalues)

    return eigenvalues

def estimate_dos(A, pflag=False, npts=1001):
    """ Estimate the density of states of the matrix A

    A should be a matrix of with eigenvalues in tha range [-1, 1].
    """
    c = moments_cheb_dos(A, A.shape[0], N=50)[0]
    return plot_chebint((c,), pflag=pflag, npts=npts)

def moments_cheb_dos(A, n, nZ=100, N=10, kind=1):
    """
    Compute a column vector of Chebyshev moments of the form c(k) = tr(T_k(A))
    for k = 0 to N-1. This routine does no scaling; the spectrum of A should
    already lie in [-1,1]. The traces are computed via a stochastic estimator
    with nZ probe

    Args:
        A: Matrix or function apply matrix (to multiple RHS)
        n: Dimension of the space
        nZ: Number of probe vectors with which we compute moments
        N: Number of moments to compute
        kind: 1 or 2 for first or second kind Chebyshev functions
                (default = 1)

    Output:
        c: a column vector of N moment estimates
        cs: standard deviation of the moment estimator
                (std/sqrt(nZ))
    """

    # Create a function handle if given a matrix
    if callable(A):
        Afun = A
    else:
        if isinstance(A, np.ndarray):
            A = ss.csr_matrix(A)

        def Afun(x):
            return A * x

    if N < 2:
        N = 2

    # Set up random probe vectors (allowed to be passed in)
    if not isinstance(nZ, int):
        Z = nZ
        nZ = Z.shape[1]
    else:
        Z = np.sign(np.random.randn(n, nZ))

    # Estimate moments for each probe vector
    cZ = moments_cheb(Afun, Z, N, kind)
    c = np.mean(cZ, 1)
    cs = np.std(cZ, 1, ddof=1) / np.sqrt(nZ)

    c = c.reshape([N, -1])
    cs = cs.reshape([N, -1])
    return c, cs

def moments_cheb(A, V, N=10, kind=1):
    """
    Compute a column vector of Chebyshev moments of the form c(k) = v'*T_k(A)*v
    for k = 0 to N-1. This routine does no scaling; the spectrum of A should
    already lie in [-1,1]

    Args:
        A: Matrix or function apply matrix (to multiple RHS)
        V: Starting vectors
        N: Number of moments to compute
        kind: 1 or 2 for first or second kind Chebyshev functions
                (default = 1)

    Output:
        c: a length N vector of moments
    """

    if N < 2:
        N = 2

    if not isinstance(V, np.ndarray):
        V = V.toarray()

    # Create a function handle if given a matrix
    if callable(A):
        Afun = A
    else:
        if isinstance(A, np.ndarray):
            A = ss.csr_matrix(A)

        def Afun(x):
            return A * x

    n, p = V.shape
    c = np.zeros((N, p))

    # Run three-term recurrence to compute moments
    TVp = V  # x
    TVk = kind * Afun(V)  # Ax
    c[0] = np.sum(V * TVp, 0)  # xx
    c[1] = np.sum(V * TVk, 0)  # xAx
    for i in range(2, N):
        TV = 2 * Afun(TVk) - TVp  # A*2T_1 - T_o
        TVp = TVk
        TVk = TV
        c[i] = sum(V * TVk, 0)
    return c

def plot_chebint(varargin, npts=1001, pflag=True):
    """
    Given a (filtered) set of first-kind Chebyshev moments, compute the integral
    of the density:
            int_0^s (2/pi)*sqrt(1-x^2)*( c(0)/2+sum_{n=1}^{N-1}c_nT_n(x) )
    Output a plot of cumulative density function by default.

    Args:
        c: Array of Chebyshev moments (on [-1,1])
        xx: Evaluation points (defaults to mesh of 1001 pts)
        ab: Mapping parameters (default to identity)
        pflag: Option to output the plot

    Output:
        yy: Estimated cumulative density up to each xx point
    """

    # Parse arguments
    c, xx, xx0, ab = plot_cheb_argparse(npts, *varargin)

    N = len(c)
    txx = np.arccos(xx)
    yy = c[0] * (txx - np.pi) / 2
    for idx in np.arange(1, N):
        yy += c[idx] * np.sin(idx * txx) / idx

    yy *= -2 / np.pi

    # Plot by default
    if pflag:
        plt.plot(xx0, yy)
        # plt.ion()
        plt.show()
        # plt.pause(1)
        # plt.clf()

    return [xx0, yy]

def plot_cheb_argparse(npts, c, xx0=-1, ab=np.array([1, 0])):
    """
    Handle argument parsing for plotting routines. Should not be called directly
    by users.

    Args:
        npts: Number of points in a default mesh
        c: Vector of moments
        xx0: Input sampling mesh (original coordinates)
        ab: Scaling map parameters

    Output:
        c: Vector of moments
        xx: Input sampling mesh ([-1,1] coordinates)
        xx0: Input sampling mesh (original coordinates)
        ab: Scaling map parameters
    """

    if isinstance(xx0, int):
        # only c is given
        xx0 = np.linspace(-1 + 1e-8, 1 - 1e-8, npts)
        xx = xx0
    else:
        if len(xx0) == 2:
            # parameters are c, ab
            ab = xx0
            xx = np.linspace(-1 + 1e-8, 1 - 1e-8, npts)
            xx0 = ab[0] * xx + ab[1]
        else:
            # parameteres are c, xx0
            xx = xx0

    # All parameters specified
    if not (ab == [1, 0]).all():
        xx = (xx0 - ab[1]) / ab[0]

    return c, xx, xx0, ab
