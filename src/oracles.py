import numpy as np
import scipy

_LOG2 = np.log(2.0)
_EXP_CLIP = 709.0


class _OraclePointView(object):
    __slots__ = ('_oracle', '_x')

    def __init__(self, oracle, x):
        self._oracle = oracle
        self._x = np.asarray(x, dtype=float)

    def grad(self):
        return self._oracle.grad(self._x)

    def hess(self):
        return self._oracle.hess(self._x)

    def func(self):
        return self._oracle.func(self._x)


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))

    def __call__(self, x):
        return _OraclePointView(self, x)


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A 


class NonConvexOracle(BaseSmoothOracle):
    """
    Oracle for test function from your assignment (функция Била, 2D).

    f(x, y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2,
    минимум в точке (3, 0.5), f* = 0.
    """

    def __init__(self):
        pass

    def func(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        xv, yv = x[0], x[1]
        u = 1.5 - xv + xv * yv
        v = 2.25 - xv + xv * yv * yv
        w = 2.625 - xv + xv * yv ** 3
        return u * u + v * v + w * w

    def grad(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        xv, yv = x[0], x[1]
        u = 1.5 - xv + xv * yv
        v = 2.25 - xv + xv * yv * yv
        w = 2.625 - xv + xv * yv ** 3
        ux, uy = yv - 1.0, xv
        vx, vy = yv * yv - 1.0, 2.0 * xv * yv
        wx, wy = yv ** 3 - 1.0, 3.0 * xv * yv * yv
        gx = 2.0 * (u * ux + v * vx + w * wx)
        gy = 2.0 * (u * uy + v * vy + w * wy)
        return np.array([gx, gy], dtype=float)

    def hess(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        xv, yv = x[0], x[1]
        u = 1.5 - xv + xv * yv
        v = 2.25 - xv + xv * yv * yv
        w = 2.625 - xv + xv * yv ** 3
        ux, uy = yv - 1.0, xv
        vx, vy = yv * yv - 1.0, 2.0 * xv * yv
        wx, wy = yv ** 3 - 1.0, 3.0 * xv * yv * yv
        uxy, vxy, wxy = 1.0, 2.0 * yv, 3.0 * yv * yv
        uyy, vyy, wyy = 0.0, 2.0 * xv, 6.0 * xv * yv
        hxx = 2.0 * (ux * ux + vx * vx + wx * wx)
        hyy = 2.0 * (
            uy * uy + u * uyy + vy * vy + v * vyy + wy * wy + w * wyy
        )
        hxy = 2.0 * (
            uy * ux + u * uxy + vy * vx + v * vxy + wy * wx + w * wxy
        )
        return np.array([[hxx, hxy], [hxy, hyy]], dtype=float)


class LogCoshL2Oracle(BaseSmoothOracle):
    """
    Oracle for Log-Cosh loss function with l2 regularization:
         check your individual assignment

    Let A and b be parameters of the model (feature matrix
    and labels vector respectively).   

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = np.asarray(b, dtype=float).ravel()
        self.regcoef = float(regcoef)
        self._m = max(int(self.b.size), 1)

    def func(self, x):
        r = self.matvec_Ax(x) - self.b
        lc = np.logaddexp(r, -r) - _LOG2
        return np.mean(lc) + (self.regcoef / 2.0) * np.dot(x, x)

    def grad(self, x):
        r = self.matvec_Ax(x) - self.b
        t = np.tanh(r)
        return (1.0 / self._m) * self.matvec_ATx(t) + self.regcoef * x

    def hess(self, x):
        r = self.matvec_Ax(x) - self.b
        sech2 = 1.0 - np.tanh(r) ** 2
        s = sech2 / self._m
        return self.matmat_ATsA(s) + self.regcoef * np.eye(len(x))


class ExponentialLossL2Oracle(BaseSmoothOracle):
    """
    Oracle for Exponential loss function with l2 regularization:
         check your individual assignment

    Let A and b be parameters of the model (feature matrix
    and labels vector respectively).   

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = np.asarray(b, dtype=float).ravel()
        self.regcoef = float(regcoef)
        self._m = max(int(self.b.size), 1)

    def func(self, x):
        Ax = self.matvec_Ax(x)
        r = self.b * Ax
        e = np.exp(np.clip(-r, -_EXP_CLIP, _EXP_CLIP))
        return np.mean(e) + (self.regcoef / 2.0) * np.dot(x, x)

    def grad(self, x):
        Ax = self.matvec_Ax(x)
        r = self.b * Ax
        e = np.exp(np.clip(-r, -_EXP_CLIP, _EXP_CLIP))
        w = -(self.b * e) / self._m
        return self.matvec_ATx(w) + self.regcoef * x

    def hess(self, x):
        Ax = self.matvec_Ax(x)
        r = self.b * Ax
        e = np.exp(np.clip(-r, -_EXP_CLIP, _EXP_CLIP))
        s = e / self._m
        return self.matmat_ATsA(s) + self.regcoef * np.eye(len(x))


def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences:
        result_i := (f(x + eps * e_i) - f(x)) / eps,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    grad = np.zeros(n)

    fx = func(x)

    for i in range(n):
        x_eps = x.copy()
        x_eps[i] += eps
        grad[i] = (func(x_eps) - fx) / eps

    return grad


def hess_finite_diff(func, x, eps=1e-5):
    """
    Returns approximation of the Hessian using finite differences:
        result_{ij} := (f(x + eps * e_i + eps * e_j)
                               - f(x + eps * e_i) 
                               - f(x + eps * e_j)
                               + f(x)) / eps^2,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    hess = np.zeros((n, n))

    fx = func(x)

    f_x_ei = np.zeros(n)
    for i in range(n):
        x_eps = x.copy()
        x_eps[i] += eps
        f_x_ei[i] = func(x_eps)

    for i in range(n):
        for j in range(i, n):
            x_eps = x.copy()
            x_eps[i] += eps
            x_eps[j] += eps

            fij = func(x_eps)

            hess_ij = (fij - f_x_ei[i] - f_x_ei[j] + fx) / (eps ** 2)

            hess[i, j] = hess_ij
            hess[j, i] = hess_ij

    return hess
