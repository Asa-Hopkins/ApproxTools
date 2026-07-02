import numpy as np
import scipy
import time

def rotate(d, g, p, q, rotations):
    #Apply all rotations to the right hand side to get a similar matrix to the original
    b = np.zeros_like(g)
    n = p.size
    
    for k in range(n - 1, 0, -1):
        Qk = rotations[k - 1]
        
        d[k - 1], b[k - 1] =  Qk.conj() @ [d[k - 1], -p[k - 1]*q[k].conj()]
        d[k] = (Qk.conj() @ [g[k - 1], d[k]])[1]
        q[k - 1], q[k] = Qk @ [q[k - 1], q[k]]
    return d, b, q, p

def elimination(d, b, p, q):
    #Eliminate the superdiagonal of A + pq* with a series of SU2 rotations
    n = p.size
    
    #sub-diagonal of A
    g = np.copy(b.conj())
    q2 = np.copy(q)

    rotations = []

    for k in range(n-1,0,-1):
        #Find rotation to eliminate k'th superdiagonal element
        x = [b[k-1] + p[k-1]*q[k].conj(), d[k] + p[k]*q[k].conj()]
            
        h  = np.hypot(abs(x[0]), abs(x[1]))
        if h == 0:
            Qk = np.identity(2, dtype = np.complex128)
        else:
            Qk = np.array([[x[1]/h, -x[0]/h],[x[0].conj()/h, x[1].conj()/h]], dtype = np.complex128)

        rotations.append(np.copy(Qk))

        #Perform rotation
        if k != 1:
            g[k - 2] = Qk[0] @ [g[k - 2], -q2[k]*p[k - 2].conj()]


        d[k - 1], g[k - 1] = Qk @ [d[k - 1], g[k - 1]]
        b[k - 1], d[k] = Qk @ [b[k - 1], d[k]]
        p[k - 1], p[k] = Qk @ [p[k - 1], p[k]]
        

        if abs(p[k-1]*q[k].conj())**2 + abs(p[k]*q[k].conj())**2 > abs(b[k-1])**2 + abs(d[k])**2:
            p[k - 1] = -b[k - 1]/q[k].conj()

        q2[k - 1], q2[k] = Qk @ [q2[k - 1], q2[k]]
    return d, g, p, q, rotations[::-1]
        

def QR(d, b, p, q, eps = 1e-15):
    n = p.size
    d = np.array(d, dtype = np.complex128, copy = True)
    b = np.array(b, dtype = np.complex128, copy = True)
    p = np.array(p, dtype = np.complex128, copy = True)
    q = np.array(q, dtype = np.complex128, copy = True)
        
    #Find eigenvalues of A + pq*, where d and b are the diagonal and superdiagonal of A
    #A + pq* must be lower Hessenberg
    for i in range(0, n - 1):
        mu_sum = 0
        
        while abs(b[i] + p[i]*q[i+1].conj()) > eps:
            A = np.array([[d[i] +p[i]*q[i].conj(), b[i] + p[i]*q[i+1].conj()],
                          [b[i].conj() + p[i+1]*q[i].conj(), d[i+1] + p[i+1]*q[i+1].conj()]], dtype = np.complex128)
            
            tr = A[0,0] + A[1,1]
            det = A[0,0]*A[1,1] - A[1,0]*A[0,1]
            
            mu1, mu2 = 0.5*(tr + (tr*tr - 4*det)**0.5), 0.5*(tr - (tr*tr - 4*det)**0.5)
            
            if abs(A[0,0] - mu1) < abs(A[0,0] - mu2):
                mu = mu1
            else:
                mu = mu2
            
            mu_sum += mu
            d[i:] -= mu
            
            a = elimination(d[i:], b[i:], p[i:], q[i:])
            d[i:], b[i:], q[i:], p[i:] = rotate(*a)
            

        d[i:] += mu_sum

    return d + p*q.conj()


#Can use FFT for around N > 500
def monomul(a,b):
    return np.convolve(a,b)

#See "On Polynomial Multiplication in Chebyshev Basis"
#By Pascal Giorgi (2010)
#He provides C++ code that reduces chebyshev mult to two monomial mults
def chebmul(a,b):
    a = np.array(a, dtype = np.complex128)
    b = np.array(b, dtype = np.complex128)
    da = a.size
    db = b.size
    #We use T_0 = 1 not T_0 = 1/2 like the code expects, so we correct
    #Could probably rederive the formulae but this is easier
    a[0]*=2

    b[0]*=2

    #Above code seems to assume db = da?
    #I'll fix later

    if db > da:
        a = np.pad(a, [0,db - da])
    if da > db:
        b = np.pad(b, [0,da - db])
    
    r = b[::-1]

    c = monomul(a,b)
    g = monomul(a,r)

    c *= 0.5
    
    #Original code writes "c2" but he meant g
    c[0] += g[da-1] -a[0]*b[0]

    i = np.arange(1, da-1)
    c[i] += 0.5 * (g[da-1 + i] + g[da-1 - i] - a[0]*b[i] - a[i]*b[0])
    
    c[0] /= 2
    b[0] /= 2
    a[0] /= 2
    return c

class Chebyshev:
    def __init__(self, coefficients):
        self.coeffs = np.trim_zeros(np.array(coefficients, dtype = np.complex128),'b')
        self.degree = self.coeffs.size - 1
        self.zeros = None

    def __str__(self):
        return f"{self.coeffs}"
    
    def __repr__(self):
        return f"{self.coeffs}"

    def __add__(self, q):
        if self.degree >= q.degree:
            c = np.copy(self.coeffs)
            c[:q.degree + 1] += q.coeffs
        else:
            c = np.copy(q.coeffs)
            c[:self.degree + 1] += self.coeffs
        return Chebyshev(c)

    def __neg__(self):
        return Chebyshev(-self.coeffs)

    def __sub__(self, q):
        return self + (-q)

    def __mul__(self, q):
        return Chebyshev(chebmul(self.coeffs, q.coeffs))

    def fit(f, n):
        m = n
        n = 1<<(n-1).bit_length()
        #FFT based interpolation at Chebyshev nodes
        x = np.cos((2*np.arange(1,n+1, dtype=np.float64) - 1) * np.pi/2/n)
        points = np.concatenate((f(x), np.zeros(n)))
        b = scipy.fft.ifft(points)
        b = 4*np.exp(1j*np.pi*np.arange(0,n)/2/n)*b[:n]
        b[0] /= 2
        return Chebyshev(b[:m].real)

    def fit2(f, n):
        m = n
        n = 1<<(n-1).bit_length()
        #FFT based interpolation at Chebyshev nodes
        x = np.cos((2*np.arange(1,n+1, dtype=np.longdouble) - 1) * np.pi/2/n)
        b = scipy.fft.dct(f(x))/n
        b[0] /= 2
        return Chebyshev(b[:m].real)

    def eval(self, x):
        #Clenshaw algorithm for evaluating at a point
        n = self.degree + 1
        c = self.coeffs
        if n == 1:
            return c[0]
        b2 = 0
        b1 = c[n-1]
        
        #Iterate from highest to lowest
        for r in range(n-2, 0, -1):
            b1, b2 = 2*x*b1 - b2 + c[r], b1
        return x*b1 -b2 + c[0]

    def __call__(self, x):
        return self.eval(x)

    def deriv(self):
        #Returns derivatve
        #Uses dT_n/dx = n*U_{n-1}, and U_n = 2*T_n + U_{n - 2}
        #So, essentially replace U_n with 2*T_n, and add to U_{n - 2} term
        coeffs = np.zeros(self.degree, dtype = np.complex128)
        n = self.degree
        
        #Iterate from highest to lowest
        for i in range(n-1, -1,-1):
            coeffs[i] = 2*(i + 1)*self.coeffs[i + 1]
            if i + 2 < n:
                coeffs[i] += coeffs[i + 2]

        #T_1 is just x so the constant term is twice what it should be
        coeffs[0] /= 2
                
        return Chebyshev(coeffs)

    def from_roots(roots):
        roots = np.complex128(roots)
        
        #We do naive way for now
        #Form (x - root) and multiply
        out = Chebyshev([-roots[0],1])
        for i in roots[1:]:
            out = out * Chebyshev([-i,1])

        out.coeffs
        return out

    def trunc(self, error):
        i = -1
        while sum(abs(self.coeffs[i:])) < error:
            i -= 1
        i += 1
        return Chebyshev(self.coeffs[:i])
        
    
    def roots(self, debug = False):
        if (self.zeros is not None) and (debug == False):
            return self.zeros
        #Find eigenvalues of colleague matrix using the stable QR algorithm from Serkh and Rokhlin (2021)
        #Note that paper uses 1-based indexing
        d = self.degree

        if self.degree == 0:
            self.zeros = np.array([])
            return self.zeros

        if self.degree == 1:
            self.zeros = np.array([-self.coeffs[0] / self.coeffs[1]])
            return self.zeros

        #n'th Unit vector
        e_n = np.zeros(d)
        e_n[-1] = 1

        #form q
        c = np.copy(self.coeffs)
        c /= c[-1]
        c = -c[:-1]/2
        c[0] *= 2**0.5
        
        #Form superdiagonal of colleague matrix
        a = np.ones(d - 1)/2
        a[0] *= 2**0.5

        #Form diagonal
        diag = np.zeros(d)

        if debug:
            test = np.diag(diag) + np.diag(a, k =1) + np.diag(a, k = -1) + np.outer(e_n, c.conj())
            self.zeros = np.linalg.eigvals(test)
            return self.zeros
            
        self.zeros = QR(diag, a, e_n, c)
        return self.zeros
