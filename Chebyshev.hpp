#pragma once

#ifdef use_otfft
#include <otfftpp/otfft.h>
#endif
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/FFT>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

int next_power(int n) {
  int m = n;
  // Taken from "Bit Twiddling Hacks" by Sean Eron Anderson
  // rounds m to the next power of 2 greater or equal to it
  m--;
  m |= m >> 1;
  m |= m >> 2;
  m |= m >> 4;
  m |= m >> 8;
  m |= m >> 16;
  m++;
  return m;
}

#ifdef use_otfft
#ifdef fft_cache
#include <memory>
#include <unordered_map>
OTFFT::RealDCT &get_thread_local_dct(size_t n) {
  thread_local std::unordered_map<size_t, std::unique_ptr<OTFFT::RealDCT>>
      cache;

  auto it = cache.find(n);
  if (it != cache.end()) {
    return *it->second;
  }

  // Create and cache a new DCT plan
  auto dct = OTFFT::createDCT(n);
  auto &ref = *dct;
  cache.emplace(n, std::move(dct));
  return ref;
}
#endif
#endif

using Eigen::seq;

// Eigen 3 complains this is deprecated but Eigen 5 removes the suggested
// alternative
using Eigen::placeholders::last;

// Spectra provides sparse matrix methods, but slows down compilation a lot
#ifdef use_spectra
#include <Spectra/SymEigsSolver.h>
template <typename Scalar_> class HankelOp {
public:
  using Scalar = Scalar_;
  using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using ComplexVec = Eigen::Matrix<std::complex<Scalar>, Eigen::Dynamic, 1>;

private:
  // FFT object must be mutable because Spectra expects perform_op(...) const
  mutable Eigen::FFT<Scalar> fft;
  ComplexVec circle; // frequency-domain embedding (complex)
  mutable Vector padded;
  mutable ComplexVec temp;
  mutable ComplexVec output;
  int n;
  int fft_size;

public:
  // Constructor takes the defining vector h (length n)
  HankelOp(const Vector &h) {
    n = static_cast<int>(h.size());
    fft_size = next_power(2 * n);
    padded.resize(fft_size);
    output.resize(fft_size);
    temp.resize(fft_size);
    circle.resize(fft_size);
    // Reverse h (h[::-1]) and pad to length 2n
    padded.head(n) =
        h.reverse().eval(); // evaluate to avoid expression lifetime issues
    padded.tail(fft_size - n).setZero();

    // Pre-size circle and run forward FFT: circle = fft(padded)
    fft.fwd(circle, padded); // dst (complex) first, src (real) second
  }

  int rows() const { return n; }
  int cols() const { return n; }

  // Matrix-free multiply y = H * x
  // Spectra calls this with pointers; must be const (Spectra API)
  void perform_op(const Scalar *x_in, Scalar *y_out) const {
    // Map input vector
    Eigen::Map<const Vector> xvec(x_in, n);

    // Zero-pad input to length 2n
    padded.head(n) = xvec;
    padded.tail(fft_size - n).setZero();

    // Forward FFT: Xf = fft(padded) (complex)
    fft.fwd(temp, padded); // non-const call => fft is mutable

    // Pointwise multiply in freq domain
    temp.array() *= circle.array();

    // Inverse FFT: ytime = ifft(Yf)
    fft.inv(output, temp);

    // Crop (first n entries), take real part, reverse to match Hankel behavior
    Eigen::Map<Vector>(y_out, n) = output.head(n).real().reverse().eval();
  }
};
#endif

template <typename Scalar = double> class Chebyshev {
public:
  using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using Complex = std::complex<Scalar>;
  using VectorC = Eigen::Matrix<Complex, Eigen::Dynamic, 1>;
  using Matrix2C = Eigen::Matrix<Complex, 2, 2>;
  using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  Vector coeffs;
  unsigned int degree;
  Scalar error;

  // Constructor
  Chebyshev() : coeffs(Vector::Zero(1)), degree(0) {}
  Chebyshev(const Vector &coefficients)
      : coeffs(coefficients),
        degree(static_cast<int>(coefficients.size()) - 1) {}

  // Print coefficients
  friend std::ostream &operator<<(std::ostream &os, const Chebyshev &c) {
    os << c.coeffs.transpose();
    return os;
  }

  // Addition
  Chebyshev operator+(const Chebyshev &q) const {
    if (degree >= q.degree) {
      Vector c = coeffs;
      c.head(q.degree + 1) += q.coeffs;
      return Chebyshev(c);
    } else {
      Vector c = q.coeffs;
      c.head(degree + 1) += coeffs;
      return Chebyshev(c);
    }
  }

  // Negation
  Chebyshev operator-() const { return Chebyshev(-coeffs); }

  // Subtraction
  Chebyshev operator-(const Chebyshev &q) const { return *this + (-q); }

  // Multiplication
  static Vector mono_mult(const Vector &x, const Vector &y,
                          const bool reverse = false) {
    Vector out = Vector::Zero(x.size() + y.size() - 1);

    // We need to replace x[0] and y[0] with 2*x[0] and 2*y[0], but without
    // modifying the actual values To do this, we set the multiplier to 2 whilst
    // i == 0, and handle the j == 0 case outside the loop
    int mult = 2;

    if (reverse) {
      for (int i = 0; i < x.size(); i++) {
        for (int j = 0; j < y.size() - 1; j++) {
          out[i + j] += x[i] * y[y.size() - 1 - j] * mult;
        }
        out[i + y.size() - 1] += x[i] * y[0] * 2 * mult;
        mult = 1;
      }
    }

    else {
      for (int i = 0; i < x.size(); i++) {
        out[i] += x[i] * y[0] * 2 * mult;
        for (int j = 1; j < y.size(); j++) {
          out[i + j] += x[i] * y[j] * mult;
        }
        mult = 1;
      }
    }

    return out;
  }

  // See "On Polynomial Multiplication in Chebyshev Basis"
  // By Pascal Giorgi (2010)
  // He provides C++ code that reduces chebyshev mult to two monomial mults
  Chebyshev operator*(const Chebyshev &q) const {
    int new_degree = q.degree + degree;
    const int da = degree + 1;
    const int db = q.degree + 1;
    const int n = da + db - 1;

    // This method uses T_0 = 0.5 rather than T_0 = 1
    // So we multiply by 2 where a(0) and b(0) appear

    Vector c = mono_mult(coeffs, q.coeffs);

    // I'm not sure why but the result of this convolution needs reversing
    // I do this by swapping the inputs
    Vector g = mono_mult(q.coeffs, coeffs, true);

    // When the two polynomials have different lengths we need to be careful
    // since the original version only works for equal lengths. We can treat
    // the shorter polynomial as having 0s up to the length of the larger one.

    // g(da-1+i) terms
    for (int i = 1; da - 1 + i < g.size(); ++i) {
      c(i) += g(da - 1 + i);
    }

    // g(da-1-i) terms
    for (int i = 1; da - 1 - i >= 0; ++i) {
      c(i) += g(da - 1 - i);
    }

    c *= 0.5;

    //-a(0) * b(i) terms
    for (int i = 1; i < db; ++i) {
      c(i) -= coeffs(0) * q.coeffs(i);
    }

    //-a(i) * b(0) terms
    for (int i = 1; i < da; ++i) {
      c(i) -= coeffs(i) * q.coeffs(0);
    }
    // Final term
    c(0) += g(da - 1) - coeffs(0) * q.coeffs(0) * 4;
    c(0) /= 2;

    return Chebyshev(c);
  }

  // Evaluate at a single point using the Clenshaw algorithm
  Scalar eval(Scalar x) const {
    int n = degree + 1;
    if (n <= 0)
      return Scalar(0);
    if (n == 1)
      return coeffs[0];

    Scalar b2 = Scalar(0);
    Scalar b1 = coeffs[n - 1];

    for (int r = n - 2; r > 0; --r) {
      Scalar tmp = b1;
      b1 = Scalar(2) * x * b1 - b2 + coeffs[r];
      b2 = tmp;
    }
    return x * b1 - b2 + coeffs[0];
  }

  // Allow function-call syntax for evaluation
  Scalar operator()(Scalar x) const { return eval(x); }

  // Evaluate on a vector of points
  // Naive way for now
  Vector eval(const Vector &xs) const {
    Vector ys(xs.size());
    for (int i = 0; i < xs.size(); ++i) {
      ys[i] = eval(xs[i]);
    }
    return ys;
  }

  // returns derivative as Chebyshev object
  // Uses dT_n/dx = n*U_{n-1}, and U_n = 2*T_n + U_{n - 2}
  // So, essentially replace U_n with 2*T_n, and add to U_{n - 2} term
  Chebyshev deriv() const {
    if (degree <= 0) {
      Vector zero = Vector::Zero(1);
      return Chebyshev(zero);
    }

    Vector new_coeffs = Vector::Zero(degree);
    int n = degree;

    for (int i = n - 1; i >= 0; --i) {
      new_coeffs[i] = 2 * (i + 1) * coeffs[i + 1];
      if (i + 2 < n) {
        new_coeffs[i] += new_coeffs[i + 2];
      }
    }
    // T_1 is just x, so its derivative is U_0 and not 2*U_0
    new_coeffs[0] /= 2;
    return Chebyshev(new_coeffs);
  }

  static Chebyshev from_roots(VectorC roots, Scalar tol = 1e-15) {
    // We only support real valued polynomials
    // So we assume all complex roots appear with their conjugate
    // therefore we directly form the quadratic with both roots when we
    // encounter the one with imag(root)>0

    int n = roots.size();

    // Divide and conquer
    if (n > 6) {
      return from_roots(roots.head(n / 2), tol) *
             from_roots(roots.tail(n - n / 2), tol);
    }

    Vector temp(1);
    temp << 1;
    Chebyshev out = Chebyshev(temp);

    Vector linear(2);
    Vector quadratic(3);

    for (auto &c : roots) {
      if (abs(imag(c)) > abs(real(c)) * tol) {
        if (imag(c) > 0) {
          // If the root is c = a+ib, then in monomial basis the quadratic is
          // x^2 - 2ax + a^2 + b^2 This is [a^2+b^2+0.5, -2a, 0.5] in Chebyshev
          // basis
          quadratic << abs(c * c) + 0.5, -2 * real(c), 0.5;
          out = out * Chebyshev(quadratic);
        }
      } else {
        linear << -real(c), 1;
        out = out * Chebyshev(linear);
      }
    }
    return out;
  }

  void trunc(Scalar tol = 1e-15) {
    // Removes trailing coefficients if they're < tol*max
    Scalar max = 0;
    int last = 0;
    for (int i = 0; i < degree + 1; i++) {
      if (abs(coeffs[i]) > max) {
        max = abs(coeffs[i]);
      }
      if (abs(coeffs[i]) > tol * max) {
        last = i;
      }
    }
    degree = last;
    coeffs.conservativeResize(last + 1);
  }

  void trunc_to_error(Scalar max_err = 1e-15, unsigned int extra = 0) {
    // Removes highest degrees up to given error tolerance
    unsigned int n = degree;
    while ((error + abs(coeffs[n]) < max_err) and n > 0) {
      error += abs(coeffs[n]);
      n--;
    }

    // For RCF to work it needs some extra entries
    extra = std::min(degree - n, extra);
    while (extra > 0) {
      n++;
      error -= abs(coeffs[n]);
      extra--;
    }

    coeffs.conservativeResize(n + 1);
    degree = n;
  }

  // FFT-based Chebyshev fit for f: [-1,1] -> R functions
  // Computes Chebyshev expansion of f(x) on [-1,1] to order n

  // It's actually better (more accurate and more consistent timings) to
  // calculate to a higher degree than requested and truncate See chapter 4 of
  // Approximation Theory and Approximation Practice (By Trefethen) for intro on
  // aliasing issues and truncation Theorem 16.1 says that going to higher n and
  // truncating has a Lebesque constant that's reduced by factor pi/2

  // This function uses a method described by Makhoul in "A Fast Cosine
  // Transform in One and Two Dimensions" To calculate a DCT using an FFT
  // without needing to pad the array with zeros. Very simple python
  // implementations can be found in the link below
  // https://dsp.stackexchange.com/questions/2807/fast-cosine-transform-via-fft

  // If OTFFT is available then we just use their DCT implementation directly
  // TODO - better error estimate for non-analytic functions

  static Chebyshev fit(std::function<Scalar(Scalar)> f, unsigned int n,
                       bool trunc = true) {
    if (n == 0)
      return Chebyshev();

    // Make sure there's some extra elements for error estimate
    unsigned int m = n + 10;

    // Could replace this with smarter choice of smooth numbers
    m = next_power(m);
    unsigned int extra = m - n;

    Vector vals(m);

    // We need to sample f(x) the Chebyshev nodes, cos((2*i + 1) * pi / 2 / m)
    // The FFT is so fast that this is a significant part of runtime even for
    // huge n So we use the angle addition formulae to save on calls to std::cos
    const Scalar pi = std::acos(Scalar(-1));

#ifdef use_otfft
    const Scalar delta = pi / m;
#else
    const Scalar delta = (2.0 * pi) / m;
#endif

    const Scalar c0 = std::cos(delta);
    const Scalar s0 = std::sin(delta);

    Scalar c = std::cos(pi / (2 * m));
    Scalar s = std::sin(pi / (2 * m));

    for (unsigned int i = 0; i < m; ++i) {
      Scalar x = c;
      vals[i] = f(x);

      if ((i & 15) != 0) {
        // update using recurrence
        const Scalar c_next = c * c0 - s * s0;
        const Scalar s_next = s * c0 + c * s0;
        c = c_next;
        s = s_next;
      } else {
        // correct every few iterations to mitigate drift in precision
#ifdef use_otfft
        Scalar theta = (Scalar(2 * i + 3) * pi) / Scalar(2 * m);
#else
        int k = 4 * i + 5;
        if (k > 2 * m) {
          k = 4 * m - k;
        }
        Scalar theta = (Scalar(k) * pi) / Scalar(2 * m);
#endif
        c = std::cos(theta);
        s = std::sin(theta);
      }
    }
#ifdef use_otfft
    OTFFT::double_vector cv = vals.data();

#ifdef fft_cache
    auto &fft = get_thread_local_dct(m);
    fft.fwdn(cv);
#else
    auto fft = OTFFT::createDCT(m);
    fft->fwdn(cv);
#endif
    vals *= 2;
#else
    const Complex I(Scalar(0), Scalar(1));

    Eigen::FFT<Scalar> fft;

    // Build array of f(cos(theta_k)) values, but shuffled a bit
    // Normally theta_k = (2*k + 1)*pi/2/m
    for (unsigned int i = 0; i < m; i++) {
      int k = 4 * i + 1;
      if (k > 2 * m) {
        k = 4 * m - k;
      }
      Scalar theta = (Scalar(k) * pi) / Scalar(2 * m);
      vals[i] = f(std::cos(theta));
    }

    VectorC fft_out(m);
    fft.fwd(fft_out, vals);
    // Scale by 2*exp(i*pi*k/(2n))/m to get DCT from FFT
    for (unsigned int k = 0; k < m; ++k) {
      Scalar angle = (pi * Scalar(k)) / Scalar(2 * m);
      Complex mult = std::exp(-I * angle) * (Scalar(2) / m);
      Complex val = fft_out[k] * mult;
      vals[k] = val.real();
    }
#endif
    vals[0] /= Scalar(2);

    if (not trunc) {
      return Chebyshev(vals);
    }

    Chebyshev output = Chebyshev(vals.head(n));

    // Estimate error from extra components
    // Doesn't work well for non-smooth functions
    output.error = vals.tail(extra).cwiseAbs().sum();

    return output;
  }

  static Chebyshev fit_bounded(std::function<Scalar(Scalar)> f,
                               Scalar tolerance = 1e-16, unsigned int n = 16,
                               unsigned int extra = 0) {
    // We essentially keep doubling the number of interpolation points until we
    // converge If user tolerance is too low then we hit precision issues first
    // so that needs to be checked
    Chebyshev output;

    // Get machine epsilon
    Scalar eps = std::numeric_limits<Scalar>::epsilon();

    Scalar tail_coeff = 1e99;
    Scalar oldtail_coeff = 1e99;

    while (true) {
      output = Chebyshev::fit(f, n, false);
      // We care about error relative to the max component for testing
      // convergence
      Scalar max_coeff = output.coeffs.cwiseAbs().maxCoeff();

      // Guarantees on FFT accuracy in the infinity norm suck, which limits our
      // ability to tell if a number is 0 or not see "Error Analysis of Some
      // Operations Involved in the Cooley-Tukey Fast Fourier Transform" By
      // Brisebarre et al.
      Scalar scale = std::max(10 * n * eps, 100 * eps);
      // If the last 5 values are effectively 0, we've converged
      // Please don't pass in sin(T_6(x)) and expect it to work
      tail_coeff = output.coeffs.tail(5).cwiseAbs().maxCoeff();
      if ((scale * max_coeff > tail_coeff) or
          ((oldtail_coeff / tail_coeff < 10) and
           (tail_coeff < 1e-6 * max_coeff))) {
        break;
      } else {
        n *= 2;
      }
      oldtail_coeff = tail_coeff;
    }

    // Now truncate to user tolerance
    output.error = 0;
    output.trunc_to_error(tolerance, extra);

    if (output.error == 0) {
      // If error is 0, we hit floating point limits before the user's tolerance
      // This estimate is larger than the tolerance and lets the user know
      // something went wrong
      output.error = output.coeffs.tail(5).cwiseAbs().maxCoeff();
    }

    return output;
  }

  // TODO Fit polynomial at arbitrary nodes
  // O(n^2) method - set up Lagrange interpolant, evaluate at Chebyshev nodes
  // then call the FFT fit above.

  // TODO - fast multipoint evaluation. I believe this can be done with a
  // non-uniform DCT (NUFFT) The inverse transform might be useful for
  // interpolation at arbitrary nodes too With fast multipoint evaluation, the
  // rootfinder can become O(NlogN) via fast subdivision

  // Finds roots of Chebyshev series polynomial
  // Constructs the "colleague matrix" which has our polynomial as its
  // characteristic polynomial Then exploits its nearly tridiagonal form to find
  // eigenvalues with fast QR iterations I implement the stable QR algorithm
  // from Serkh and Rokhlin (2021) Good convergence - around 2n iterations
  // needed consistently
  // TODO Use explicit vectorisation, it's slower than matlab's O(N^3) method

  // ================= Root finder =================
  VectorC roots() {
    int d = degree;
    if (d <= 0)
      return VectorC();

    if (d == 1) {
      VectorC root(1);
      root << -coeffs[0] / coeffs[1];
      return root;
    }

    // nth unit vector
    Vector e_n = Vector::Zero(d);
    e_n[d - 1] = 1;

    // coeffs (normalised to monic)
    Vector c = coeffs / coeffs[degree];
    c = -c / Scalar(2);
    c[0] *= std::sqrt(Scalar(2));

    // superdiagonal
    Vector a = Vector::Constant(d - 1, Scalar(0.5));
    a[0] *= std::sqrt(Scalar(2));

    // diagonal
    Vector diag = Vector::Zero(d);

    return QR(diag.template cast<Complex>(), a.template cast<Complex>(),
              e_n.template cast<Complex>(), c.head(d).template cast<Complex>());
  }

  // ================= Algorithm 2: Rotate =================
  static void rotate(Eigen::Ref<VectorC> d, Eigen::Ref<VectorC> b,
                     Eigen::Ref<VectorC> p, Eigen::Ref<VectorC> q,
                     Eigen::Ref<VectorC> g,
                     const std::vector<Matrix2C> &rotations) {
    int n = p.size();

    for (int k = n - 1; k > 0; --k) {
      Matrix2C Qk = rotations[k - 1].conjugate();

      auto temp = Qk(0, 0) * d[k - 1] - Qk(0, 1) * p[k - 1] * std::conj(q[k]);
      b[k - 1] = Qk(1, 0) * d[k - 1] - Qk(1, 1) * p[k - 1] * std::conj(q[k]);
      d[k - 1] = temp;

      d[k] = Qk(1, 0) * g[k - 1] + Qk(1, 1) * d[k];

      Qk = Qk.conjugate();

      temp = Qk(0, 0) * q[k - 1] + Qk(0, 1) * q[k];
      q[k] = Qk(1, 0) * q[k - 1] + Qk(1, 1) * q[k];
      q[k - 1] = temp;
    }

    return;
  }

  // ================= Algorithm 1: Elimination =================
  static void elimination(Eigen::Ref<VectorC> d, Eigen::Ref<VectorC> b,
                          Eigen::Ref<VectorC> p, Eigen::Ref<VectorC> q,
                          Eigen::Ref<VectorC> g,
                          std::vector<Matrix2C> &rotations) {
    int n = p.size();
    g = b.conjugate();

    static thread_local VectorC q2;

    q2 = q;

    for (int k = n - 1; k > 0; --k) {
      Complex s = b[k - 1] + p[k - 1] * std::conj(q[k]);
      Complex c = d[k] + p[k] * std::conj(q[k]);

      // I'm aware std::hypot exists but it's like 10x slower
      Scalar h = std::sqrt(std::norm(c) + std::norm(s));

      Matrix2C Qk;
      if (h == 0) {
        Qk.setIdentity();
      } else {
        Qk << c / h, -s / h, std::conj(s) / h, std::conj(c) / h;
      }
      rotations[k - 1] = Qk;

      if (k != 1) {
        g[k - 2] = Qk(0, 0) * g[k - 2] - Qk(0, 1) * q2[k] * std::conj(p[k - 2]);
      }

      Complex temp = Qk(0, 0) * d[k - 1] + Qk(0, 1) * g[k - 1];
      g[k - 1] = Qk(1, 0) * d[k - 1] + Qk(1, 1) * g[k - 1];
      d[k - 1] = temp;

      temp = Qk(0, 0) * b[k - 1] + Qk(0, 1) * d[k];
      d[k] = Qk(1, 0) * b[k - 1] + Qk(1, 1) * d[k];
      b[k - 1] = temp;

      temp = Qk(0, 0) * p[k - 1] + Qk(0, 1) * p[k];
      p[k] = Qk(1, 0) * p[k - 1] + Qk(1, 1) * p[k];
      p[k - 1] = temp;

      if (std::norm(p[k - 1] * std::conj(q[k])) +
              std::norm(p[k] * std::conj(q[k])) >
          std::norm(b[k - 1]) + std::norm(d[k])) {
        p[k - 1] = -b[k - 1] / std::conj(q[k]);
      }

      temp = Qk(0, 0) * q2[k - 1] + Qk(0, 1) * q2[k];
      q2[k] = Qk(1, 0) * q2[k - 1] + Qk(1, 1) * q2[k];
      q2[k - 1] = temp;
    }

    return;
  }

  // ================= Algorithm 4: Shifted QR iteration =================
  static VectorC QR(VectorC d, VectorC b, VectorC p, VectorC q,
                    Scalar eps = 1e-15) {
    // pre-allocate as much as possible
    Eigen::Matrix<Complex, 2, 2> A;
    int n = p.size();
    std::vector<Matrix2C> rotations;
    rotations.resize(n - 1);
    VectorC g = b.conjugate();

    for (int i = 0; i < n - 1; ++i) {
      Complex mu_sum(0, 0);

      while (std::abs(b[i] + p[i] * std::conj(q[i + 1])) > eps) {
        // Take a 2x2 section of our matrix and use its eigenvalues as our shift
        // amount
        A << d[i] + p[i] * std::conj(q[i]), b[i] + p[i] * std::conj(q[i + 1]),
            std::conj(b[i]) + p[i + 1] * std::conj(q[i]),
            d[i + 1] + p[i + 1] * std::conj(q[i + 1]);

        Complex tr = A(0, 0) + A(1, 1);
        Complex det = A(0, 0) * A(1, 1) - A(1, 0) * A(0, 1);

        Complex sq = std::sqrt(tr * tr - Scalar(4) * det);
        Complex mu1 = Scalar(0.5) * (tr + sq);
        Complex mu2 = Scalar(0.5) * (tr - sq);

        // pick the one that A(0,0) is closest to
        Complex mu =
            (std::abs(A(0, 0) - mu1) < std::abs(A(0, 0) - mu2)) ? mu1 : mu2;

        mu_sum += mu;

        d.tail(n - i).array() -= mu;

        // Perform a QR step

        elimination(d.tail(n - i), b.tail(n - i - 1), p.tail(n - i),
                    q.tail(n - i), g.tail(n - i - 1), rotations);

        rotate(d.tail(n - i), b.tail(n - i - 1), p.tail(n - i), q.tail(n - i),
               g.tail(n - i - 1), rotations);
      }

      d.tail(n - i).array() += mu_sum;
    }
    return d + p.cwiseProduct(q.conjugate());
  }
  // A 2x2 Rayleigh-Ritz step
  static Vector eigen_est2(const Matrix &T, Vector &v1) {
    v1.normalize();
    Vector v2 = T * v1;
    Vector eigenvec = v1 * 0;

    // If residual is small enough, skip refining as it harms accuracy
    Scalar eigenval = v1.dot(v2);
    Scalar residual = (v2 - eigenval * v1).squaredNorm();
    if (residual < v1.size() * 1e-25) {
      return v2 / eigenval;
    }

    v2.normalize();
    v1 -= v1.dot(v2) * v2;
    v1.normalize();

    // Form matrix Q from v1 and v2, then R = Q.T @ T @ Q
    // Then we find the top eigenvalue/vector of R with a stable quadratic
    // formula
    Scalar a = v1.dot(T * v1);
    Scalar b = v1.dot(T * v2);
    Scalar c = v2.dot(T * v2);

    Scalar delta = (c - a) / 2.0;
    int sign = (delta > 0) ? 1 : -1;
    // Get largest magnitude eigenvalue
    eigenval = (a + c) / 2.0 + sign * sqrt(delta * delta + b * b);

    // Get eigenvector of T
    if (abs(b) > abs(eigenval - a)) {
      eigenvec = (eigenval - c) * v1 + b * v2;
    } else {
      eigenvec = b * v1 + (eigenval - a) * v2;
    }
    return eigenvec.normalized();
  }

  // Uses the method from "Real Polynomial Chebyshev Approximation by the
  // Carathéodory-Fejér Method" By Gutknecht and Trefethen (1982) The algorithm
  // in the above paper is simpler than their 1983 one which covers rational
  // approximation
  static Chebyshev RCF_truncate(Chebyshev c, unsigned int n,
                                bool no_edit = false) {
    // Input is a Chebyshev series that needs to be truncated to degree n using
    // the Carathéodory-Fejér Method Result is polynomial of degree n which is
    // very, very near the best approximant to the original degree N one.

    // Basic idea is to form a Hankel matrix from the Chebyshev coefficients
    // between orders n and N Then the largest eigenvalue estimates the
    // approximation error and the eigenvector describes a rational function for
    // the error, called b(x) We use the knowledge of that error to correct the
    // Chebyshev series of order n

    // Some notes on optimisation
    // The 1983 paper suggests multiple places where FFTs can be applied
    // They refer to "FAST FOURIER METHODS IN COMPUTATIONAL COMPLEX ANALYSIS" by
    // Peter Henrici (1979)

    // The laurent series for the Blaskche product b(x) can be calculated via
    // FFT, since it's the ratio of two polynomials in the monomial basis, and
    // multiplication/division is pointwise in fourier space. This isn't
    // actually faster unless N-n is huge, plus the direct method has an
    // early-exit condition

    // For the rational case (not yet implemented), forming the polynomial q(x)
    // which is p(x) but with the zeros inside the unit disc factored out can be
    // done with an FFT method too, see section 3.2 of Peter's book The
    // algorithm there seems to give p(x)/q(x) and I haven't figured out a way
    // to get q(x) directly

    // Something they don't mention is that mutliplication by a Hankel matrix
    // can be done with an FFT too This would make the fastest way to get the
    // largest eigenvalue/vector a Lanczos iteration using these FFTs
    unsigned int N = c.degree;
    if (N <= n + 1) {
      return c;
    }

    Vector u = Vector::Zero(N - n);
    Scalar val;

// Crossover for dense solver vs sparse solver. I haven't checked with OTFFT
#ifdef use_spectra
    if ((N - n) < 90) {
#endif

      // We only need the top eigenvalue so it's faster to use power iteration
      // than do a full solve Convergence rate is roughly equal to Chebyshev
      // series convergence rate at T_n

      // We have a guarantee that u_0 != 0 in the eigenvector from "Remarks on
      // an Algebraic Problem" by Tagaki (1925) As such, the vector u = [1, 0,
      // ..., 0] is a safer and often much better guess than random. H * u is
      // then just the first column of H

      Vector h = c.coeffs(seq(n + 1, last));
      u = h;

      // Build Hankel matrix
      Matrix H = Matrix::Zero(N - n, N - n);

      for (int i = 0; i < N - n; i++) {
        H(i, seq(0, last - i)) = c.coeffs(seq(i + n + 1, last));
      }

      // Do a single step of Rayleigh-Ritz, roughly halves the iterations needed
      // by eliminating second highest eigenvector's component
      u = eigen_est2(H, u);

      int M = h.size();
      Vector y = Vector::Zero(M);
      Scalar res = 1;
      Scalar oldres = 2;
      // Get machine epsilon
      Scalar eps = std::numeric_limits<Scalar>::epsilon();

      int iters = 0;

      // We can multiply by a Hankel matrix in a matrix-free way
      // It's as many FLOPs but seems around twice as fast due to better memory
      // locality
      while (abs(res - oldres) > 10 * eps) {
        oldres = res;
        for (int i = 0; i < M; ++i) {
          int len = M - i;
          y(i) = h.tail(len).dot(u.head(len));
        }

        for (int i = 0; i < M; ++i) {
          int len = M - i;
          u(i) = h.tail(len).dot(y.head(len));
        }

        u.normalize();
        val = y.dot(u);
        res = (u - y / val).norm();

        // Max iterations
        if (iters == (N - n)) {
          break;
        }

        iters++;
      }
      // Switch to safe method if above fails
      if (abs(res) > 1000 * eps) {
        Eigen::SelfAdjointEigenSolver<Matrix> es(H);
        auto evals = es.eigenvalues();  // Should be ascending order but we do a
                                        // full search anyway
        auto evecs = es.eigenvectors(); // columns are eigenvectors

        // We want argmax of abs(evals)
        Eigen::Index idx;
        (evals.array().abs()).maxCoeff(&idx);

        val = evals(idx);
        u = evecs.col(idx).transpose();
      }
      /*
          The eigenvalue is an extremely tight lower bound on the error between
         the n'th order CF approximation and N'th order Chebyshev series The
         terms of |b_{-k}| could be summed to turn the lower bound on error into
         an upper bound For n >> N - n they are effectively 0 as they drop off
         geometrically. I multiply by 100 for now rather than estimating 1/(1-r)
         from the geometric series
      */

#ifdef use_spectra
    } else {
      // Pass in first row of H
      HankelOp<Scalar> op(c.coeffs(seq(n + 1, last)));
      Spectra::SymEigsSolver<HankelOp<Scalar>> eigs(op, 1, 20);
      eigs.init();
      int nconv = eigs.compute(Spectra::SortRule::LargestMagn);
      auto evecs = eigs.eigenvectors();
      u = evecs.col(0).transpose();
      auto evals = eigs.eigenvalues();
      val = evals[0];
    }
#endif

    c.error += abs(val);

    // Return error without modifying c.coeffs
    if (no_edit) {
      return c;
    }

    /*
        Let v(z) be the polynomial with u[1:]/u[0] as its coefficients
        We want the Laurent series ("outside unit disc") of val * z^N *
       v(z)/v(z*) Only need coefficients b_k for orders -n to n

        We essentially use the long division algorithm to get this iterative
       formula b_k = c_k for n+1 <= k <= N Then the others are given by b_k =
       -1/u_0 ( b_{k+1} * u_1 + b_{k+2} * u_2 + ... + b_{k+M-m-1} * u_{M-m-1})
    */
    Vector b = Vector::Zero(n + N + 1);
    b(seq(2 * n + 1, last)) = c.coeffs(seq(n + 1, last));
    // Each iteration is effectively multiplication by a companion matrix
    // We know every root is below 1 from the theory so the terms must decay
    // (eventually) and we can safely exit early
    for (int k = n; k + n + 1 > 0; k--) {
      int i = k + n;

      b[i] = -1 / u[0] * b(seq(i + 1, i + N - n - 1)).dot(u(seq(1, last)));

      // Correct the Chebyshev series with laurent series
      c.coeffs[abs(k)] -= b[i];

      if (not(i & 15)) {
        // When norm is negligible, we exit
        if (b(seq(i + 1, i + N - n - 1)).norm() < c.error * 1e-16) {
          break;
        }
      }
    }
    c.error += 100 * abs(b(seq(0, N - n - 2)).dot(u(seq(1, last))) / u[0]);

    c.coeffs.conservativeResize(n + 1);
    c.degree = n;
    return c;
  }

  static Chebyshev RCF(std::function<Scalar(Scalar)> f, unsigned int n,
                       unsigned int N) {
    // inputs: f is a function to be approximated on [-1,1]
    // n is the returned polynomial degree
    // N is the polynomial degree for calculations, the error of fit(f,N+1)
    // should be negligible compared to fit(f,n+1)
    Chebyshev c = Chebyshev::fit(f, N + 1);
    return Chebyshev::RCF_truncate(c, n);
  }

  static Chebyshev RCF_bounded_truncate(Chebyshev c, Scalar tolerance) {
    // inputs: f is a function to be approximated on [-1,1]
    // tol is the maximum error that the user will allow on the approximation

    // Start with Chebyshev series that approximates f closer than the desired
    // tolerance
    int N = c.degree + 1;
    if (N <= 2)
      return c;

    // Now we need to choose the degree to truncate to
    // It's possible to bound the largest eigenvalue of the Hankel matrix that
    // appears We find the lowest degree which is guaranteed to have error too
    // large and binary search The RCF method breaks if N-n < 2

    // Lower limit is largest unacceptable degree
    // Upper limit is smallest acceptable degree
    int upper_limit = N - 2;
    int lower_limit = N - 2;
    Scalar upper_bound = 1e99;
    Scalar lower_bound = 0;
    Scalar error = c.error;

    while (lower_bound < tolerance) {
      lower_limit--;
      if (lower_limit < 0) {
        return c;
      }
      unsigned int n = N - lower_limit - 1;
      // For a lower bound, take a normalised test vector x and find x.dot(H*x)
      // where H is the Hankel matrix Taking x to be the first row of the Hankel
      // matrix works very well
      Vector h = c.coeffs.tail(n);
      lower_bound = h.cwiseAbs().maxCoeff();

      // For upper bound, use matrix 1-norm
      upper_bound = h.cwiseAbs().sum();
      if (upper_bound < tolerance) {
        upper_limit = lower_limit;
      }
    }
    while (upper_limit - lower_limit > 1) {
      unsigned int mid = (upper_limit + lower_limit) / 2;
      c = Chebyshev::RCF_truncate(c, mid, true);
      if (c.error > tolerance) {
        lower_limit = mid;
      } else {
        upper_limit = mid;
      }
      c.error = error;
    }
    return Chebyshev::RCF_truncate(c, upper_limit);
  }

  static Chebyshev RCF_bounded(std::function<Scalar(Scalar)> f,
                               Scalar tolerance, unsigned int initial_n = 16) {
    // inputs: f is a function to be approximated on [-1,1]
    // n is the returned polynomial degree
    // N is the polynomial degree for calculations, the error of fit(f,N+1)
    // should be negligible compared to fit(f,n+1)
    Chebyshev c =
        Chebyshev::fit_bounded(f, tolerance / Scalar(100), initial_n, 2);
    return Chebyshev::RCF_bounded_truncate(c, tolerance);
  }

  static Chebyshev RCF_odd_even(std::function<Scalar(Scalar)> f,
                                Scalar tolerance, unsigned int initial_n = 16) {
    // Take the input function as the sum of an odd function and an even
    // function, and find the best approximant for each simultaneously
    Chebyshev c =
        Chebyshev::fit_bounded(f, tolerance / Scalar(200), initial_n, 4);

    // If the expansion is very short, this code breaks so we just exit early.
    if (c.degree <= 2)
      return c;

    Chebyshev odd = Chebyshev(c.coeffs(seq(1, last, 2)));
    Chebyshev even = Chebyshev(c.coeffs(seq(0, last, 2)));
    odd.error = even.error = c.error;

    odd.trunc_to_error(tolerance / Scalar(100), 2);
    even.trunc_to_error(tolerance / Scalar(100), 2);

    odd = Chebyshev::RCF_bounded_truncate(odd, tolerance);
    even = Chebyshev::RCF_bounded_truncate(even, tolerance);

    c.coeffs(seq(1, 2 * odd.degree + 1, 2)) = odd.coeffs;
    c.coeffs(seq(0, 2 * even.degree, 2)) = even.coeffs;
    c.error = std::max(even.error, odd.error);
    c.degree = std::max(2 * even.degree, 2 * odd.degree + 1);
    c.coeffs.conservativeResize(c.degree + 1);
    return c;
  }

  // Use Miller's backwards algorithm to produce the Bessel functions
  static Chebyshev BesselJ(Scalar scale) {
    int n = 1.3591409142295225 * scale +
            50; // e/2 terms plus a small constant is always enough
    Vector coeffs = Vector::Zero(n);
    coeffs(n - 2) = 1;

    // We can overflow so keep track of most recent two
    Scalar max_value = 1;
    int overflow_1 = -1;
    int overflow_2 = -1;

    Scalar limit = pow(2, 100);

    for (int i = n - 2; i > 0; i--) {
      coeffs(i - 1) = 2 * i * coeffs(i) / scale - coeffs(i + 1);
      if (abs(coeffs(i - 1)) > max_value)
        max_value = abs(coeffs(i - 1));
      if (max_value > limit) {
        overflow_2 = overflow_1;
        overflow_1 = i;
        coeffs(i - 1) /= limit;
        coeffs(i) /= limit;
        max_value /= limit;
      }
    }
    if (overflow_2 > 0)
      coeffs.conservativeResize(overflow_2 + 1);
    if (overflow_1 > 0)
      coeffs(Eigen::seq(overflow_1+1, last, 1)) /= limit;

    coeffs(0) /= 2;

    Scalar norm = 2*coeffs(Eigen::seq(0, last, 2)).sum();
    Chebyshev c = Chebyshev(coeffs/norm);
    return c;
  }

  // Use the Bessel functions to return the Chebyshev expansion of sin(scale*x)
  // + cos(scale*x)
  static Chebyshev sincos(Scalar scale, Scalar tolerance) {
    Chebyshev c = BesselJ(scale);
    c.coeffs *= 2;
    c.coeffs(Eigen::seq(2, last, 4)) *= -1;
    c.coeffs(Eigen::seq(3, last, 4)) *= -1;
    c.trunc_to_error(tolerance);
    return c;
  }
};
