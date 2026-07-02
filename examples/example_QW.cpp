#include "../Chebyshev.hpp"
#include "vcl2/vectorclass.h"
#include <chrono>
#include <eigen3/Eigen/Core>
#include <iostream>
#include <math.h>
#include <random>
#include <string>

#define PI 3.1415926535897932384626

using namespace Eigen;

constexpr int max_cache = 15;

// Get least significant bit
unsigned int LSB(int n) { return n & (-n); }

// Convert index to grey code
unsigned int grey(unsigned int n) { return n ^ (n >> 1); }

// Get position of the only set bit
// Taken from
// http://graphics.stanford.edu/~seander/bithacks.html#IntegerLogDeBruijn
unsigned int log2(unsigned int v) {
  static const int MultiplyDeBruijnBitPosition2[32] = {
      0,  1,  28, 2,  29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4,  8,
      31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6,  11, 5,  10, 9};
  return MultiplyDeBruijnBitPosition2[(uint32_t)(v * 0x077CB531U) >> 27];
}

// This is the SplitMix64 PRNG, which I use to generate seeds for C++'s Mersene
// Twister
uint64_t seed_hash(uint64_t x) {
  x += 0x9e3779b97f4a7c15;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
  x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
  return x ^ (x >> 31);
}

// Load H_P for a given problem from file, returning E_loc and E_abs
// J and h must be pre-allocated to (n,n) and (n,) respectively
void load_problem(Eigen::ArrayXf &H_P, Eigen::ArrayXXf &J, Eigen::ArrayXf &h,
                  unsigned int &E_loc, float &E_abs) {
  using namespace Eigen;
  unsigned int n = h.size();
  unsigned int N = 1 << n;
  ArrayXXf state(n, n);

  state.setConstant(1);

  // The way we calculate E is prone to error so use a double
  // Start with the energy of the all -1s state
  double E = J.sum() - h.sum();
  H_P[0] = E;

  double E_0 = E, E_max = E;
  E_loc = 0;

  // Use a grey code to efficiently evaluate all energies
  for (unsigned int i = 1; i < N; i++) {
    unsigned int flip = log2(LSB(i));
    state.row(flip) *= -1;
    state.col(flip) *= -1;
    state(flip, flip) *= -1;
    E += 4 * (J.row(flip) * state.row(flip)).sum() -
         2 * h(flip) * state(flip, flip);
    H_P[grey(i)] = E;

    // keep track of ground state
    if (E < E_0) {
      E_0 = E;
      E_loc = grey(i);
    }

    // keep track of highest state too
    if (E > E_max) {
      E_max = E;
    }
  }

  E_abs = (E_max - E_0) / 2;
  return;
}

void Clenshaw_step(float *b1, float *b2, float *hp, float *psi,
                   const unsigned int n, float scale, float gamma, float coef) {
  // This function is a bit of a mess, originally it calculated b2 += H_G @ b1
  // by essentially using a fast walsh-hadamard transform But the first part of
  // it is compute heavy enough that we can put bandwidth limited calculations
  // next to it for free So now it does a full step of the Clenshaw algorithm,
  // setting b2 = coef*psi + 2*(H @ b1) - b2 Where H = (H_P - gamma*H_G) / scale

  // Multiplying H_G by something is slow here as it'd require a multiplication
  // on every "b += a" We rescale b2 and then scale the final result so that H_G
  // has a coefficient of 1

  // We set new_scale = -2*gamma/scale, then the calculation is
  // b2 = new_scale * (H_G @ b1 - (H_P @ b1)/gamma - b2/new_scale)
  // In the final step we add psi*coeff as it saves a second pass over b2
  // At that point in the calculation,

  const unsigned int N = (1 << n);

  Vec16f a;
  Vec16f b;
  Vec16f H;
  int h = 16;
  //(1 << max_cache) should line up with cache size in some sense
  // Needs to be tuned for each machine ideally

  constexpr int max_cache = 15;
  float new_scale = -2.0f * gamma / scale;
  float new_scale_inv = 1.0f / new_scale;

  // Use permutations for h<16 cases
  // We do enough computation here that we can load some out-of-cache data for
  // free
  for (int i = 0; i < N; i += 16) {
    a.load(b1 + i);
    b.load(b2 + i);

    b *= -new_scale_inv;
    H.load(hp + i);
    b -= a * H / gamma;

    b += permute16<1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14>(a);
    b += permute16<2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13>(a);
    b += permute16<4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11>(a);
    b += permute16<8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7>(a);

    // For n larger than this, we can't fit vectors in cache anymore so these
    // are guaranteed cache misses
    if (n > max_cache) {
      for (h = 1 << max_cache; h < N; h *= 2) {
        int temp = h & i ? -h : h;
        a.load(b1 + i + temp);
        b += a;
      }
    }

    b.store(b2 + i);
  }

  // The vector can be kept in cache for these sizes
  int end = N < (1 << max_cache) ? N : 1 << max_cache;
  for (int i = 0; i < N; i += 16) {
    b.load(b2 + i);
    for (h = 16; h < end; h *= 2) {
      int temp = h & i ? -h : h;
      a.load(b1 + i + temp);
      b += a;
    }
    b *= new_scale;
    // add psi*coef
    H.load(psi + i);
    b += H * coef;
    b.store(b2 + i);
  }
  return;
}

void Clenshaw(Eigen::Ref<VectorXf> coeffs, Eigen::Ref<ArrayXf> psi,
              Eigen::Ref<ArrayXf> H_P, float gamma, float scale) {
  // Calculate (sin(H*t) + cos(H*t))@psi using the Clenshaw algorithm
  // with polynomial coefficients stored in coeffs.

  int N = H_P.size();
  int n = log2(H_P.size());
  thread_local ArrayXf b1, b2;
  b1 = psi * coeffs(last);
  b2.setZero(N);

  for (int r = coeffs.size() - 2; r > 0; --r) {
    Clenshaw_step(b1.data(), b2.data(), H_P.data(), psi.data(), n, scale, gamma,
                  coeffs[r]);
    // Swap b1 and b2 without actually copying
    std::swap(b1, b2);
  }
  // Final iteration
  Clenshaw_step(b1.data(), b2.data(), H_P.data(), psi.data(), n, 2.0 * scale,
                gamma, coeffs[0]);

  psi = b2;
}

void H_G(float *psi, float *psi2, unsigned int n) {
  // Calculates psi2 = H_G @ psi
  unsigned int N = (1 << n);

  Vec16f a;
  Vec16f b;
  int h = 16;

  // Use permutations for h<16 cases
  for (int i = 0; i < N; i += 16) {
    a.load(psi + i);
    b = permute16<1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14>(a);
    b += permute16<2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13>(a);
    b += permute16<4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11>(a);
    b += permute16<8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7>(a);
    b.store(psi2 + i);
  }

  // The vector can be kept in cache for these sizes
  if (n > 4) {
    int end = N < 1 << max_cache ? N : 1 << max_cache;
    for (int i = 0; i < N; i += 16) {
      b.load(psi2 + i);
      for (h = 16; h < end; h *= 2) {
        int temp = h & i ? -h : h;
        a.load(psi + i + temp);
        b += a;
      }
      b.store(psi2 + i);
    }
  }

  // Cache can't save us so we have to suffer
  if (n > max_cache) {
    for (int i = 0; i < N; i += 16) {
      b.load(psi2 + i);
      for (h = 1 << max_cache; h < N; h *= 2) {
        int temp = h & i ? -h : h;
        a.load(psi + i + temp);
        b += a;
      }
      b.store(psi2 + i);
    }
  }
  return;
}

ArrayXf ClenshawNaive(VectorXf coeffs, ArrayXf psi, ArrayXf H_P, float gamma,
                      float scale) {
  // Calculate (sin(H*t) + cos(H*t))@psi using the Clenshaw algorithm,
  // unoptimised
  int n = log2(psi.size());
  ArrayXf b1 = coeffs(last) * psi;
  ArrayXf b2 = b1 * 0;
  ArrayXf temp = b1 * 0;

  for (int r = coeffs.size() - 2; r > 0; r--) {
    H_G(b1.data(), temp.data(), n);
    b2 = (2 * (H_P * b1 - gamma * temp) / scale - b2) + psi * coeffs[r];
    std::swap(b1, b2);
  }
  H_G(b1.data(), temp.data(), n);
  b2 = ((H_P * b1 - gamma * temp) / scale - b2) + psi * coeffs[0];

  return b2;
}

ArrayXf Horner(VectorXf coeffs, ArrayXf psi, ArrayXf H_P, float gamma,
               float scale) {
  // Calculate (sin(H*t) + cos(H*t))@psi using Horner's algorithm, unoptimised
  int n = log2(psi.size());
  ArrayXf b1 = coeffs(last) * psi;
  ArrayXf temp = b1 * 0;

  for (int r = coeffs.size() - 2; r >= 0; r--) {
    H_G(b1.data(), temp.data(), n);
    b1 = (H_P * b1 - gamma * temp) / scale + psi * coeffs[r];
  }

  return b1;
}

float QW(int n, int method) {
  unsigned int N = 1 << n;
  unsigned int problems = 10;

  // A degree 50 polynomial in both monomial basis and Chebyshev basis
  Eigen::VectorXf TaylorCoeffs(50);
  TaylorCoeffs << -0.05953213, 1.30002918, 0.63623466, -0.62729774, 0.83885969,
      -1.9951871, 0.14096763, -1.06925196, 0.56182912, 0.37484684, 2.36702465,
      0.89734579, 0.10351181, 0.31886346, -0.08309524, -2.48047841, -0.23413448,
      1.83602183, 0.73995595, -0.6188986, -1.72576297, -0.83175887, -1.83336287,
      -0.94887372, -0.86919628, 0.05338438, 0.5632059, 0.8218555, 0.21622404,
      -0.52640973, -0.49650506, -0.01690748, -0.71861385, 0.16710707,
      -0.34098613, 0.87088418, 1.21060144, 0.49954353, 0.14739503, 0.00456723,
      0.07093225, 0.01019736, 0.27079913, -0.36488257, 1.8774893, -1.20092647,
      -0.12418828, 0.33310797, -0.84361517, 0.2096738;

  Eigen::VectorXf ChebCoeffs(50);
  ChebCoeffs << 9.33249531e-01, -1.10819377e+00, 1.26476422e+00,
      -1.27082573e+00, 2.65114899e-01, -3.90339773e-01, -5.36267567e-02,
      -1.25448135e-01, -6.19956677e-02, -5.43555792e-02, -5.07649857e-03,
      -1.97733323e-02, 2.46202733e-02, -6.77735122e-03, 2.42214515e-02,
      -3.36536701e-03, 1.45722987e-02, -2.14943111e-03, 6.70640936e-03,
      -1.26039054e-03, 2.50578088e-03, -6.11507110e-04, 7.67484252e-04,
      -2.41923947e-04, 1.86016721e-04, -7.78298208e-05, 3.08314966e-05,
      -2.00630093e-05, 7.35794115e-07, -3.93492312e-06, -1.76690775e-06,
      -4.88440721e-07, -7.98013721e-07, 5.51437845e-09, -2.25567420e-07,
      2.16063312e-08, -4.78309219e-08, 6.49852627e-09, -7.90081881e-09,
      1.22159665e-09, -1.01271956e-09, 1.63446169e-10, -9.78149032e-11,
      1.56865431e-11, -6.71042758e-12, 1.03023673e-12, -2.91253465e-13,
      4.12343865e-14, -5.99424629e-15, 7.44910970e-16;

  Eigen::ArrayXf success_probabilities(problems);
  success_probabilities *= 0;

  // Problem energy levels
  ArrayXf H_P(N);

  // Our quantum register, purely real in this example
  ArrayXf psi(N);

  // Ising problem parameters
  ArrayXXf J(n, n);
  ArrayXf h(n);
  J.setZero();
  h.setZero();

  long int base_seed = 29552825458725;

  for (int problem = 0; problem < problems; problem++) {
    float E_abs = 0;
    unsigned int E_loc = 0;

    // Initialise PRNG in a reproducible way that avoids correlations.
    std::mt19937 gen(seed_hash(base_seed + problem));
    std::normal_distribution<float> rand_t(0, 1);

    // Generate J matrix
    for (int i = 1; i < n; i++) {
      for (int j = 0; j < i; j++) {
        J(i, j) = rand_t(gen);
      }
    }

    // Make J symmetric
    J += J.transpose().eval();
    J /= 2;

    for (int i = 0; i < n; i++) {
      h(i) = rand_t(gen);
    }

    load_problem(H_P, J, h, E_loc, E_abs);

    psi.setConstant(1 / sqrt(N));

    float gamma = PI;
    float onenorm = (E_abs + gamma * n);

    if (method == 1)
      Clenshaw(ChebCoeffs, psi, H_P, gamma, onenorm);
    if (method == 2)
      psi = ClenshawNaive(ChebCoeffs, psi, H_P, gamma, onenorm);
    if (method == 3)
      psi = Horner(TaylorCoeffs, psi, H_P, gamma, onenorm);

    // Approximation errors make this method non-unitary so we renormalise
    // psi /= psi.matrix().norm();

    success_probabilities(problem) = psi[E_loc] * psi[E_loc] + H_P[E_loc];
  }
  return success_probabilities.mean();
}

int main() {
  using clock = std::chrono::high_resolution_clock;
  for (int n = 14; n < 26; n++) {
    for (int method = 1; method <= 3; method++) {
      // no optimising away
      volatile float guard = 0;
      auto t1 = clock::now();
      float ans = QW(n, method);
      auto t2 = clock::now();
      guard += ans;
      std::string alg;

      if (method == 1)
        alg = "Clenshaw";
      if (method == 2)
        alg = "Naive Clenshaw";
      if (method == 3)
        alg = "Naive Horner";
      std::cout << n << " " << alg << " "
                << std::chrono::duration<double>(t2 - t1).count() << " " << ans
                << "\n";
    }
    std::cout << "\n";
  }
  return 0;
}
