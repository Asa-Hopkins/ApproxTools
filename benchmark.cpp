#include "Chebyshev.hpp"
#include "remez.cpp"
#include <iostream>
#include <chrono>
#include <iomanip>

using Scalar = double;

int main() {
  using Poly = Chebyshev<Scalar>;
  using clock = std::chrono::high_resolution_clock;
  
  const Scalar pi = std::acos(Scalar(-1));
  int n = 10;
  double scale = 1.0;
  int remez_limit = 8;
  double tolerance = 1e-9;
  int runs = 100;
  for (int j = 0; j < 16; j++){
    int N = 4 + 4*j;
    n = (scale*12)/10 + 256;

    auto f = [scale](Scalar x) {
        return sin(scale*x) + cos(scale*x);
    };

    //This weight function gives us the infinity norm
    auto w = [](Scalar x){
      return 1.0;
    };

    Poly p3 = Poly::fit(f, n+1);
    Poly p0 = Poly::fit_bounded(f, tolerance);
    Poly r0 = Poly::sincos(scale, 1e-12);
    Poly r1 = Poly::RCF_bounded_truncate(r0, tolerance);

    int max_n = p0.degree;

    auto benchmark_fit = [&](int i) {
      //no optimising away
      volatile Scalar guard = 0;
      auto t1 = clock::now();
      Poly p = Poly::fit(f, max_n+1);
      auto t2 = clock::now();
      guard += p(i);
      return std::chrono::duration<double>(t2 - t1).count();
    };

    auto benchmark_RCF = [&](int i) {
      volatile Scalar guard = 0;
      auto t1 = clock::now();
      Poly p = Poly::RCF(f, max_n, N+max_n);
      auto t2 = clock::now();
      guard += p(i);
      return std::chrono::duration<double>(t2 - t1).count();
    };

    auto benchmark_fit2 = [&](int i) {
      //no optimising away
      volatile Scalar guard = 0;
      auto t1 = clock::now();
      Poly p = Poly::fit_bounded(f, tolerance);

      auto t2 = clock::now();
      guard += p(i);
      return std::chrono::duration<double>(t2 - t1).count();
    };

    auto benchmark_RCF2 = [&](int i) {
      volatile Scalar guard = 0;
      auto t1 = clock::now();

      Poly p = Poly::RCF_bounded_truncate(r0, tolerance);

      auto t2 = clock::now();
      guard += p(i);
      return std::chrono::duration<double>(t2 - t1).count();
    };

    auto benchmark_RCF3 = [&](int i) {
      volatile Scalar guard = 0;
      auto t1 = clock::now();

      Poly p = Poly::RCF_odd_even(f, tolerance);

      auto t2 = clock::now();
      guard += p(i);
      return std::chrono::duration<double>(t2 - t1).count();
    };

    auto benchmark_Remez = [&](int i) {
      volatile Scalar guard = 0;
      auto t1 = clock::now();
      Poly p = Remez<Scalar>::SimpleRemez(f, max_n, N+max_n,w);
      auto t2 = clock::now();
      guard += p(i);
      return std::chrono::duration<double>(t2 - t1).count();
    };

    auto benchmark_Miller = [&](int i) {
      //no optimising away
      volatile Scalar guard = 0;
      auto t1 = clock::now();
      Poly p = Poly::BesselJ(scale);
      auto t2 = clock::now();
      guard += p(i);
      return std::chrono::duration<double>(t2 - t1).count();
    };

    // warm-up
    volatile Scalar sink = 0;
    sink += Poly::fit(f, max_n)(0.1);

    sink += Poly::RCF(f, max_n, max_n + N)(0.1);

    double total = 0;
    for (int i = 0; i < runs; ++i) {total += benchmark_fit(i);}
    std::cout << "avg fit() time: " << total / runs << " seconds\n";
    
/*
    total = 0;
    for (int i = 0; i < runs; ++i) {total += benchmark_RCF(i);}
    std::cout << "avg RCF() time: " << total / runs << " seconds\n";
*/

    total = 0;
    for (int i = 0; i < runs; ++i) {total += benchmark_fit2(i);}
    std::cout << "avg fit_bounded() time: " << total / runs << " seconds\n";

    total = 0;
    for (int i = 0; i < runs; ++i) {total += benchmark_RCF2(i);}
    std::cout << "avg RCF_bounded() time: " << total / runs << " seconds\n"; 

    /*
    total = 0;
    for (int i = 0; i < runs; ++i) {total += benchmark_RCF3(i);}
    std::cout << "avg RCF_odd_even() time: " << total / runs << " seconds\n"; */

    total = 0;
    for (int i = 0; i < runs; ++i) {total += benchmark_Miller(i);}
    std::cout << "avg Miller time: " << total / runs << " seconds\n";

    if (j < remez_limit){
      total = 0;
      for (int i = 0; i < runs; ++i) {total += benchmark_Remez(i);}
      std::cout << "avg Remez() time: " << total / runs << " seconds\n";
    }

    Poly p1 = Poly::fit(f, max_n+1);
    Poly p2 = Poly::RCF(f, max_n, max_n+N);
    std::cout << std::setprecision (10) << "\n";

    std::cout << "x-axis scaled by:" << scale << "\n";
    std::cout << "Estimated error of Chebyshev: " << (double)p1.error << " with degree " << p1.degree << "\n";
    std::cout << "Estimated error of RCF: " << (double)p2.error << " with degree " << p2.degree << "\n";
    std::cout << "Estimated error of analytic Chebyshev:" << (double)r0.error << " with degree " << r0.degree << "\n";
    std::cout << "Estimated error of bounded RCF: " << (double)r1.error << " with degree " << r1.degree << "\n";

    if (j < remez_limit){
      Poly p4 = Remez<Scalar>::SimpleRemez(f, max_n, max_n + N, w);
      std::cout << "Max error of Remez: " << (double)p4.error << " with degree " << p4.degree << "\n";
    }

    std::cout << "RCF is " << (double)p1.error/(double)p2.error << "x more accurate\n";
    std::cout << "Measured error of analytic Chebyshev is " << (double)f(-1) - (double)r0(-1) << " " << (double)f(1) - (double)r0(1) << "\n\n\n";
    std::cout << "Measured error of bounded RCF is " << (double)f(-1) - (double)r1(-1) << " " << (double)f(1) - (double)r1(1) << "\n\n\n";

    scale *= 2;
    }


  return 0;
}
