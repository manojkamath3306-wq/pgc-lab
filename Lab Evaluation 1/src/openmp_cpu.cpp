#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

void compute_openmp(const float* X, const float* Y, float* Z, float alpha, size_t N) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        Z[i] = std::sqrt(X[i] * X[i] + Y[i] * Y[i]) + alpha * X[i];
    }
}

int main(int argc, char* argv[]) {
    size_t N = (argc > 1) ? std::stoull(argv[1]) : 10000000;
    int threads = (argc > 2) ? std::stoi(argv[2]) : 4;
    float alpha = 2.5f;

    omp_set_num_threads(threads);
    std::vector<float> X(N, 1.5f), Y(N, 2.5f), Z(N, 0.0f);

    double start = omp_get_wtime();
    compute_openmp(X.data(), Y.data(), Z.data(), alpha, N);
    double end = omp_get_wtime();

    double duration_ms = (end - start) * 1000.0;
    std::cout << "OpenMP,N=" << N << ",Threads=" << threads 
              << ",Time_ms=" << duration_ms << std::endl;
    return 0;
}
