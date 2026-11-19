#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

void compute_heavy_openmp(const float* X, const float* Y, float* Z, float alpha, size_t N, int K) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        float x = X[i];
        float y = Y[i];
        float sum = 0.0f;

        for (int k = 1; k <= K; ++k) {
            sum += (std::sin(k * x) + std::cos(k * y)) / (float)k;
        }

        Z[i] = sum + alpha * x;
    }
}

int main(int argc, char* argv[]) {
    size_t N = (argc > 1) ? std::stoull(argv[1]) : 10000000;
    int threads = (argc > 2) ? std::stoi(argv[2]) : 8;
    int K = (argc > 3) ? std::stoi(argv[3]) : 50;
    float alpha = 2.5f;

    omp_set_num_threads(threads);
    std::vector<float> X(N, 1.25f), Y(N, 0.75f), Z(N, 0.0f);

    double start = omp_get_wtime();
    compute_heavy_openmp(X.data(), Y.data(), Z.data(), alpha, N, K);
    double end = omp_get_wtime();

    double duration_ms = (end - start) * 1000.0;
    std::cout << "OpenMP_Heavy,N=" << N << ",Threads=" << threads 
              << ",K=" << K << ",Time_ms=" << duration_ms << std::endl;
    return 0;
}
