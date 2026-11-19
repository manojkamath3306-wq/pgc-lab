#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

void compute_sequential(const float* X, const float* Y, float* Z, float alpha, size_t N) {
    for (size_t i = 0; i < N; ++i) {
        Z[i] = std::sqrt(X[i] * X[i] + Y[i] * Y[i]) + alpha * X[i];
    }
}

int main(int argc, char* argv[]) {
    size_t N = (argc > 1) ? std::stoull(argv[1]) : 10000000;
    float alpha = 2.5f;

    std::vector<float> X(N, 1.5f), Y(N, 2.5f), Z(N, 0.0f);

    auto start = std::chrono::high_resolution_clock::now();
    compute_sequential(X.data(), Y.data(), Z.data(), alpha, N);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Sequential,N=" << N << ",Time_ms=" << duration.count() << std::endl;
    return 0;
}
