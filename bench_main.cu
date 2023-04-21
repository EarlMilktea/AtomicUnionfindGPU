#include <chrono>
#include <cstring>
#include <iostream>
#include <random>

#include "cluster2d_gpu.cuh"
#include "umatrix_gpu.cuh"

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "./bench_main [n] [block size] [threads]" << std::endl;
        return 1;
    }
    const auto n = std::atoll(argv[1]);
    const auto bsize = std::atoll(argv[2]);
    const auto nthread = std::atoll(argv[3]);
    if (n % bsize != 0)
    {
        std::cerr << "n % bsize != 0" << std::endl;
        return 2;
    }
    auto map_gpu = MatrixGPU<bool>(n, n);
    auto rng = std::mt19937_64(0);
    auto dist = std::uniform_real_distribution<>();
    for (std::size_t ii = 0; ii < n * n; ++ii)
    {
        map_gpu[ii] = dist(rng) < 0.5;
    }
    map_gpu.push();
    auto calc = UF2dGPU(map_gpu);
    {
        const auto t0 = std::chrono::system_clock::now();
        calc.run(bsize, nthread);
        const auto dt = std::chrono::system_clock::now() - t0;
        std::cout
            << std::chrono::duration_cast<std::chrono::microseconds>(dt).count()
            << " usec" << std::endl;
    }
    return 0;
}
