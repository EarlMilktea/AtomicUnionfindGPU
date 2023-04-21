#include <chrono>
#include <cstring>
#include <iostream>
#include <random>

#include "cluster2d_cpu.hpp"
#include "umatrix_cpu.hpp"

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "./bench_main_cpu [n]" << std::endl;
        return 1;
    }
    const auto n = std::atoll(argv[1]);
    auto map = MatrixCPU<bool>(n, n);
    auto rng = std::mt19937_64(0);
    auto dist = std::uniform_real_distribution<>();
    for (auto&& mij : map)
    {
        mij = dist(rng) < 0.5;
    }
    auto calc = UF2dCPU(map);
    {
        const auto t0 = std::chrono::system_clock::now();
        calc.run();
        const auto dt = std::chrono::system_clock::now() - t0;
        std::cout
            << std::chrono::duration_cast<std::chrono::microseconds>(dt).count()
            << " usec" << std::endl;
    }
    return 0;
}
