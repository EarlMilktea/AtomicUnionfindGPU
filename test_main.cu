#include <gtest/gtest.h>

#include <random>

#include "cluster2d_cpu.hpp"
#include "cluster2d_debug.hpp"
#include "cluster2d_gpu.cuh"
#include "umatrix_cpu.hpp"
#include "umatrix_gpu.cuh"

class TwoParam
    : public testing::TestWithParam<std::tuple<std::size_t, std::size_t>>
{
};

class ThreeParam : public testing::TestWithParam<
                       std::tuple<std::size_t, double, std::size_t>>
{
};

static void randomimg_test(const std::size_t n, const double p,
                           const std::size_t bsize)
{
    auto rng = std::mt19937_64(0);
    auto dist = std::uniform_real_distribution<>();
    auto map_gpu = MatrixGPU<bool>(n, n);
    auto map = MatrixCPU<bool>(n, n);
    auto po = &map_gpu[0];
    for (auto&& mapij : map)
    {
        mapij = dist(rng) < p;
        *(po++) = mapij;
    }
    map_gpu.push();
    auto ins1 = UF2dGPU(map_gpu);
    auto ins2 = UF2dDebug(map);
    auto ins3 = UF2dCPU(map);
    ins1.run(bsize);
    ins2.run();
    ins3.run();
    for (std::size_t ii = 0; ii < n * n; ++ii)
    {
        EXPECT_EQ(ins1.root(ii), ins2.root(ii));
        EXPECT_EQ(ins2.root(ii), ins3.root(ii));
    }
}
static void patchwork_test(const std::size_t n, const bool staggered,
                           const bool transposed, const std::size_t bsize)
{
    auto map_gpu = MatrixGPU<bool>(n, n);
    auto map = MatrixCPU<bool>(n, n);
    for (std::size_t i = 0; i < n; ++i)
    {
        bool c = staggered ? i % 2 == 0 : false;
        for (std::size_t j = 0; j < n; ++j)
        {
            if (transposed)
            {
                map(j, i) = c;
                map_gpu[map.index(j, i)] = c;
            }
            else
            {
                map(i, j) = c;
                map_gpu[map.index(i, j)] = c;
            }
            c = !c;
        }
    }
    map_gpu.push();
    auto ins1 = UF2dGPU(map_gpu);
    auto ins2 = UF2dDebug(map);
    auto ins3 = UF2dCPU(map);
    ins1.run(bsize);
    ins2.run();
    ins3.run();
    for (std::size_t ii = 0; ii < n * n; ++ii)
    {
        EXPECT_EQ(ins1.root(ii), ins2.root(ii));
        EXPECT_EQ(ins2.root(ii), ins3.root(ii));
    }
}

constexpr std::initializer_list<std::size_t> SIZES = {32, 64, 96, 128, 256};
constexpr std::initializer_list<std::size_t> BSIZES = {1, 2, 4, 8, 16, 32};
constexpr auto PS = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};

TEST_P(TwoParam, PatchWork)
{
    auto&& [n, bs] = GetParam();
    patchwork_test(n, true, false, bs);
}

TEST_P(TwoParam, PatchWorkTransposed)
{
    auto&& [n, bs] = GetParam();
    patchwork_test(n, true, true, bs);
}

TEST_P(TwoParam, HorizontalLine)
{
    auto&& [n, bs] = GetParam();
    patchwork_test(n, false, false, bs);
}

TEST_P(TwoParam, VerticalLine)
{
    auto&& [n, bs] = GetParam();
    patchwork_test(n, false, true, bs);
}

TEST_P(ThreeParam, RandomImg)
{
    auto&& [n, p, bs] = GetParam();
    randomimg_test(n, p, bs);
}

INSTANTIATE_TEST_SUITE_P(CartesianProd2, TwoParam,
                         testing::Combine(testing::ValuesIn(SIZES),
                                          testing::ValuesIn(BSIZES)));

INSTANTIATE_TEST_SUITE_P(CartesianProd3, ThreeParam,
                         testing::Combine(testing::ValuesIn(SIZES),
                                          testing::ValuesIn(PS),
                                          testing::ValuesIn(BSIZES)));
