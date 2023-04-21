#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <iostream>
#include <utility>
#include <vector>

#include "umatrix_cpu.hpp"

class UF2dCPU
{
   public:
    auto root(const std::size_t ind,
              const std::memory_order ord = std::memory_order_seq_cst) const
    {
        auto pos = ind;
        while (true)
        {
            const auto pos_new = parent[pos].load(ord);
            if (pos_new == pos)
            {
                break;
            }
            else
            {
                pos = pos_new;
            }
        }
        return pos;
    }

   private:
    MatrixCPU<bool> c;
    MatrixCPU<std::atomic<std::size_t>> parent;

    void rowwise_merge(const std::size_t i, const std::size_t j)
    {
        if (j != 0 && c(i, j) == c(i, j - 1))
        {
            const auto p = parent(i, j - 1).load(std::memory_order_relaxed);
            parent(i, j).store(p, std::memory_order_relaxed);
        }
    }

    void colwise_merge(const std::size_t i, const std::size_t j)
    {
        if (i != 0 && c(i, j) == c(i - 1, j))
        {
            const auto p = parent(i - 1, j).load(std::memory_order_relaxed);
            parent(i, j).store(p, std::memory_order_relaxed);
        }
    }

    void pathcomp(const std::size_t ind)
    {
        parent[ind].store(root(ind, std::memory_order_relaxed),
                          std::memory_order_relaxed);
    }

    void merge(std::size_t ind1, std::size_t ind2)
    {
        // Needs seq_cst ordering?
        // -> maybe NO
        while (true)
        {
            ind1 = root(ind1, std::memory_order_relaxed);
            ind2 = root(ind2, std::memory_order_relaxed);
            if (ind1 == ind2)
            {
                return;
            }
            else if (ind1 > ind2)
            {
                std::swap(ind1, ind2);
            }
            auto expected = parent[ind2].load(std::memory_order_relaxed);
            const auto desired = std::min({expected, ind1});
            parent[ind2].compare_exchange_weak(expected, desired,
                                               std::memory_order_relaxed);
        }
    }

   public:
    UF2dCPU(const MatrixCPU<bool>& map) : c(map), parent(map.rows(), map.cols())
    {
        assert(c.size() != 0);
        // row-major index
        std::size_t id = 0;
        for (auto&& pij : parent)
        {
            pij = id++;
        }
    }

    void run(void)
    {
#pragma omp parallel
#pragma omp for collapse(2)
        for (std::size_t i = 0; i < c.rows(); ++i)
        {
            for (std::size_t j = 0; j < c.cols(); ++j)
            {
                rowwise_merge(i, j);
            }
        }
#pragma omp for collapse(2)
        for (std::size_t i = 0; i < c.rows(); ++i)
        {
            for (std::size_t j = 0; j < c.cols(); ++j)
            {
                colwise_merge(i, j);
            }
        }
#pragma omp for
        for (std::size_t ii = 0; ii < c.size(); ++ii)
        {
            pathcomp(ii);
        }
#pragma omp for collapse(2) nowait
        for (std::size_t i = 0; i < c.rows(); ++i)
        {
            for (std::size_t j = 1; j < c.cols(); ++j)
            {
                if (c(i, j) == c(i, j - 1))
                {
                    merge(c.index(i, j), c.index(i, j - 1));
                }
            }
        }
#pragma omp for collapse(2)
        for (std::size_t i = 1; i < c.rows(); ++i)
        {
            for (std::size_t j = 0; j < c.cols(); ++j)
            {
                if (c(i, j) == c(i - 1, j))
                {
                    merge(c.index(i, j), c.index(i - 1, j));
                }
            }
        }
#pragma omp for
        for (std::size_t ii = 0; ii < c.size(); ++ii)
        {
            pathcomp(ii);
        }
    }

    void debug(void) const
    {
        c.print();
        std::cout << std::endl;
        for (std::size_t i = 0; i < parent.rows(); ++i)
        {
            for (std::size_t j = 0; j < parent.cols(); ++j)
            {
                std::cout << root(parent.index(i, j))
                          << (j + 1 == parent.cols() ? "\n" : " ");
            }
        }
        std::cout << std::endl;
    }
};
