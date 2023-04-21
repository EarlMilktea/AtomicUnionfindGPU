#pragma once

#include <algorithm>
#include <cassert>
#include <iostream>
#include <utility>
#include <vector>

#include "umatrix_cpu.hpp"

class UF2dDebug
{
   private:
    static constexpr auto DUMMY = static_cast<std::size_t>(-1);
    MatrixCPU<bool> c;
    MatrixCPU<std::size_t> parent;

    void dfs(const std::size_t i, const std::size_t j, const std::size_t root)
    {
        auto&& pij = parent(i, j);
        const auto cij = c(i, j);
        if (pij == root)
        {
            return;
        }
        else
        {
            pij = root;
        }
        if (i != 0 && cij == c(i - 1, j))
        {
            dfs(i - 1, j, root);
        }
        if (i + 1 != c.rows() && cij == c(i + 1, j))
        {
            dfs(i + 1, j, root);
        }
        if (j != 0 && cij == c(i, j - 1))
        {
            dfs(i, j - 1, root);
        }
        if (j + 1 != c.cols() && cij == c(i, j + 1))
        {
            dfs(i, j + 1, root);
        }
    }

   public:
    UF2dDebug(const MatrixCPU<bool>& map)
        : c(map), parent(map.rows(), map.cols())
    {
        std::fill(parent.begin(), parent.end(), DUMMY);
        assert(c.size() != 0);
    }

    void run(void)
    {
        for (std::size_t i = 0; i < c.rows(); ++i)
        {
            for (std::size_t j = 0; j < c.cols(); ++j)
            {
                if (parent(i, j) == DUMMY)
                {
                    dfs(i, j, parent.index(i, j));
                }
            }
        }
    }

    auto root(std::size_t ind) const { return parent[ind]; }

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
