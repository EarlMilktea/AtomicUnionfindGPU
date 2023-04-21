#pragma once

#include <iostream>

#include "cluster2d_kernels.cuh"
#include "errorcheck.cuh"
#include "umatrix_gpu.cuh"

class UF2dGPU
{
   private:
    MatrixGPU<bool> c;
    MatrixGPU<unsigned long long> parent;

   public:
    UF2dGPU(const MatrixGPU<bool>& map) : c(map), parent(map.rows(), map.cols())
    {
        c.push();
    }

    void run(const unsigned bsize = 32U, const unsigned nthreads = 32U)
    {
        assert(nthreads > 0);
        assert(bsize > 0);
        assert(c.rows() % bsize == 0);
        assert(c.cols() % bsize == 0);
        const auto size_shared =
            bsize * (bsize + 1) * (sizeof(bool) + sizeof(unsigned));
        const auto nb_row = static_cast<unsigned>(c.rows() / bsize);
        const auto nb_col = static_cast<unsigned>(c.cols() / bsize);
        run_kernel_pre<<<dim3(nb_row, nb_col), nthreads, size_shared>>>(
            &c[0], &parent[0], bsize);
        cudaCheck(cudaPeekAtLastError());
        run_kernel_post<<<dim3(nb_row, nb_col), nthreads>>>(&c[0], &parent[0],
                                                            bsize);
        cudaCheck(cudaPeekAtLastError());
        parent.pull();
        cudaCheck(cudaDeviceSynchronize());
    }

    auto root(const unsigned long long ind, const bool pull = false) const
    {
        auto pos = ind;
        if (pull)
        {
            parent.pull();
        }
        while (true)
        {
            const auto pos_new = parent[pos];
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

    void debug(void) const
    {
        c.print();
        parent.pull();
        std::cout << std::endl;
        for (auto i = 0ULL; i < parent.rows(); ++i)
        {
            for (auto j = 0ULL; j < parent.cols(); ++j)
            {
                std::cout << root(parent.index(i, j))
                          << (j + 1 == parent.cols() ? "\n" : " ");
            }
        }
        std::cout << std::endl;
    }
};
