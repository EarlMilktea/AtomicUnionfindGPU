#pragma once

#include <cassert>
#include <initializer_list>
#include <iostream>
#include <utility>

#include "errorcheck.cuh"

template <class T>
class MatrixGPU
{
   private:
    std::size_t m, n;
    T *data;

    auto memsize(void) const noexcept { return sizeof(T) * size(); }

    void allocate_uninit(void)
    {
        cudaCheck(
            cudaMallocManaged(reinterpret_cast<void **>(&data), memsize()));
    }

   public:
    MatrixGPU(const std::size_t row, const std::size_t col) : m(row), n(col)
    {
        allocate_uninit();
        for (auto ii = 0U; ii < size(); ++ii)
        {
            data[ii] = T();
        }
    }

    MatrixGPU(void) : MatrixGPU(0, 0) {}

    MatrixGPU(const std::size_t row, const std::size_t col,
              std::initializer_list<T> init)
        : m(row), n(col)
    {
        assert(init.size() == size());
        auto p = data;
        for (auto &&initj : init)
        {
            *(p++) = initj;
        }
    }

    MatrixGPU(const MatrixGPU &src) : m(src.m), n(src.n)
    {
        allocate_uninit();
        cudaCheck(cudaMemcpy(static_cast<void *>(data),
                             static_cast<void *>(src.data), memsize(),
                             cudaMemcpyDefault));
    }

    MatrixGPU &operator=(const MatrixGPU &rhs) &
    {
        m = rhs.m;
        n = rhs.n;
        allocate_uninit();
        cudaCheck(cudaMemcpy(static_cast<void *>(data),
                             static_cast<void>(rhs.data), memsize(),
                             cudaMemcpyDefault));
        return *this;
    }

    MatrixGPU(MatrixGPU &&rhs) noexcept : m(rhs.m), n(rhs.n), data(rhs.data)
    {
        rhs.data = nullptr;
    }

    MatrixGPU &operator=(MatrixGPU &&rhs) &noexcept
    {
        m = rhs.m;
        n = rhs.n;
        data = rhs.data;
        rhs.data = nullptr;
        return *this;
    }

    ~MatrixGPU() noexcept
    {
        if (data != nullptr)
        {
            cudaCheck(cudaFree(data));
        }
    }

    void print(void) const
    {
        pull();
        for (std::size_t i = 0; i < m; ++i)
        {
            for (std::size_t j = 0; j < n; ++j)
            {
                std::cout << data[index(i, j)] << (j + 1 == n ? "\n" : " ");
            }
        }
    }

    auto index(const std::size_t i, const std::size_t j) const noexcept
    {
        assert(i < rows());
        assert(j < cols());
        return i * n + j;
    }

    decltype(auto) operator()(const std::size_t i, const std::size_t j) const
    {
        assert(i < rows());
        assert(j < cols());
        assert(index(i, j) < size());
        return data[index(i, j)];
    }

    decltype(auto) operator()(const std::size_t i, const std::size_t j)
    {
        assert(i < rows());
        assert(j < cols());
        assert(index(i, j) < size());
        return data[index(i, j)];
    }

    decltype(auto) operator[](const std::size_t ii) const
    {
        assert(ii < size());
        return data[ii];
    }

    decltype(auto) operator[](const std::size_t ii)
    {
        assert(ii < size());
        return data[ii];
    }

    auto rows(void) const noexcept { return m; }

    auto cols(void) const noexcept { return n; }

    auto size(void) const noexcept { return m * n; }

    void swap(MatrixGPU &src) noexcept
    {
        std::swap(src.m, m);
        std::swap(src.n, n);
        std::swap(src.data, data);
    }

    auto raw(void) { return data; }

    auto raw(void) const { return data; }

    void pull(void) const
    {
        cudaCheck(cudaMemPrefetchAsync(static_cast<void *>(data), memsize(),
                                       cudaCpuDeviceId));
    }

    void push(void) const
    {
        int dev;
        cudaCheck(cudaGetDevice(&dev));
        cudaCheck(
            cudaMemPrefetchAsync(static_cast<void *>(data), memsize(), dev));
    }
};
