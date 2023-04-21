#pragma once

#include <cassert>
#include <initializer_list>
#include <iostream>
#include <utility>
#include <vector>

template <class T>
class MatrixCPU
{
   private:
    std::size_t m, n;
    std::vector<T> data;

   public:
    MatrixCPU(const std::size_t row, const std::size_t col)
        : m(row), n(col), data(row * col)
    {
    }

    MatrixCPU(void) : MatrixCPU(0, 0) {}

    MatrixCPU(const std::size_t row, const std::size_t col,
              std::initializer_list<T> init)
        : m(row), n(col), data(init)
    {
        assert(init.size() == size());
    }

    void print(void) const
    {
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

    auto size(void) const noexcept { return data.size(); }

    void swap(MatrixCPU &src) noexcept(noexcept(src.data.swap(data)))
    {
        std::swap(src.m, m);
        std::swap(src.n, n);
        src.data.swap(data);
    }

    auto raw(void) { return data.data(); }

    auto raw(void) const { return data.data(); }

    auto begin(void) { return std::begin(data); }

    auto end(void) { return std::end(data); }

    void pull(void) const noexcept {}

    void push(void) const noexcept {}
};
