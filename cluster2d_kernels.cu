#include <cassert>
#include <cstdio>

template <class I>
__device__ auto root(I* parent, const I iio, const I padding)
{
    const auto origin = iio;
    auto now = origin;
    auto next = parent[origin];
    while (true)
    {
        if (now == next)
        {
            break;
        }
        parent[origin] = next;
        now = next;
        next = parent[now];
    }
    return next;
}

template <class I>
__device__ void merge_kernel_block(I* parent, I iio, I jjo, const I padding)
{
    while (true)
    {
        iio = root(parent, iio, padding);
        jjo = root(parent, jjo, padding);
        if (iio == jjo)
        {
            // This CAN fail!
            // assert(root(parent, iio, padding) ==
            //        root(parent, jjo, padding));
            break;
        }
        else if (iio > jjo)
        {
            const auto tmp = jjo;
            jjo = iio;
            iio = tmp;
        }
        assert(iio < jjo);
        // MUST BE atomicCAS, not atomicMin!
        atomicCAS_block(parent + jjo, jjo, iio);
    }
}

template <class I>
__device__ void merge_kernel(I* parent, I iio, I jjo, const I padding)
{
    while (true)
    {
        iio = root(parent, iio, padding);
        jjo = root(parent, jjo, padding);
        if (iio == jjo)
        {
            // This CAN fail!
            // assert(root(parent, iio, padding) ==
            //        root(parent, jjo, padding));
            break;
        }
        else if (iio > jjo)
        {
            const auto tmp = jjo;
            jjo = iio;
            iio = tmp;
        }
        assert(iio < jjo);
        // MUST BE atomicCAS, not atomicMin!
        atomicCAS(parent + jjo, jjo, iio);
    }
}

__global__ void run_kernel_pre(const bool* cg, unsigned long long* parentg,
                               const unsigned n)
{
    // Needs padding?
    extern __shared__ unsigned work[];
    auto parent = reinterpret_cast<unsigned*>(&work[0]);
    auto c = reinterpret_cast<bool*>(&parent[n * (n + 1)]);
    const auto ne = n * n;
    const auto cols_glob = static_cast<unsigned long long>(n) * gridDim.y;
    const auto offset = n * (blockIdx.x * cols_glob + blockIdx.y);
#define index(i, j) ((i) * (n + 1) + (j))
    {
        // Initialization
        for (auto ii = threadIdx.x; ii < ne; ii += blockDim.x)
        {
            const auto i = ii / n;
            const auto j = ii % n;
            const auto src = offset + i * cols_glob + j;
            const auto dst = index(i, j);
            // Memory offset, not index
            parent[dst] = dst;
            c[dst] = cg[src];
        }
    }
    __syncthreads();
    {
        // Rowwise merge
        for (auto ii = threadIdx.x; ii < ne; ii += blockDim.x)
        {
            const auto i = ii / n;
            const auto j = ii % n;
            const auto dst = index(i, j);
            const auto src = index(i, j - 1);
            if (j != 0 && c[dst] == c[src])
            {
                assert(parent[src] < parent[dst]);
                parent[dst] = parent[src];
            }
        }
    }
    __syncthreads();
    {
        // Columnwise merge
        for (auto ii = threadIdx.x; ii < ne; ii += blockDim.x)
        {
            const auto i = ii / n;
            const auto j = ii % n;
            const auto dst = index(i, j);
            const auto src = index(i - 1, j);
            if (i != 0 && c[dst] == c[src])
            {
                assert(parent[src] < parent[dst]);
                parent[dst] = parent[src];
            }
        }
    }
    __syncthreads();
    {
        // Path compression (pre)
        for (auto ii = threadIdx.x; ii < ne; ii += blockDim.x)
        {
            const auto i = ii / n;
            const auto j = ii % n;
            // Pass by offset
            root(parent, index(i, j), 1U);
        }
    }
    __syncthreads();
    {
        // Rowwise merge
        for (auto ii = threadIdx.x; ii < ne; ii += blockDim.x)
        {
            const auto i = ii / n;
            const auto j = ii % n;
            const auto tgt1 = index(i, j);
            if (j != 0)
            {
                const auto tgt2 = index(i, j - 1);
                if (c[tgt1] == c[tgt2])
                {
                    merge_kernel_block(parent, parent[tgt1], parent[tgt2], 1U);
                }
            }
        }
    }
    {
        // Colwise merge
        for (auto ii = threadIdx.x; ii < ne; ii += blockDim.x)
        {
            const auto i = ii / n;
            const auto j = ii % n;
            const auto tgt1 = index(i, j);
            if (i != 0)
            {
                const auto tgt2 = index(i - 1, j);
                if (c[tgt1] == c[tgt2])
                {
                    merge_kernel_block(parent, parent[tgt1], parent[tgt2], 1U);
                }
            }
        }
    }
    __syncthreads();
    {
        // Path compression (post)
        for (auto ii = threadIdx.x; ii < ne; ii += blockDim.x)
        {
            const auto i = ii / n;
            const auto j = ii % n;
            // Pass by offset
            root(parent, index(i, j), 1U);
        }
    }
    __syncthreads();
    {
        // Fetch from shared memory
        for (auto ii = threadIdx.x; ii < ne; ii += blockDim.x)
        {
            const auto i = ii / n;
            const auto j = ii % n;
            // Decode from offsets
            const auto iip_offset = parent[index(i, j)];
            const auto ip = iip_offset / (n + 1);
            const auto jp = iip_offset % (n + 1);
            assert(ip < n);
            assert(jp < n);
            const auto dst = offset + i * cols_glob + j;
            parentg[dst] = offset + ip * cols_glob + jp;
        }
    }
}

__global__ void run_kernel_post(const bool* cg, unsigned long long* parentg,
                                const unsigned n)
{
    // Glue
    const auto cols_glob = static_cast<unsigned long long>(n) * gridDim.y;
    const auto offset = static_cast<unsigned long long>(n) *
                        (blockIdx.x * cols_glob + blockIdx.y);
    const auto ne = n * n;
    for (auto k = threadIdx.x; k < n; k += blockDim.x)
    {
        if (blockIdx.x != 0)
        {
            const auto tgt1 = offset + k;
            assert(tgt1 >= cols_glob);
            const auto tgt2 = tgt1 - cols_glob;
            if (cg[tgt1] == cg[tgt2])
            {
                merge_kernel(parentg, parentg[tgt1], parentg[tgt2], 0ULL);
            }
        }
        if (blockIdx.y != 0)
        {
            const auto tgt1 = offset + k * cols_glob;
            assert(tgt1 >= 1);
            const auto tgt2 = tgt1 - 1;
            if (cg[tgt1] == cg[tgt2])
            {
                merge_kernel(parentg, parentg[tgt1], parentg[tgt2], 0ULL);
            }
        }
    }
    __syncthreads();
    for (auto ii = threadIdx.x; ii < ne; ii += blockDim.x)
    {
        // Path compression (final)
        const auto i = ii / n;
        const auto j = ii % n;
        const auto src = offset + i * cols_glob + j;
        root(parentg, src, 0ULL);
    }
}
