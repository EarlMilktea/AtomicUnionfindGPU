#pragma once

#include <iostream>

#ifndef NDEBUG
#define cudaCheck(expr)                                                       \
    do                                                                        \
    {                                                                         \
        const cudaError_t err = (expr);                                       \
        if (err != cudaSuccess)                                               \
        {                                                                     \
            std::cerr << cudaGetErrorString(err) << " at " << __FILE__ << ":" \
                      << __LINE__ << std::endl;                               \
            std::exit(1);                                                     \
        }                                                                     \
    } while (false);
#else
#define cudaCheck(expr) (expr)
#endif
