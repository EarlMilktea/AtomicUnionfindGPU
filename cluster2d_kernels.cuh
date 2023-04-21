#pragma once

__global__ void run_kernel_pre(const bool*, unsigned long long*,
                               const unsigned);

__global__ void run_kernel_post(const bool*, unsigned long long*,
                                const unsigned);
