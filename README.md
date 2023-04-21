# AtomicUnionfindGPU

## Description

Parallel lock-free DSU (disjoint set union) based on atomic CAS.

```cpp
while (true)
{
    // Find root nodes
    iio = root(parent, iio, padding);
    jjo = root(parent, jjo, padding);
    if (iio == jjo)
    {
        // Successfully merged
        break;
    }
    else if (iio > jjo)
    {
        // Ensure iio <= jjo
        const auto tmp = jjo;
        jjo = iio;
        iio = tmp;
    }
    // Bind jjo to iio
    atomicCAS_block(parent + jjo, jjo, iio);
}
```

Project at Fixstarts (2023 Feb. - Apr.)
