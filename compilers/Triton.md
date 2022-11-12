[github repo](https://github.com/openai/triton/tree/master)

https://triton-lang.org/master/programming-guide/chapter-1/introduction.html

> The main challenge posed by our proposed paradigm is that of work scheduling, i.e., how the work done by each program instance should be partitioned for efficient execution on modern GPUs. To address this issue, the Triton compiler makes heavy use of block-level data-flow analysis, a technique for scheduling iteration blocks statically based on the control- and data-flow structure of the target program. The resulting system actually works surprisingly well: our compiler manages to apply a broad range of interesting optimization automatically (e.g., automatic coalescing, thread swizzling, pre-fetching, automatic vectorization, tensor core-aware instruction selection, shared memory allocation/synchronization, asynchronous copy scheduling).
