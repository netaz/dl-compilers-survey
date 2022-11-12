# DL Compiler Survey
A survey of Deep Learning optimizers, compilers and everything in between.


<H2>Compiler Frameworks</H2>

| Library       | DL Framework  | Status | Language | Use Case | Quantization | Sparsity | Distributed  | OSS | Platform |
| ------------- | ------------- | ------ | -------- | -------- | ------------ | -------- | ------------ | --- | -------- |
| [MLIR]()  |   | 
| [TVM](https://tvm.apache.org/) | Keras, MXNet, PyTorch, Tensorflow, CoreML, DarkNet, ONNX | status? | lang? | Inference | Y | ? | ? | Y | CPUs, GPUs and accelerators
| | |

<H2>Intermediate Representations</H2>

<details>
  <summary>FX (PyTorch)</summary>
 
  * Represents a program as a DAG
  * Uses SSA
  * Nodes are operations (Python call-site: Module, Method, function)
  * Edges are values
</details>
<details>
  <summary>ONNX</summary>
  
  * https://github.com/onnx/onnx/blob/main/docs/IR.md
  
</details>
<details>
  <summary>TorchScript (PyTorch)</summary>
 
  * "the IR is represented in structured control flow composed of ifs & loops"

</details>
<details>
  <summary>Relay (TVM)</summary>
</details>
<details>
  <summary>ONNX</summary>
</details>
<details>
  <summary>XLA HLO</summary>
</details>

<H2>Compilers and Optimizers</H2>

| Library       | DL Framework  | Status | Language | Use Case | Quantization | Sparsity | Distributed  | OSS | Platform | Compiler FW | IR | Mixed Precision
| ------------- | ------------- | ------ | -------- | -------- | ------------ | -------- | ------------ | --- | -------- | ----------- | -- | ---------------
| [IREE](https://github.com/iree-org/iree)  | TF, TFLite, PyTorch, JAX | Early development | C, Python, Java | Training, Inference | N | N | ? | Y | ? | MLIR |
| [TensorComprehensions](https://github.com/facebookresearch/TensorComprehensions) | PyTorch, Caffe2 | ? | C++ | Inference | ? | ? | ? | Y | ? | - |
| nGraph (Intel) |   |
| [XLA](https://www.tensorflow.org/xla/)  | Tensorflow, Pytorch, Julia, JAX, Nx | Active | 
| [Glow](https://github.com/pytorch/glow) | Pytorch | Active | C++ | Inference | Y | ? | ? | Y | Accelerators | - |
| [Myelin (Sling)](https://github.com/ringgaard/sling/blob/master/doc/guide/myelin.md) | | | C++, Python | Training, Inference | 
| [TensorFlow Grappler](https://www.tensorflow.org/guide/graph_optimization#overview) | |
| [ONNC (Open NN Compiler)](https://onnc.ai/#) | ONNX | 
| ONNX Runtime | |
| [NUPHAR](https://github.com/microsoft/onnxruntime-openenclave/blob/openenclave-public/docs/execution_providers/Nuphar-ExecutionProvider.md) | ONNX Runtime | Preview 
| TensorRT | |
| PyTorch JIT | |
| JAX | TF | 
| TorchScript | |
| LazyTensor | |
| Shark | |
| [OpenAI Triton](https://github.com/openai/triton) | | 
| NVFuser | | 
| Dojo (Tesla) | |
| TorchDynamo/TorchInductor ||
| [NNC](https://dev-discuss.pytorch.org/t/nnc-walkthrough-how-pytorch-ops-get-fused/125/4) | |
| [DeepSpeed](https://github.com/microsoft/DeepSpeed) | | 
| AIT (AI Template) | | 
| [Kernl](https://github.com/ELS-RD/kernl/) | |
| [NNFusion](https://github.com/microsoft/nnfusion) | |
| [Antares](https://github.com/microsoft/antares) | |
| [Tiramisu](http://tiramisu-compiler.org/) | |

<H2>Related</H2>

| Library       | DL Framework  | Status | Language | Use Case | Quantization | Sparsity | Distributed  | OSS | Platform | Compiler FW |
| ------------- | ------------- | ------ | -------- | -------- | ------------ | -------- | ------------ | --- | -------- | ----------- |
| [Taco (Tensor Algebra Compiler)](http://tensor-compiler.org/index.html) | |
| SymPy |
| [Numba](https://numba.pydata.org/numba-doc/latest/user/5minguide.html) | |

<H2>Academic Research</H2>
 
| Paper | Date | Details | 
| ------|------|---------|
| [DISC : A Dynamic Shape Compiler for Machine Learning Workloads](https://arxiv.org/pdf/2103.05288.pdf) | Nov 2021 | This paper provides a compiler system to natively support optimization for dynamic shape workloads <br> MLIR based, extends XLA HLO |
| [Nimble: Efficiently Compiling Dynamic Neural Networks for Model Inference](https://arxiv.org/abs/2006.03031) | June 2020 | TVM based
| [Rammer: Enabling Holistic Deep Learning Compiler Optimizations with rTasks](https://www.usenix.org/conference/osdi20/presentation/ma)| Nov 2020 | A DNN compiler design that optimizes the execution of DNN workloads on massively parallel accelerators. Rammer generates an efficient static spatio-temporal schedule for a DNN at compile time to minimize scheduling overhead. It maximizes hardware utilization by holistically exploiting parallelism through inter- and intra- operator co-scheduling. Rammer achieves this by proposing several novel, hardware neutral, and clean abstractions for the computation tasks and the hardware accelerators. These abstractions expose a much richer scheduling space to Rammer, which employs several heuristics to explore this space and finds efficient schedules.<br> Neta: scheduling parallelism. Break kernel to units (e.g. tiles) and allow concurrent scheduling of compute units from several layers. Choose kernels based on resource limits, not latency optimization when parallelizing <br> Microsoft Research |

<H2>Misc</H2>

* based on polyhedral machinery - Tiramisu, Tensor Comprehensions
* based on scheduling languages - Halide, TVM


<H2> Elevator Pitches </H2>
<H3>IREE (Intermediate Representation Execution Environment)</H3>

> an MLIR-based end-to-end compiler and runtime that lowers Machine Learning (ML) models to a unified IR that scales up to meet the needs of the datacenter and down to satisfy the constraints and special considerations of mobile and edge deployments.
* [Github](https://github.com/apache/tvm)
* Previously [NNVM](https://github.com/dmlc/nnvm) 

<H3>Apache TVM</H3>

> an open source machine learning compiler framework for CPUs, GPUs, and machine learning accelerators."
 
<H3>Glow</H3>

> a machine learning compiler and execution engine for hardware accelerators. It is designed to be used as a backend for high-level machine learning frameworks. The compiler is designed to allow state of the art compiler optimizations and code generation of neural network graphs.
 
<H3>Tensor Comprehensions</H3>

> a fully-functional C++ library to automatically synthesize high-performance machine learning kernels using Halide, ISL and NVRTC or LLVM. TC additionally provides basic integration with Caffe2 and PyTorch.
 
<H3>XLA (Accelerated Linear Algebra)</H3>

> a domain-specific compiler for linear algebra that can accelerate TensorFlow models with potentially no source code changes.

> Given a computation graph, XLA firstly translates it into HLO IR.
It then finds ops that can be fused together and generates fusion kernels, which will be cached according to fusion pattern.
The fusion pattern contains op sequence with full shape information. 
When XLA meets a fusion pattern, it will first check whether this pattern is already cached. 
It will use the binary directly if hit, otherwise it will compile for the new pattern and cache the compiled result. ([src](https://arxiv.org/pdf/2103.05288.pdf))

<H3>Myelin (Sling)</H3>

> a just-in-time compiler for neural networks. It compiles a flow into x64 assembly code at runtime. The flow contains the graph for the neural network computations as well as the learned weights from training the network. The generated code takes the CPU features of the machine into account when generating the code so it can take advantage of specialized features like SSE, AVX, and FMA3.

<H3> TACO: The Tensor Algebra Compiler </H3>

> A fast and versatile compiler-based library for sparse linear and tensor algebra

<H3> ONNC (Open Neural Network Compiler) </H3>

> a retargetable compilation framework designed specifically for proprietary deep learning accelerators.

<H3> NUPHAR (Neural-network Unified Preprocessing Heterogeneous ARchitecture) </H3>

> As an execution provider in the ONNX Runtime, it is built on top of TVM and LLVM to accelerate ONNX models by compiling nodes in subgraphs into optimized functions via JIT. It also provides JIT caching to save compilation time at runtime.

<H3> Numba </H3>

> Numba is a just-in-time compiler for Python that works best on code that uses NumPy arrays and functions, and loops.

<H3> NNFusion </H3>

> A flexible and efficient DNN compiler that can generate high-performance executables from a DNN model description (e.g., TensorFlow frozen models and ONNX format).

<H3> Antares </H3>

> An automatic engine for multi-platform kernel generation and optimization.

<H3> Tiramisu </H3>

> Tiramisu is a polyhedral compiler for dense and sparse deep learning and data parallel algorithms. It provides a simple C++ API for expressing algorithms and how these algorithms should be optimized by the compiler.
