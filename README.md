# DL Compiler Survey
A survey of Deep Learning optimizers, compilers and everything in between.

<H2>Terminology</H2>

| Concept | Meaning |
| ------- | ------- |
| Deep Learning Model | | 
| DL Compute Graph | |
| Graph optimization | |
| DL Optimizer | |
| DL Compiler | |
| LLVM | |
| Polyhedral | |
| Intermediate Representation (IR) | |
| Auto Tuning | |
| Kernel Fusion | |
| Dynamic Shape Inference | |
| Data-dependent Shapes | |
| Lowering | Rewriting higher abstraction operation (or construct) in terms of a sequence of simpler abstractions |
| AOT | Ahead-of-Time compilation |
| JIT | Just-in-Time compilation |

<H2>Compiler Frameworks</H2>

| Library       | DL Framework  | Status | Language | Use Case | Quantization | Sparsity | Distributed  | OSS | Platform |
| ------------- | ------------- | ------ | -------- | -------- | ------------ | -------- | ------------ | --- | -------- |
| [MLIR]()  |   | 
| [TVM](https://tvm.apache.org/) | Keras, MXNet, PyTorch, Tensorflow, CoreML, DarkNet, ONNX | status? | lang? | Inference | Y | ? | ? | Y | CPUs, GPUs and accelerators
| | |

<H2>Intermediate Representations</H2>

* ONNX
* TorchScript (PyTorch)
* FX (PyTorch)
* Relay (TVM)
* XLA HLO (XLA)

<H2>Compilers and Optimizers</H2>

| Library       | DL Framework  | Status | Language | Use Case | Quantization | Sparsity | Distributed  | OSS | Platform | Compiler FW | IR | Mixed Precision
| ------------- | ------------- | ------ | -------- | -------- | ------------ | -------- | ------------ | --- | -------- | ----------- | -- | ---------------
| [IREE](https://github.com/iree-org/iree)  | TF, TFLite, PyTorch, JAX | Early development | C, Python, Java | Training, Inference | N | N | ? | Y | ? | MLIR |
| [TensorComprehensions](https://github.com/facebookresearch/TensorComprehensions) | PyTorch, Caffe2 | ? | C++ | Inference | ? | ? | ? | Y | ? | - |
| nGraph  |   |
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

<H2>Related</H2>

| Library       | DL Framework  | Status | Language | Use Case | Quantization | Sparsity | Distributed  | OSS | Platform | Compiler FW |
| ------------- | ------------- | ------ | -------- | -------- | ------------ | -------- | ------------ | --- | -------- | ----------- |
| [Taco (Tensor Algebra Compiler)](http://tensor-compiler.org/index.html) | |
| SymPy |
| [Numba](https://numba.pydata.org/numba-doc/latest/user/5minguide.html) | |

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

