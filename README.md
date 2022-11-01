# dl-compilers-survey
A survey of Deep Learning optimizers and compilers

Terminology:
* Deep Learning Model
* DL Compute Graph
* Graph optimization
* DL Optimizer
* DL Compiler
* LLVM
* Polyhedral
* Autograd
 
Compiler Frameworks
| Library       | DL Framework  | Status | Language | Use Case | Quantization | Sparsity | Distributed  | OSS | Platform |
| ------------- | ------------- | ------ | -------- | -------- | ------------ | -------- | ------------ | --- | -------- |
| [MLIR]()  |   | 
| [TVM](https://tvm.apache.org/) | Keras, MXNet, PyTorch, Tensorflow, CoreML, DarkNet, ONNX | status? | lang? | Inference | Y | ? | ? | Y | CPUs, GPUs and accelerators
| | |

Compilers and Optimizers

| Library       | DL Framework  | Status | Language | Use Case | Quantization | Sparsity | Distributed  | OSS | Platform | Compiler FW |
| ------------- | ------------- | ------ | -------- | -------- | ------------ | -------- | ------------ | --- | -------- | ----------- |
| [IREE](https://github.com/iree-org/iree)  | TF, TFLite, PyTorch, JAX | Early development | C, Python, Java | Training, Inference | N | N | ? | Y | ? | MLIR |
| [TensorComprehensions](https://github.com/facebookresearch/TensorComprehensions) | PyTorch, Caffe2 | ? | C++ | Inference | ? | ? | ? | Y | ? | - |
| NGraph  |   |
| [XLA](https://www.tensorflow.org/xla/)  | Tensorflow, Pytorch, Julia, JAX, Nx | Active | 
| [Glow](https://github.com/pytorch/glow) | Pytorch | Active | C++ | Inference | Y | ? | ? | Y | Accelerators | - |
| [Myelin (Sling)](https://github.com/ringgaard/sling/blob/master/doc/guide/myelin.md) | | | C++, Python | Training, Inference | 
| TensorFlow Grappler | |
| [ONNC (Open NN Compiler)](https://onnc.ai/#) | ONNX | 
| ONNX Runtime | |
| [NUPHAR](https://github.com/microsoft/onnxruntime-openenclave/blob/openenclave-public/docs/execution_providers/Nuphar-ExecutionProvider.md) | ONNX Runtime | Preview 
| TensorRT | |
| PyTorch JIT | |
| JAX | TF | 
| TorchScript | |
| LazyTensor | |
| Shark | |
| OpenAI Triton | | 
| NVFuser | | 
| Dojo (Tesla) | |
| TorchDynamo/TorchInductor ||


Related

| Library       | DL Framework  | Status | Language | Use Case | Quantization | Sparsity | Distributed  | OSS | Platform | Compiler FW |
| ------------- | ------------- | ------ | -------- | -------- | ------------ | -------- | ------------ | --- | -------- | ----------- |
| [Taco (Tensor Algebra Compiler)](http://tensor-compiler.org/index.html) | |
| SymPy |


<H2>IREE (Intermediate Representation Execution Environment)</H2>

> an MLIR-based end-to-end compiler and runtime that lowers Machine Learning (ML) models to a unified IR that scales up to meet the needs of the datacenter and down to satisfy the constraints and special considerations of mobile and edge deployments.
* [Github](https://github.com/apache/tvm)
* Previously [NNVM](https://github.com/dmlc/nnvm) 

<H2>Apache TVM</H2>

> an open source machine learning compiler framework for CPUs, GPUs, and machine learning accelerators."
 
<H2>Glow</H2>

> a machine learning compiler and execution engine for hardware accelerators. It is designed to be used as a backend for high-level machine learning frameworks. The compiler is designed to allow state of the art compiler optimizations and code generation of neural network graphs.
 
<H2>Tensor Comprehensions</H2>

> a fully-functional C++ library to automatically synthesize high-performance machine learning kernels using Halide, ISL and NVRTC or LLVM. TC additionally provides basic integration with Caffe2 and PyTorch.
 
<H2>XLA (Accelerated Linear Algebra)</H2>

> a domain-specific compiler for linear algebra that can accelerate TensorFlow models with potentially no source code changes.
 
<H2>Myelin (Sling)</H2>

> a just-in-time compiler for neural networks. It compiles a flow into x64 assembly code at runtime. The flow contains the graph for the neural network computations as well as the learned weights from training the network. The generated code takes the CPU features of the machine into account when generating the code so it can take advantage of specialized features like SSE, AVX, and FMA3.

<H2> TACO: The Tensor Algebra Compiler </H2>

> A fast and versatile compiler-based library for sparse linear and tensor algebra

<H2> ONNC (Open Neural Network Compiler) </H2>

> a retargetable compilation framework designed specifically for proprietary deep learning accelerators.

<H2> NUPHAR (Neural-network Unified Preprocessing Heterogeneous ARchitecture) </H2>

> As an execution provider in the ONNX Runtime, it is built on top of TVM and LLVM to accelerate ONNX models by compiling nodes in subgraphs into optimized functions via JIT. It also provides JIT caching to save compilation time at runtime.
