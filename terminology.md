<H2>Terminology</H2>

| Concept | Meaning |
| ------- | ------- |
| Deep Learning Model | |
| DL Operation | |
| DL Compute Graph | |
| Program representation | |
| Program acquisition or generation | | 
| Symbolic tracing | feed fake values to the program and record the operations on these values |
| Intermediate Representation (IR) | the data structure or code used internally by a compiler or virtual machine to represent source code. An IR is designed to be conducive to further processing, such as optimization and translation. ([src](https://en.wikipedia.org/wiki/Intermediate_representation))|
| AST | Abstract Syntax Tree; used to represent the structure of program code |
| CFG | Control-Flow Graph; a representation, using graph notation, of all paths that might be traversed through a program during its execution. ([src](https://en.wikipedia.org/wiki/Control-flow_graph))|
| DDG | Data Dependence Graph; In its simplest form the DDG represents data dependencies between individual instructions. Each node in such a graph represents a single instruction and is referred to as an “atomic” node. ([src](https://llvm.org/docs/DependenceGraphs/index.html))|
| SSA | Static Single Assignment | a property of an intermediate representation (IR) that requires each variable to be assigned exactly once and defined before it is used. Existing variables in the original IR are split into versions, new variables typically indicated by the original name with a subscript in textbooks, so that every definition gets its own version. ([src](https://en.wikipedia.org/wiki/Static_single-assignment_form))|
| Graph optimization | |
| DL Optimizer | |
| DL Compiler | |
| Polyhedral | |
| Graph rewriting | The application of a set of compute graph transformations that change the graph structure but maintain semantics. |
| Auto Tuning | |
| Kernel Fusion | |
| Type inference / propagation | |
| Dynamic Shape Analysis/Inference | |
| Data-dependent Shapes | |
| Lowering | Rewriting higher abstraction operation (or construct) in terms of a sequence of simpler abstractions |
| Lifting / hoisting | | 
| AOT | Ahead-of-Time compilation |
| JIT | Just-in-Time compilation |
| LLVM | |
