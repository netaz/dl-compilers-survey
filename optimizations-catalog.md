A catalog of DL compiler optimizations

Also see [this list](https://en.wikipedia.org/wiki/Optimizing_compiler) of compiler optimizations, many of which are also applicable to DL compilers.

<H2>Graph Transformation Optimizations</H2>
DAG-to-DAG transformations.

* Horizontal Fusion
* Vertical Fusion
* Constant Folding
* CSE (Dedouping)
* NoP and Identity removal
* Dead-code Elimination -
  Dead code is code that does not have side effects. In DL models an operation that is not contributing to the generation of any of the outputs does not have side effects.
* Pattern Matching
* BatchNorm Folding
* Shape Pushdown
* Constant Pushdown
* Concat Elision
* Pushdown
* Hoisting
* [Canonicalization](https://sunfishcode.github.io/blog/2018/10/22/Canonicalization.html)

<H2> Other Optimizations </H2>

* Layout Optimization
* Memory Planning (Liveness Analysis)
