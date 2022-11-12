https://www.tensorflow.org/xla/architecture#how_does_xla_work

> XLA takes graphs ("computations") defined in HLO and compiles them into machine instructions for various architectures.

> XLA is modular in the sense that it is easy to slot in an alternative backend to target some novel HW architecture.
 
> XLA comes with several optimizations and analysis passes that are target-independent, such as CSE, target-independent operation fusion, and buffer analysis for allocating runtime memory for the computation.

> After the target-independent step, XLA sends the HLO computation to a backend. The backend can perform further HLO-level optimizations, this time with target specific information and needs in mind.
