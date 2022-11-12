I recommend reading [PyTorch's JIT  overview](https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/OVERVIEW.md) - it is a great summary of the JIT compiler and IR.
<br>
I'm bringing here some highlights:
* [TorchScript IR](https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/OVERVIEW.md#graph):
  * Graph - DAG. Owns Nodes, Blocks, Values.
  * Node - operation instruction. A Node has a "kind" which identifies the operation represented by the Node (e.g. "aten::mul"). A Node has attributes.
  * Block - sequential Node container
  * Value - Node input/output. Value has a type (int64_t, double, Tensor, Graph, std::string, List)

Each Value is defined (produced) by exactly one Node.

"A NodeKind is a Symbol object, which is just an interned string inside some namespace. Symbols can be created from strings, e.g. through Symbol::fromQualString("aten::add"), so there is not a closed set of NodeKind values that a Node might have. This design was chosen to allow the open registration of new operators and user-defined operators."

 To help avoid this, Graph provides the method Graph::insert for constructing Nodes that guarantees Nodes have the correct setup. This method uses the database of registered Operators and their FunctionSchema to construct Nodes using that schema.

PyTorch IR supports function overloading, which means that a single NodeKind may correspond to multiple operators.

SSA - single-static assignment (SSA) form, meaning that each Value has precisely one defining Node that can be looked up directly from the Value (Node* n = v.node()).

Torch resources:
* [pytorch design philosophy](https://pytorch.org/docs/stable/community/design.html)
* https://open.spotify.com/show/6UzHKeiy368jKfQMKKvJY5
