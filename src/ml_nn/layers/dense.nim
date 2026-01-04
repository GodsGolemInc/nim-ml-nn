## Dense Layers
##
## Fully connected (linear) layers for neural networks.

import std/[options, strformat]
import ml_core
import ../module
import ../compute

type
  Linear* = ref object of Module
    ## Fully connected layer: output = input @ weight^T + bias
    inFeatures*: int
    outFeatures*: int
    useBias*: bool
    weight*: Parameter
    bias*: Option[Parameter]

  LazyLinear* = ref object of Module
    ## Linear layer with lazy initialization (infers input size)
    outFeatures*: int
    useBias*: bool
    initialized*: bool
    weight*: Option[Parameter]
    bias*: Option[Parameter]

# Weight initialization utilities

proc kaiming_uniform*(shape: Shape, fanIn: int): TensorData =
  ## Kaiming uniform initialization (He initialization)
  ## Used for ReLU-based networks
  # bound = sqrt(3.0 / fanIn) would be used for actual random initialization
  discard fanIn  # Reserved for future implementation
  newTensorData(shape, dtFloat32)

proc xavier_uniform*(shape: Shape, fanIn, fanOut: int): TensorData =
  ## Xavier/Glorot uniform initialization
  ## Used for Sigmoid/Tanh-based networks
  # bound = sqrt(6.0 / (fanIn + fanOut)) would be used for actual random initialization
  discard fanIn  # Reserved for future implementation
  discard fanOut
  newTensorData(shape, dtFloat32)

proc uniform_init*(shape: Shape, bound: float): TensorData =
  ## Uniform initialization in [-bound, bound]
  discard bound  # Reserved for future implementation
  newTensorData(shape, dtFloat32)

# Linear layer

proc newLinear*(inFeatures, outFeatures: int,
                useBias: bool = true,
                dtype: DType = dtFloat32): Linear =
  ## Create a new Linear (fully connected) layer
  ##
  ## Args:
  ##   inFeatures: Size of input features
  ##   outFeatures: Size of output features
  ##   useBias: Whether to include a bias term
  ##   dtype: Data type for parameters
  ##
  ## The weight matrix is shape (outFeatures, inFeatures).
  ## The bias vector is shape (outFeatures,).
  ##
  ## Forward: output = input @ weight^T + bias
  result = Linear()
  result.initModule("Linear")
  result.inFeatures = inFeatures
  result.outFeatures = outFeatures
  result.useBias = useBias

  # Register weight parameter
  # Shape is (outFeatures, inFeatures) for efficient matmul
  result.weight = result.registerParameter(
    "weight",
    newShape(outFeatures, inFeatures),
    dtype
  )

  # Register optional bias parameter
  if useBias:
    let biasParam = result.registerParameter(
      "bias",
      newShape(outFeatures),
      dtype
    )
    result.bias = some(biasParam)
  else:
    result.bias = none(Parameter)

method forward*(l: Linear, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass: output = input @ weight^T + bias
  ##
  ## Args:
  ##   inputs: Input tensor of shape (*, inFeatures)
  ##
  ## Returns:
  ##   Output tensor of shape (*, outFeatures)
  if inputs.len == 0:
    raise newException(ModuleError, "Linear forward requires at least one input")

  let input = inputs[0]

  # Validate input shape
  let inputDims = input.shape.dims
  if inputDims.len == 0:
    raise newException(ModuleError, "Linear input cannot be scalar")

  let lastDim = inputDims[^1]
  if lastDim != l.inFeatures:
    raise newException(ModuleError,
      fmt"Linear input size mismatch: expected {l.inFeatures}, got {lastDim}")

  # Compute actual result
  let biasData = if l.bias.isSome: l.bias.get.data else: nil
  computeLinear(input, l.weight.data, biasData)

method reset*(l: Linear) =
  ## Reset parameters using Kaiming initialization
  # Would reinitialize weight and bias to random values
  discard

proc numInputs*(l: Linear): int =
  ## Get number of input features
  l.inFeatures

proc numOutputs*(l: Linear): int =
  ## Get number of output features
  l.outFeatures

proc extraRepr*(l: Linear): string =
  ## Extra representation for printing
  result = fmt"in_features={l.inFeatures}, out_features={l.outFeatures}"
  if not l.useBias:
    result &= ", bias=False"

# LazyLinear - Linear layer with deferred initialization

proc newLazyLinear*(outFeatures: int,
                    useBias: bool = true): LazyLinear =
  ## Create a LazyLinear layer
  ##
  ## The input feature size is inferred from the first forward pass.
  result = LazyLinear()
  result.initModule("LazyLinear")
  result.outFeatures = outFeatures
  result.useBias = useBias
  result.initialized = false
  result.weight = none(Parameter)
  result.bias = none(Parameter)

proc materialize*(l: LazyLinear, inFeatures: int, dtype: DType = dtFloat32) =
  ## Materialize the lazy layer with actual dimensions
  if l.initialized:
    return

  l.weight = some(l.registerParameter(
    "weight",
    newShape(l.outFeatures, inFeatures),
    dtype
  ))

  if l.useBias:
    l.bias = some(l.registerParameter(
      "bias",
      newShape(l.outFeatures),
      dtype
    ))

  l.initialized = true

method forward*(l: LazyLinear, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass with lazy initialization
  if inputs.len == 0:
    raise newException(ModuleError, "LazyLinear forward requires at least one input")

  let input = inputs[0]
  let inputDims = input.shape.dims

  if inputDims.len == 0:
    raise newException(ModuleError, "LazyLinear input cannot be scalar")

  # Materialize on first forward
  if not l.initialized:
    l.materialize(inputDims[^1], input.dtype)

  # Calculate output shape
  var outputDims = inputDims
  outputDims[^1] = l.outFeatures
  let outputShape = newShape(outputDims)

  newTensorRef(outputShape, input.dtype)

# Bilinear layer for combining two inputs

type
  Bilinear* = ref object of Module
    ## Bilinear transformation: output = input1^T @ weight @ input2 + bias
    inFeatures1*: int
    inFeatures2*: int
    outFeatures*: int
    useBias*: bool
    weight*: Parameter
    bias*: Option[Parameter]

proc newBilinear*(inFeatures1, inFeatures2, outFeatures: int,
                  useBias: bool = true,
                  dtype: DType = dtFloat32): Bilinear =
  ## Create a Bilinear layer
  ##
  ## Args:
  ##   inFeatures1: Size of first input
  ##   inFeatures2: Size of second input
  ##   outFeatures: Size of output
  result = Bilinear()
  result.initModule("Bilinear")
  result.inFeatures1 = inFeatures1
  result.inFeatures2 = inFeatures2
  result.outFeatures = outFeatures
  result.useBias = useBias

  # Weight shape: (outFeatures, inFeatures1, inFeatures2)
  result.weight = result.registerParameter(
    "weight",
    newShape(outFeatures, inFeatures1, inFeatures2),
    dtype
  )

  if useBias:
    let biasParam = result.registerParameter("bias", newShape(outFeatures), dtype)
    result.bias = some(biasParam)
  else:
    result.bias = none(Parameter)

method forward*(b: Bilinear, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass: output = input1^T @ weight @ input2 + bias
  if inputs.len < 2:
    raise newException(ModuleError, "Bilinear forward requires two inputs")

  let input1 = inputs[0]
  let input2 = inputs[1]

  # Validate input shapes
  let dims1 = input1.shape.dims
  let dims2 = input2.shape.dims

  if dims1.len == 0 or dims2.len == 0:
    raise newException(ModuleError, "Bilinear inputs cannot be scalar")

  if dims1[^1] != b.inFeatures1:
    raise newException(ModuleError,
      fmt"Bilinear input1 size mismatch: expected {b.inFeatures1}, got {dims1[^1]}")

  if dims2[^1] != b.inFeatures2:
    raise newException(ModuleError,
      fmt"Bilinear input2 size mismatch: expected {b.inFeatures2}, got {dims2[^1]}")

  # Calculate output shape (batch dimensions + outFeatures)
  var outputDims = dims1
  outputDims[^1] = b.outFeatures
  let outputShape = newShape(outputDims)

  newTensorRef(outputShape, input1.dtype)

# Identity layer (pass-through)

type
  Identity* = ref object of Module
    ## Identity layer - returns input unchanged

proc newIdentity*(): Identity =
  ## Create an Identity layer
  result = Identity()
  result.initModule("Identity")

method forward*(i: Identity, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass - returns input unchanged
  if inputs.len > 0:
    inputs[0]
  else:
    nil

# Flatten layer

type
  Flatten* = ref object of Module
    ## Flatten layer - flattens dimensions
    startDim*: int
    endDim*: int

proc newFlatten*(startDim: int = 1, endDim: int = -1): Flatten =
  ## Create a Flatten layer
  ##
  ## Args:
  ##   startDim: First dimension to flatten (default 1, keeps batch dim)
  ##   endDim: Last dimension to flatten (default -1, all remaining)
  result = Flatten()
  result.initModule("Flatten")
  result.startDim = startDim
  result.endDim = endDim

method forward*(f: Flatten, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass - flatten specified dimensions
  if inputs.len == 0:
    return nil

  let input = inputs[0]
  let dims = input.shape.dims

  if dims.len == 0:
    return input

  # Normalize negative indices
  let startDim = if f.startDim < 0: dims.len + f.startDim else: f.startDim
  let endDim = if f.endDim < 0: dims.len + f.endDim else: f.endDim

  if startDim > endDim or startDim < 0 or endDim >= dims.len:
    return input

  # Calculate flattened size
  var flatSize = 1
  for i in startDim..endDim:
    flatSize *= dims[i]

  # Build new shape
  var newDims: seq[int] = @[]
  for i in 0..<startDim:
    newDims.add(dims[i])
  newDims.add(flatSize)
  for i in (endDim + 1)..<dims.len:
    newDims.add(dims[i])

  newTensorRef(newShape(newDims), input.dtype)

# Unflatten layer

type
  Unflatten* = ref object of Module
    ## Unflatten layer - expands a dimension
    dim*: int
    unflattendSize*: seq[int]

proc newUnflatten*(dim: int, unflattenedSize: seq[int]): Unflatten =
  ## Create an Unflatten layer
  ##
  ## Args:
  ##   dim: Dimension to unflatten
  ##   unflattenedSize: Target shape for the dimension
  result = Unflatten()
  result.initModule("Unflatten")
  result.dim = dim
  result.unflattendSize = unflattenedSize

method forward*(u: Unflatten, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass - unflatten specified dimension
  if inputs.len == 0:
    return nil

  let input = inputs[0]
  let dims = input.shape.dims

  if dims.len == 0:
    return input

  # Normalize negative index
  let dim = if u.dim < 0: dims.len + u.dim else: u.dim

  if dim < 0 or dim >= dims.len:
    raise newException(ModuleError, fmt"Unflatten dim {u.dim} out of bounds")

  # Verify product matches
  var product = 1
  for s in u.unflattendSize:
    product *= s

  if product != dims[dim]:
    raise newException(ModuleError,
      fmt"Unflatten size mismatch: {product} vs {dims[dim]}")

  # Build new shape
  var newDims: seq[int] = @[]
  for i in 0..<dim:
    newDims.add(dims[i])
  for s in u.unflattendSize:
    newDims.add(s)
  for i in (dim + 1)..<dims.len:
    newDims.add(dims[i])

  newTensorRef(newShape(newDims), input.dtype)
