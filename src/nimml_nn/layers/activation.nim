## Activation Layers
##
## Non-linear activation functions for neural networks.

import std/[strformat]
import nimml_core
import ../module

type
  ReLU* = ref object of Module
    ## Rectified Linear Unit: max(0, x)
    inplace*: bool

  LeakyReLU* = ref object of Module
    ## Leaky ReLU: max(negative_slope * x, x)
    negativeSlope*: float
    inplace*: bool

  PReLU* = ref object of Module
    ## Parametric ReLU with learned slope
    numParameters*: int
    weight*: Parameter

  ELU* = ref object of Module
    ## Exponential Linear Unit
    alpha*: float
    inplace*: bool

  SELU* = ref object of Module
    ## Scaled ELU (self-normalizing)
    inplace*: bool

  GELU* = ref object of Module
    ## Gaussian Error Linear Unit
    approximate*: string  # "none" or "tanh"

  Sigmoid* = ref object of Module
    ## Sigmoid: 1 / (1 + exp(-x))

  Tanh* = ref object of Module
    ## Hyperbolic tangent

  Softmax* = ref object of Module
    ## Softmax along a dimension
    dim*: int

  LogSoftmax* = ref object of Module
    ## Log of softmax (numerically stable)
    dim*: int

  Softplus* = ref object of Module
    ## Softplus: log(1 + exp(x))
    beta*: float
    threshold*: float

  Softsign* = ref object of Module
    ## Softsign: x / (1 + |x|)

  Hardtanh* = ref object of Module
    ## Clamped linear: clamp(x, minVal, maxVal)
    minVal*: float
    maxVal*: float
    inplace*: bool

  Hardswish* = ref object of Module
    ## Hard swish activation
    inplace*: bool

  Hardsigmoid* = ref object of Module
    ## Hard sigmoid activation
    inplace*: bool

  SiLU* = ref object of Module
    ## Sigmoid Linear Unit (Swish): x * sigmoid(x)
    inplace*: bool

  Mish* = ref object of Module
    ## Mish: x * tanh(softplus(x))
    inplace*: bool

  GLU* = ref object of Module
    ## Gated Linear Unit
    dim*: int

# ReLU

proc newReLU*(inplace: bool = false): ReLU =
  ## Create ReLU activation
  result = ReLU()
  result.initModule("ReLU")
  result.inplace = inplace

method forward*(r: ReLU, inputs: varargs[TensorRef]): TensorRef =
  ## Forward: max(0, x)
  if inputs.len == 0:
    return nil
  # Output has same shape as input
  newTensorRef(inputs[0].shape, inputs[0].dtype)

# LeakyReLU

proc newLeakyReLU*(negativeSlope: float = 0.01, inplace: bool = false): LeakyReLU =
  ## Create LeakyReLU activation
  result = LeakyReLU()
  result.initModule("LeakyReLU")
  result.negativeSlope = negativeSlope
  result.inplace = inplace

method forward*(l: LeakyReLU, inputs: varargs[TensorRef]): TensorRef =
  ## Forward: max(negative_slope * x, x)
  if inputs.len == 0:
    return nil
  newTensorRef(inputs[0].shape, inputs[0].dtype)

# PReLU

proc newPReLU*(numParameters: int = 1): PReLU =
  ## Create Parametric ReLU
  ##
  ## Args:
  ##   numParameters: Number of learnable parameters (1 or channels)
  result = PReLU()
  result.initModule("PReLU")
  result.numParameters = numParameters
  result.weight = result.registerParameter("weight", newShape(numParameters), dtFloat32)

method forward*(p: PReLU, inputs: varargs[TensorRef]): TensorRef =
  ## Forward: max(0, x) + weight * min(0, x)
  if inputs.len == 0:
    return nil
  newTensorRef(inputs[0].shape, inputs[0].dtype)

# ELU

proc newELU*(alpha: float = 1.0, inplace: bool = false): ELU =
  ## Create ELU activation
  result = ELU()
  result.initModule("ELU")
  result.alpha = alpha
  result.inplace = inplace

method forward*(e: ELU, inputs: varargs[TensorRef]): TensorRef =
  ## Forward: x if x > 0 else alpha * (exp(x) - 1)
  if inputs.len == 0:
    return nil
  newTensorRef(inputs[0].shape, inputs[0].dtype)

# SELU

proc newSELU*(inplace: bool = false): SELU =
  ## Create SELU activation
  result = SELU()
  result.initModule("SELU")
  result.inplace = inplace

method forward*(s: SELU, inputs: varargs[TensorRef]): TensorRef =
  ## Forward: scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
  if inputs.len == 0:
    return nil
  newTensorRef(inputs[0].shape, inputs[0].dtype)

# GELU

proc newGELU*(approximate: string = "none"): GELU =
  ## Create GELU activation
  ##
  ## Args:
  ##   approximate: "none" for exact, "tanh" for faster approximation
  result = GELU()
  result.initModule("GELU")
  result.approximate = approximate

method forward*(g: GELU, inputs: varargs[TensorRef]): TensorRef =
  ## Forward: x * Phi(x) where Phi is CDF of standard normal
  if inputs.len == 0:
    return nil
  newTensorRef(inputs[0].shape, inputs[0].dtype)

# Sigmoid

proc newSigmoid*(): Sigmoid =
  ## Create Sigmoid activation
  result = Sigmoid()
  result.initModule("Sigmoid")

method forward*(s: Sigmoid, inputs: varargs[TensorRef]): TensorRef =
  ## Forward: 1 / (1 + exp(-x))
  if inputs.len == 0:
    return nil
  newTensorRef(inputs[0].shape, inputs[0].dtype)

# Tanh

proc newTanh*(): Tanh =
  ## Create Tanh activation
  result = Tanh()
  result.initModule("Tanh")

method forward*(t: Tanh, inputs: varargs[TensorRef]): TensorRef =
  ## Forward: tanh(x)
  if inputs.len == 0:
    return nil
  newTensorRef(inputs[0].shape, inputs[0].dtype)

# Softmax

proc newSoftmax*(dim: int = -1): Softmax =
  ## Create Softmax activation
  ##
  ## Args:
  ##   dim: Dimension to apply softmax (default -1 for last dim)
  result = Softmax()
  result.initModule("Softmax")
  result.dim = dim

method forward*(s: Softmax, inputs: varargs[TensorRef]): TensorRef =
  ## Forward: exp(x) / sum(exp(x)) along dim
  if inputs.len == 0:
    return nil
  newTensorRef(inputs[0].shape, inputs[0].dtype)

# LogSoftmax

proc newLogSoftmax*(dim: int = -1): LogSoftmax =
  ## Create LogSoftmax activation
  result = LogSoftmax()
  result.initModule("LogSoftmax")
  result.dim = dim

method forward*(l: LogSoftmax, inputs: varargs[TensorRef]): TensorRef =
  ## Forward: log(softmax(x))
  if inputs.len == 0:
    return nil
  newTensorRef(inputs[0].shape, inputs[0].dtype)

# Softplus

proc newSoftplus*(beta: float = 1.0, threshold: float = 20.0): Softplus =
  ## Create Softplus activation
  ##
  ## Args:
  ##   beta: Coefficient for scaling
  ##   threshold: Values above this revert to linear
  result = Softplus()
  result.initModule("Softplus")
  result.beta = beta
  result.threshold = threshold

method forward*(s: Softplus, inputs: varargs[TensorRef]): TensorRef =
  ## Forward: (1/beta) * log(1 + exp(beta * x))
  if inputs.len == 0:
    return nil
  newTensorRef(inputs[0].shape, inputs[0].dtype)

# Softsign

proc newSoftsign*(): Softsign =
  ## Create Softsign activation
  result = Softsign()
  result.initModule("Softsign")

method forward*(s: Softsign, inputs: varargs[TensorRef]): TensorRef =
  ## Forward: x / (1 + |x|)
  if inputs.len == 0:
    return nil
  newTensorRef(inputs[0].shape, inputs[0].dtype)

# Hardtanh

proc newHardtanh*(minVal: float = -1.0, maxVal: float = 1.0,
                  inplace: bool = false): Hardtanh =
  ## Create Hardtanh (clamped linear) activation
  result = Hardtanh()
  result.initModule("Hardtanh")
  result.minVal = minVal
  result.maxVal = maxVal
  result.inplace = inplace

method forward*(h: Hardtanh, inputs: varargs[TensorRef]): TensorRef =
  ## Forward: clamp(x, minVal, maxVal)
  if inputs.len == 0:
    return nil
  newTensorRef(inputs[0].shape, inputs[0].dtype)

# Hardswish

proc newHardswish*(inplace: bool = false): Hardswish =
  ## Create Hardswish activation
  result = Hardswish()
  result.initModule("Hardswish")
  result.inplace = inplace

method forward*(h: Hardswish, inputs: varargs[TensorRef]): TensorRef =
  ## Forward: x * relu6(x + 3) / 6
  if inputs.len == 0:
    return nil
  newTensorRef(inputs[0].shape, inputs[0].dtype)

# Hardsigmoid

proc newHardsigmoid*(inplace: bool = false): Hardsigmoid =
  ## Create Hardsigmoid activation
  result = Hardsigmoid()
  result.initModule("Hardsigmoid")
  result.inplace = inplace

method forward*(h: Hardsigmoid, inputs: varargs[TensorRef]): TensorRef =
  ## Forward: clamp((x + 3) / 6, 0, 1)
  if inputs.len == 0:
    return nil
  newTensorRef(inputs[0].shape, inputs[0].dtype)

# SiLU (Swish)

proc newSiLU*(inplace: bool = false): SiLU =
  ## Create SiLU (Swish) activation
  result = SiLU()
  result.initModule("SiLU")
  result.inplace = inplace

method forward*(s: SiLU, inputs: varargs[TensorRef]): TensorRef =
  ## Forward: x * sigmoid(x)
  if inputs.len == 0:
    return nil
  newTensorRef(inputs[0].shape, inputs[0].dtype)

# Mish

proc newMish*(inplace: bool = false): Mish =
  ## Create Mish activation
  result = Mish()
  result.initModule("Mish")
  result.inplace = inplace

method forward*(m: Mish, inputs: varargs[TensorRef]): TensorRef =
  ## Forward: x * tanh(softplus(x))
  if inputs.len == 0:
    return nil
  newTensorRef(inputs[0].shape, inputs[0].dtype)

# GLU

proc newGLU*(dim: int = -1): GLU =
  ## Create Gated Linear Unit
  ##
  ## Args:
  ##   dim: Dimension to split on (default -1)
  result = GLU()
  result.initModule("GLU")
  result.dim = dim

method forward*(g: GLU, inputs: varargs[TensorRef]): TensorRef =
  ## Forward: a * sigmoid(b) where (a, b) = split(input)
  if inputs.len == 0:
    return nil

  let input = inputs[0]
  let dims = input.shape.dims

  if dims.len == 0:
    raise newException(ModuleError, "GLU requires at least 1D input")

  # Normalize dim
  let dim = if g.dim < 0: dims.len + g.dim else: g.dim

  if dim < 0 or dim >= dims.len:
    raise newException(ModuleError, fmt"GLU dim {g.dim} out of bounds")

  if dims[dim] mod 2 != 0:
    raise newException(ModuleError,
      fmt"GLU requires even size on dim {dim}, got {dims[dim]}")

  # Output has half the size on the specified dimension
  var outputDims = dims
  outputDims[dim] = dims[dim] div 2

  newTensorRef(newShape(outputDims), input.dtype)

# Extra repr for printing

proc extraRepr*(r: ReLU): string =
  if r.inplace: "inplace=True" else: ""

proc extraRepr*(l: LeakyReLU): string =
  result = fmt"negative_slope={l.negativeSlope}"
  if l.inplace:
    result &= ", inplace=True"

proc extraRepr*(s: Softmax): string =
  fmt"dim={s.dim}"

proc extraRepr*(h: Hardtanh): string =
  result = fmt"min_val={h.minVal}, max_val={h.maxVal}"
  if h.inplace:
    result &= ", inplace=True"
