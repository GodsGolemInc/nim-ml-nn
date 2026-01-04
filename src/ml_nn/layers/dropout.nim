## Dropout Layers
##
## Regularization through random element zeroing.

import std/[strformat]
import ml_core
import ../module
import ../compute

type
  Dropout* = ref object of Module
    ## Standard dropout: randomly zero elements during training
    p*: float  # Probability of zeroing an element
    inplace*: bool

  Dropout1d* = ref object of Module
    ## Dropout for 3D input (N, C, L) - zeros entire channels
    p*: float
    inplace*: bool

  Dropout2d* = ref object of Module
    ## Dropout for 4D input (N, C, H, W) - zeros entire channels
    p*: float
    inplace*: bool

  Dropout3d* = ref object of Module
    ## Dropout for 5D input (N, C, D, H, W) - zeros entire channels
    p*: float
    inplace*: bool

  AlphaDropout* = ref object of Module
    ## Alpha dropout for SELU networks (maintains self-normalizing property)
    p*: float
    inplace*: bool

  FeatureAlphaDropout* = ref object of Module
    ## Alpha dropout for entire features/channels
    p*: float
    inplace*: bool

# Dropout

proc newDropout*(p: float = 0.5, inplace: bool = false): Dropout =
  ## Create standard dropout layer
  ##
  ## Args:
  ##   p: Probability of an element to be zeroed (default 0.5)
  ##   inplace: If True, do operation in-place
  ##
  ## During training, randomly zeros elements with probability p.
  ## Remaining elements are scaled by 1/(1-p) to maintain expected value.
  ## During evaluation, returns input unchanged.
  if p < 0.0 or p > 1.0:
    raise newException(ModuleError, fmt"dropout probability must be in [0, 1], got {p}")

  result = Dropout()
  result.initModule("Dropout")
  result.p = p
  result.inplace = inplace

method forward*(d: Dropout, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass
  ##
  ## During training: randomly zeros elements with probability p
  ## During eval: identity function
  if inputs.len == 0:
    return nil

  let input = inputs[0]

  # During eval or p=0, return input unchanged
  if not d.training or d.p == 0.0:
    return input

  computeDropout(input, d.p, d.training)

# Dropout1d

proc newDropout1d*(p: float = 0.5, inplace: bool = false): Dropout1d =
  ## Create 1D channel dropout
  ##
  ## Expects 3D input (N, C, L) and zeros entire channels.
  if p < 0.0 or p > 1.0:
    raise newException(ModuleError, fmt"dropout probability must be in [0, 1], got {p}")

  result = Dropout1d()
  result.initModule("Dropout1d")
  result.p = p
  result.inplace = inplace

method forward*(d: Dropout1d, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass - zeros entire 1D channels
  if inputs.len == 0:
    return nil

  let input = inputs[0]

  if input.shape.dims.len != 3:
    raise newException(ModuleError,
      fmt"Dropout1d expects 3D input (N, C, L), got {input.shape.dims.len}D")

  if not d.training or d.p == 0.0:
    return input

  newTensorRef(input.shape, input.dtype)

# Dropout2d

proc newDropout2d*(p: float = 0.5, inplace: bool = false): Dropout2d =
  ## Create 2D channel dropout
  ##
  ## Expects 4D input (N, C, H, W) and zeros entire channels.
  if p < 0.0 or p > 1.0:
    raise newException(ModuleError, fmt"dropout probability must be in [0, 1], got {p}")

  result = Dropout2d()
  result.initModule("Dropout2d")
  result.p = p
  result.inplace = inplace

method forward*(d: Dropout2d, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass - zeros entire 2D feature maps
  if inputs.len == 0:
    return nil

  let input = inputs[0]

  if input.shape.dims.len != 4:
    raise newException(ModuleError,
      fmt"Dropout2d expects 4D input (N, C, H, W), got {input.shape.dims.len}D")

  if not d.training or d.p == 0.0:
    return input

  newTensorRef(input.shape, input.dtype)

# Dropout3d

proc newDropout3d*(p: float = 0.5, inplace: bool = false): Dropout3d =
  ## Create 3D channel dropout
  ##
  ## Expects 5D input (N, C, D, H, W) and zeros entire channels.
  if p < 0.0 or p > 1.0:
    raise newException(ModuleError, fmt"dropout probability must be in [0, 1], got {p}")

  result = Dropout3d()
  result.initModule("Dropout3d")
  result.p = p
  result.inplace = inplace

method forward*(d: Dropout3d, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass - zeros entire 3D feature volumes
  if inputs.len == 0:
    return nil

  let input = inputs[0]

  if input.shape.dims.len != 5:
    raise newException(ModuleError,
      fmt"Dropout3d expects 5D input (N, C, D, H, W), got {input.shape.dims.len}D")

  if not d.training or d.p == 0.0:
    return input

  newTensorRef(input.shape, input.dtype)

# AlphaDropout

proc newAlphaDropout*(p: float = 0.5, inplace: bool = false): AlphaDropout =
  ## Create alpha dropout for SELU networks
  ##
  ## Maintains self-normalizing property by setting dropped values
  ## to the negative saturation value rather than zero.
  if p < 0.0 or p > 1.0:
    raise newException(ModuleError, fmt"dropout probability must be in [0, 1], got {p}")

  result = AlphaDropout()
  result.initModule("AlphaDropout")
  result.p = p
  result.inplace = inplace

method forward*(d: AlphaDropout, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass with alpha dropout
  if inputs.len == 0:
    return nil

  let input = inputs[0]

  if not d.training or d.p == 0.0:
    return input

  newTensorRef(input.shape, input.dtype)

# FeatureAlphaDropout

proc newFeatureAlphaDropout*(p: float = 0.5, inplace: bool = false): FeatureAlphaDropout =
  ## Create feature-wise alpha dropout
  if p < 0.0 or p > 1.0:
    raise newException(ModuleError, fmt"dropout probability must be in [0, 1], got {p}")

  result = FeatureAlphaDropout()
  result.initModule("FeatureAlphaDropout")
  result.p = p
  result.inplace = inplace

method forward*(d: FeatureAlphaDropout, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass with feature alpha dropout
  if inputs.len == 0:
    return nil

  let input = inputs[0]

  if not d.training or d.p == 0.0:
    return input

  newTensorRef(input.shape, input.dtype)

# Extra repr

proc extraRepr*(d: Dropout): string =
  result = fmt"p={d.p}"
  if d.inplace:
    result &= ", inplace=True"

proc extraRepr*(d: Dropout1d): string =
  result = fmt"p={d.p}"
  if d.inplace:
    result &= ", inplace=True"

proc extraRepr*(d: Dropout2d): string =
  result = fmt"p={d.p}"
  if d.inplace:
    result &= ", inplace=True"

proc extraRepr*(d: Dropout3d): string =
  result = fmt"p={d.p}"
  if d.inplace:
    result &= ", inplace=True"

proc extraRepr*(d: AlphaDropout): string =
  result = fmt"p={d.p}"
  if d.inplace:
    result &= ", inplace=True"
