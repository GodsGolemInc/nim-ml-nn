## Pooling Layers
##
## Pooling operations for neural networks.
## Includes MaxPool, AvgPool, AdaptivePool, and GlobalPool.

import std/[options, strformat]
import ml_core
import ../module
import ../compute

type
  # =============================================================================
  # 1D Pooling
  # =============================================================================

  MaxPool1d* = ref object of Module
    ## 1D max pooling over input signal (N, C, L)
    kernelSize*: int
    stride*: int
    padding*: int
    dilation*: int
    returnIndices*: bool
    ceilMode*: bool

  AvgPool1d* = ref object of Module
    ## 1D average pooling
    kernelSize*: int
    stride*: int
    padding*: int
    ceilMode*: bool
    countIncludePad*: bool

  # =============================================================================
  # 2D Pooling
  # =============================================================================

  MaxPool2d* = ref object of Module
    ## 2D max pooling over input images (N, C, H, W)
    kernelSize*: tuple[h, w: int]
    stride*: tuple[h, w: int]
    padding*: tuple[h, w: int]
    dilation*: tuple[h, w: int]
    returnIndices*: bool
    ceilMode*: bool

  AvgPool2d* = ref object of Module
    ## 2D average pooling
    kernelSize*: tuple[h, w: int]
    stride*: tuple[h, w: int]
    padding*: tuple[h, w: int]
    ceilMode*: bool
    countIncludePad*: bool
    divisorOverride*: Option[int]

  # =============================================================================
  # 3D Pooling
  # =============================================================================

  MaxPool3d* = ref object of Module
    ## 3D max pooling over volumetric data (N, C, D, H, W)
    kernelSize*: tuple[d, h, w: int]
    stride*: tuple[d, h, w: int]
    padding*: tuple[d, h, w: int]
    dilation*: tuple[d, h, w: int]
    returnIndices*: bool
    ceilMode*: bool

  AvgPool3d* = ref object of Module
    ## 3D average pooling
    kernelSize*: tuple[d, h, w: int]
    stride*: tuple[d, h, w: int]
    padding*: tuple[d, h, w: int]
    ceilMode*: bool
    countIncludePad*: bool
    divisorOverride*: Option[int]

  # =============================================================================
  # Adaptive Pooling
  # =============================================================================

  AdaptiveMaxPool1d* = ref object of Module
    ## Adaptive 1D max pooling (fixed output size)
    outputSize*: int
    returnIndices*: bool

  AdaptiveAvgPool1d* = ref object of Module
    ## Adaptive 1D average pooling (fixed output size)
    outputSize*: int

  AdaptiveMaxPool2d* = ref object of Module
    ## Adaptive 2D max pooling (fixed output size)
    outputSize*: tuple[h, w: int]
    returnIndices*: bool

  AdaptiveAvgPool2d* = ref object of Module
    ## Adaptive 2D average pooling (fixed output size)
    ## Output shape is always (N, C, outputSize.h, outputSize.w)
    outputSize*: tuple[h, w: int]

  AdaptiveMaxPool3d* = ref object of Module
    ## Adaptive 3D max pooling
    outputSize*: tuple[d, h, w: int]
    returnIndices*: bool

  AdaptiveAvgPool3d* = ref object of Module
    ## Adaptive 3D average pooling
    outputSize*: tuple[d, h, w: int]

  # =============================================================================
  # Other Pooling Types
  # =============================================================================

  LPPool1d* = ref object of Module
    ## Power-average pooling
    normType*: float
    kernelSize*: int
    stride*: int
    ceilMode*: bool

  LPPool2d* = ref object of Module
    ## 2D power-average pooling
    normType*: float
    kernelSize*: tuple[h, w: int]
    stride*: tuple[h, w: int]
    ceilMode*: bool

  FractionalMaxPool2d* = ref object of Module
    ## 2D fractional max pooling
    kernelSize*: tuple[h, w: int]
    outputSize*: Option[tuple[h, w: int]]
    outputRatio*: Option[tuple[h, w: float]]
    returnIndices*: bool

# =============================================================================
# Helper Functions
# =============================================================================

proc calcPoolOutputSize(inputSize, kernelSize, stride, padding, dilation: int,
                        ceilMode: bool): int =
  ## Calculate output size for pooling
  let dilatedKernel = dilation * (kernelSize - 1) + 1
  if ceilMode:
    (inputSize + 2 * padding - dilatedKernel + stride - 1) div stride + 1
  else:
    (inputSize + 2 * padding - dilatedKernel) div stride + 1

# =============================================================================
# MaxPool1d Implementation
# =============================================================================

proc newMaxPool1d*(kernelSize: int,
                   stride: int = 0,
                   padding: int = 0,
                   dilation: int = 1,
                   returnIndices: bool = false,
                   ceilMode: bool = false): MaxPool1d =
  ## Create a MaxPool1d layer
  ##
  ## Args:
  ##   kernelSize: Size of the pooling window
  ##   stride: Stride of the pooling (default: kernelSize)
  ##   padding: Zero-padding added to both sides
  ##   dilation: Spacing between kernel elements
  ##   returnIndices: Whether to return max indices
  ##   ceilMode: Use ceil instead of floor for output size
  result = MaxPool1d()
  result.initModule("MaxPool1d")
  result.kernelSize = kernelSize
  result.stride = if stride == 0: kernelSize else: stride
  result.padding = padding
  result.dilation = dilation
  result.returnIndices = returnIndices
  result.ceilMode = ceilMode

method forward*(pool: MaxPool1d, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass for MaxPool1d
  if inputs.len == 0:
    raise newException(ModuleError, "MaxPool1d forward requires at least one input")

  let input = inputs[0]
  let dims = input.shape.dims

  if dims.len != 3:
    raise newException(ModuleError,
      fmt"MaxPool1d expects 3D input (N,C,L), got {dims.len}D")

  computeMaxPool1d(input, pool.kernelSize, pool.stride, pool.padding, pool.dilation)

# =============================================================================
# AvgPool1d Implementation
# =============================================================================

proc newAvgPool1d*(kernelSize: int,
                   stride: int = 0,
                   padding: int = 0,
                   ceilMode: bool = false,
                   countIncludePad: bool = true): AvgPool1d =
  ## Create an AvgPool1d layer
  result = AvgPool1d()
  result.initModule("AvgPool1d")
  result.kernelSize = kernelSize
  result.stride = if stride == 0: kernelSize else: stride
  result.padding = padding
  result.ceilMode = ceilMode
  result.countIncludePad = countIncludePad

method forward*(pool: AvgPool1d, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass for AvgPool1d
  if inputs.len == 0:
    raise newException(ModuleError, "AvgPool1d forward requires at least one input")

  let input = inputs[0]
  let dims = input.shape.dims

  if dims.len != 3:
    raise newException(ModuleError,
      fmt"AvgPool1d expects 3D input (N,C,L), got {dims.len}D")

  let batchSize = dims[0]
  let channels = dims[1]
  let inputLen = dims[2]

  let outLen = calcPoolOutputSize(inputLen, pool.kernelSize, pool.stride,
                                   pool.padding, 1, pool.ceilMode)

  newTensorRef(newShape(batchSize, channels, outLen), input.dtype)

# =============================================================================
# MaxPool2d Implementation
# =============================================================================

proc newMaxPool2d*(kernelSize: int | tuple[h, w: int],
                   stride: int | tuple[h, w: int] = 0,
                   padding: int | tuple[h, w: int] = 0,
                   dilation: int | tuple[h, w: int] = 1,
                   returnIndices: bool = false,
                   ceilMode: bool = false): MaxPool2d =
  ## Create a MaxPool2d layer
  result = MaxPool2d()
  result.initModule("MaxPool2d")
  result.returnIndices = returnIndices
  result.ceilMode = ceilMode

  when kernelSize is int:
    result.kernelSize = (h: kernelSize, w: kernelSize)
  else:
    result.kernelSize = kernelSize

  when stride is int:
    if stride == 0:
      result.stride = result.kernelSize
    else:
      result.stride = (h: stride, w: stride)
  else:
    result.stride = stride

  when padding is int:
    result.padding = (h: padding, w: padding)
  else:
    result.padding = padding

  when dilation is int:
    result.dilation = (h: dilation, w: dilation)
  else:
    result.dilation = dilation

method forward*(pool: MaxPool2d, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass for MaxPool2d
  if inputs.len == 0:
    raise newException(ModuleError, "MaxPool2d forward requires at least one input")

  let input = inputs[0]
  let dims = input.shape.dims

  if dims.len != 4:
    raise newException(ModuleError,
      fmt"MaxPool2d expects 4D input (N,C,H,W), got {dims.len}D")

  computeMaxPool2d(input, pool.kernelSize.h, pool.kernelSize.w,
                   pool.stride.h, pool.stride.w,
                   pool.padding.h, pool.padding.w,
                   pool.dilation.h, pool.dilation.w)

# =============================================================================
# AvgPool2d Implementation
# =============================================================================

proc newAvgPool2d*(kernelSize: int | tuple[h, w: int],
                   stride: int | tuple[h, w: int] = 0,
                   padding: int | tuple[h, w: int] = 0,
                   ceilMode: bool = false,
                   countIncludePad: bool = true,
                   divisorOverride: Option[int] = none(int)): AvgPool2d =
  ## Create an AvgPool2d layer
  result = AvgPool2d()
  result.initModule("AvgPool2d")
  result.ceilMode = ceilMode
  result.countIncludePad = countIncludePad
  result.divisorOverride = divisorOverride

  when kernelSize is int:
    result.kernelSize = (h: kernelSize, w: kernelSize)
  else:
    result.kernelSize = kernelSize

  when stride is int:
    if stride == 0:
      result.stride = result.kernelSize
    else:
      result.stride = (h: stride, w: stride)
  else:
    result.stride = stride

  when padding is int:
    result.padding = (h: padding, w: padding)
  else:
    result.padding = padding

method forward*(pool: AvgPool2d, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass for AvgPool2d
  if inputs.len == 0:
    raise newException(ModuleError, "AvgPool2d forward requires at least one input")

  let input = inputs[0]
  let dims = input.shape.dims

  if dims.len != 4:
    raise newException(ModuleError,
      fmt"AvgPool2d expects 4D input (N,C,H,W), got {dims.len}D")

  computeAvgPool2d(input, pool.kernelSize.h, pool.kernelSize.w,
                   pool.stride.h, pool.stride.w,
                   pool.padding.h, pool.padding.w,
                   pool.countIncludePad)

# =============================================================================
# MaxPool3d Implementation
# =============================================================================

proc newMaxPool3d*(kernelSize: int | tuple[d, h, w: int],
                   stride: int | tuple[d, h, w: int] = 0,
                   padding: int | tuple[d, h, w: int] = 0,
                   dilation: int | tuple[d, h, w: int] = 1,
                   returnIndices: bool = false,
                   ceilMode: bool = false): MaxPool3d =
  ## Create a MaxPool3d layer
  result = MaxPool3d()
  result.initModule("MaxPool3d")
  result.returnIndices = returnIndices
  result.ceilMode = ceilMode

  when kernelSize is int:
    result.kernelSize = (d: kernelSize, h: kernelSize, w: kernelSize)
  else:
    result.kernelSize = kernelSize

  when stride is int:
    if stride == 0:
      result.stride = result.kernelSize
    else:
      result.stride = (d: stride, h: stride, w: stride)
  else:
    result.stride = stride

  when padding is int:
    result.padding = (d: padding, h: padding, w: padding)
  else:
    result.padding = padding

  when dilation is int:
    result.dilation = (d: dilation, h: dilation, w: dilation)
  else:
    result.dilation = dilation

method forward*(pool: MaxPool3d, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass for MaxPool3d
  if inputs.len == 0:
    raise newException(ModuleError, "MaxPool3d forward requires at least one input")

  let input = inputs[0]
  let dims = input.shape.dims

  if dims.len != 5:
    raise newException(ModuleError,
      fmt"MaxPool3d expects 5D input (N,C,D,H,W), got {dims.len}D")

  let batchSize = dims[0]
  let channels = dims[1]
  let inD = dims[2]
  let inH = dims[3]
  let inW = dims[4]

  let outD = calcPoolOutputSize(inD, pool.kernelSize.d, pool.stride.d,
                                 pool.padding.d, pool.dilation.d, pool.ceilMode)
  let outH = calcPoolOutputSize(inH, pool.kernelSize.h, pool.stride.h,
                                 pool.padding.h, pool.dilation.h, pool.ceilMode)
  let outW = calcPoolOutputSize(inW, pool.kernelSize.w, pool.stride.w,
                                 pool.padding.w, pool.dilation.w, pool.ceilMode)

  newTensorRef(newShape(batchSize, channels, outD, outH, outW), input.dtype)

# =============================================================================
# Adaptive Pooling Implementations
# =============================================================================

proc newAdaptiveMaxPool1d*(outputSize: int,
                           returnIndices: bool = false): AdaptiveMaxPool1d =
  ## Create an AdaptiveMaxPool1d layer
  result = AdaptiveMaxPool1d()
  result.initModule("AdaptiveMaxPool1d")
  result.outputSize = outputSize
  result.returnIndices = returnIndices

method forward*(pool: AdaptiveMaxPool1d, inputs: varargs[TensorRef]): TensorRef =
  if inputs.len == 0:
    raise newException(ModuleError, "AdaptiveMaxPool1d forward requires at least one input")

  let input = inputs[0]
  let dims = input.shape.dims

  if dims.len != 3:
    raise newException(ModuleError,
      fmt"AdaptiveMaxPool1d expects 3D input (N,C,L), got {dims.len}D")

  newTensorRef(newShape(dims[0], dims[1], pool.outputSize), input.dtype)

proc newAdaptiveAvgPool1d*(outputSize: int): AdaptiveAvgPool1d =
  ## Create an AdaptiveAvgPool1d layer
  result = AdaptiveAvgPool1d()
  result.initModule("AdaptiveAvgPool1d")
  result.outputSize = outputSize

method forward*(pool: AdaptiveAvgPool1d, inputs: varargs[TensorRef]): TensorRef =
  if inputs.len == 0:
    raise newException(ModuleError, "AdaptiveAvgPool1d forward requires at least one input")

  let input = inputs[0]
  let dims = input.shape.dims

  if dims.len != 3:
    raise newException(ModuleError,
      fmt"AdaptiveAvgPool1d expects 3D input (N,C,L), got {dims.len}D")

  newTensorRef(newShape(dims[0], dims[1], pool.outputSize), input.dtype)

proc newAdaptiveMaxPool2d*(outputSize: int | tuple[h, w: int],
                           returnIndices: bool = false): AdaptiveMaxPool2d =
  ## Create an AdaptiveMaxPool2d layer
  ##
  ## Output size can be int (square) or tuple
  result = AdaptiveMaxPool2d()
  result.initModule("AdaptiveMaxPool2d")
  result.returnIndices = returnIndices

  when outputSize is int:
    result.outputSize = (h: outputSize, w: outputSize)
  else:
    result.outputSize = outputSize

method forward*(pool: AdaptiveMaxPool2d, inputs: varargs[TensorRef]): TensorRef =
  if inputs.len == 0:
    raise newException(ModuleError, "AdaptiveMaxPool2d forward requires at least one input")

  let input = inputs[0]
  let dims = input.shape.dims

  if dims.len != 4:
    raise newException(ModuleError,
      fmt"AdaptiveMaxPool2d expects 4D input (N,C,H,W), got {dims.len}D")

  newTensorRef(newShape(dims[0], dims[1], pool.outputSize.h, pool.outputSize.w), input.dtype)

proc newAdaptiveAvgPool2d*(outputSize: int | tuple[h, w: int]): AdaptiveAvgPool2d =
  ## Create an AdaptiveAvgPool2d layer
  ##
  ## Commonly used with outputSize=1 for global average pooling
  result = AdaptiveAvgPool2d()
  result.initModule("AdaptiveAvgPool2d")

  when outputSize is int:
    result.outputSize = (h: outputSize, w: outputSize)
  else:
    result.outputSize = outputSize

method forward*(pool: AdaptiveAvgPool2d, inputs: varargs[TensorRef]): TensorRef =
  if inputs.len == 0:
    raise newException(ModuleError, "AdaptiveAvgPool2d forward requires at least one input")

  let input = inputs[0]
  let dims = input.shape.dims

  if dims.len != 4:
    raise newException(ModuleError,
      fmt"AdaptiveAvgPool2d expects 4D input (N,C,H,W), got {dims.len}D")

  computeAdaptiveAvgPool2d(input, pool.outputSize.h, pool.outputSize.w)

proc newAdaptiveAvgPool3d*(outputSize: int | tuple[d, h, w: int]): AdaptiveAvgPool3d =
  ## Create an AdaptiveAvgPool3d layer
  result = AdaptiveAvgPool3d()
  result.initModule("AdaptiveAvgPool3d")

  when outputSize is int:
    result.outputSize = (d: outputSize, h: outputSize, w: outputSize)
  else:
    result.outputSize = outputSize

method forward*(pool: AdaptiveAvgPool3d, inputs: varargs[TensorRef]): TensorRef =
  if inputs.len == 0:
    raise newException(ModuleError, "AdaptiveAvgPool3d forward requires at least one input")

  let input = inputs[0]
  let dims = input.shape.dims

  if dims.len != 5:
    raise newException(ModuleError,
      fmt"AdaptiveAvgPool3d expects 5D input (N,C,D,H,W), got {dims.len}D")

  newTensorRef(newShape(dims[0], dims[1],
                        pool.outputSize.d, pool.outputSize.h, pool.outputSize.w),
               input.dtype)

# =============================================================================
# Global Pooling (shortcuts)
# =============================================================================

proc newGlobalMaxPool2d*(): AdaptiveMaxPool2d =
  ## Create a global max pooling layer (output size 1x1)
  newAdaptiveMaxPool2d(1)

proc newGlobalAvgPool2d*(): AdaptiveAvgPool2d =
  ## Create a global average pooling layer (output size 1x1)
  newAdaptiveAvgPool2d(1)

proc newGlobalMaxPool1d*(): AdaptiveMaxPool1d =
  ## Create a global max pooling layer (output size 1)
  newAdaptiveMaxPool1d(1)

proc newGlobalAvgPool1d*(): AdaptiveAvgPool1d =
  ## Create a global average pooling layer (output size 1)
  newAdaptiveAvgPool1d(1)

# =============================================================================
# LP Pooling Implementations
# =============================================================================

proc newLPPool1d*(normType: float,
                  kernelSize: int,
                  stride: int = 0,
                  ceilMode: bool = false): LPPool1d =
  ## Create a power-average 1D pooling layer
  ##
  ## f(x) = (sum(|x|^p) / n)^(1/p)
  result = LPPool1d()
  result.initModule("LPPool1d")
  result.normType = normType
  result.kernelSize = kernelSize
  result.stride = if stride == 0: kernelSize else: stride
  result.ceilMode = ceilMode

method forward*(pool: LPPool1d, inputs: varargs[TensorRef]): TensorRef =
  if inputs.len == 0:
    raise newException(ModuleError, "LPPool1d forward requires at least one input")

  let input = inputs[0]
  let dims = input.shape.dims

  if dims.len != 3:
    raise newException(ModuleError,
      fmt"LPPool1d expects 3D input (N,C,L), got {dims.len}D")

  let outLen = calcPoolOutputSize(dims[2], pool.kernelSize, pool.stride, 0, 1, pool.ceilMode)
  newTensorRef(newShape(dims[0], dims[1], outLen), input.dtype)

proc newLPPool2d*(normType: float,
                  kernelSize: int | tuple[h, w: int],
                  stride: int | tuple[h, w: int] = 0,
                  ceilMode: bool = false): LPPool2d =
  ## Create a power-average 2D pooling layer
  result = LPPool2d()
  result.initModule("LPPool2d")
  result.normType = normType
  result.ceilMode = ceilMode

  when kernelSize is int:
    result.kernelSize = (h: kernelSize, w: kernelSize)
  else:
    result.kernelSize = kernelSize

  when stride is int:
    if stride == 0:
      result.stride = result.kernelSize
    else:
      result.stride = (h: stride, w: stride)
  else:
    result.stride = stride

method forward*(pool: LPPool2d, inputs: varargs[TensorRef]): TensorRef =
  if inputs.len == 0:
    raise newException(ModuleError, "LPPool2d forward requires at least one input")

  let input = inputs[0]
  let dims = input.shape.dims

  if dims.len != 4:
    raise newException(ModuleError,
      fmt"LPPool2d expects 4D input (N,C,H,W), got {dims.len}D")

  let outH = calcPoolOutputSize(dims[2], pool.kernelSize.h, pool.stride.h, 0, 1, pool.ceilMode)
  let outW = calcPoolOutputSize(dims[3], pool.kernelSize.w, pool.stride.w, 0, 1, pool.ceilMode)
  newTensorRef(newShape(dims[0], dims[1], outH, outW), input.dtype)

# =============================================================================
# MaxUnpool (for segmentation tasks)
# =============================================================================

type
  MaxUnpool2d* = ref object of Module
    ## 2D max unpooling (inverse of MaxPool2d)
    ## Requires indices from MaxPool2d with returnIndices=true
    kernelSize*: tuple[h, w: int]
    stride*: tuple[h, w: int]
    padding*: tuple[h, w: int]

proc newMaxUnpool2d*(kernelSize: int | tuple[h, w: int],
                     stride: int | tuple[h, w: int] = 0,
                     padding: int | tuple[h, w: int] = 0): MaxUnpool2d =
  ## Create a MaxUnpool2d layer
  result = MaxUnpool2d()
  result.initModule("MaxUnpool2d")

  when kernelSize is int:
    result.kernelSize = (h: kernelSize, w: kernelSize)
  else:
    result.kernelSize = kernelSize

  when stride is int:
    if stride == 0:
      result.stride = result.kernelSize
    else:
      result.stride = (h: stride, w: stride)
  else:
    result.stride = stride

  when padding is int:
    result.padding = (h: padding, w: padding)
  else:
    result.padding = padding

method forward*(pool: MaxUnpool2d, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass requires (input, indices, [output_size])
  if inputs.len < 2:
    raise newException(ModuleError, "MaxUnpool2d forward requires input and indices")

  let input = inputs[0]
  # indices = inputs[1]
  let dims = input.shape.dims

  if dims.len != 4:
    raise newException(ModuleError,
      fmt"MaxUnpool2d expects 4D input (N,C,H,W), got {dims.len}D")

  # Calculate output size (inverse of pooling formula)
  let outH = (dims[2] - 1) * pool.stride.h - 2 * pool.padding.h + pool.kernelSize.h
  let outW = (dims[3] - 1) * pool.stride.w - 2 * pool.padding.w + pool.kernelSize.w

  newTensorRef(newShape(dims[0], dims[1], outH, outW), input.dtype)
