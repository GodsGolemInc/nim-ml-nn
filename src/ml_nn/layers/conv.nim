## Convolutional Layers
##
## Convolution and transposed convolution operations for neural networks.
## Includes Conv1d, Conv2d, Conv3d and their transposed variants.

import std/[options, strformat, math]
import ml_core
import ../module
import ../compute

type
  PaddingMode* = enum
    ## Padding modes for convolution
    pmZeros      ## Zero padding (default)
    pmReflect    ## Reflect padding
    pmReplicate  ## Replicate edge values
    pmCircular   ## Circular/wrap padding

  # =============================================================================
  # 1D Convolution
  # =============================================================================

  Conv1d* = ref object of Module
    ## 1D convolution over input signal (N, C_in, L)
    inChannels*: int
    outChannels*: int
    kernelSize*: int
    stride*: int
    padding*: int
    dilation*: int
    groups*: int
    useBias*: bool
    paddingMode*: PaddingMode
    weight*: Parameter
    bias*: Option[Parameter]

  ConvTranspose1d* = ref object of Module
    ## 1D transposed convolution
    inChannels*: int
    outChannels*: int
    kernelSize*: int
    stride*: int
    padding*: int
    outputPadding*: int
    dilation*: int
    groups*: int
    useBias*: bool
    weight*: Parameter
    bias*: Option[Parameter]

  # =============================================================================
  # 2D Convolution
  # =============================================================================

  Conv2d* = ref object of Module
    ## 2D convolution over input images (N, C_in, H, W)
    inChannels*: int
    outChannels*: int
    kernelSize*: tuple[h, w: int]
    stride*: tuple[h, w: int]
    padding*: tuple[h, w: int]
    dilation*: tuple[h, w: int]
    groups*: int
    useBias*: bool
    paddingMode*: PaddingMode
    weight*: Parameter
    bias*: Option[Parameter]

  ConvTranspose2d* = ref object of Module
    ## 2D transposed convolution (deconvolution)
    inChannels*: int
    outChannels*: int
    kernelSize*: tuple[h, w: int]
    stride*: tuple[h, w: int]
    padding*: tuple[h, w: int]
    outputPadding*: tuple[h, w: int]
    dilation*: tuple[h, w: int]
    groups*: int
    useBias*: bool
    weight*: Parameter
    bias*: Option[Parameter]

  # =============================================================================
  # 3D Convolution
  # =============================================================================

  Conv3d* = ref object of Module
    ## 3D convolution over volumetric data (N, C_in, D, H, W)
    inChannels*: int
    outChannels*: int
    kernelSize*: tuple[d, h, w: int]
    stride*: tuple[d, h, w: int]
    padding*: tuple[d, h, w: int]
    dilation*: tuple[d, h, w: int]
    groups*: int
    useBias*: bool
    paddingMode*: PaddingMode
    weight*: Parameter
    bias*: Option[Parameter]

  ConvTranspose3d* = ref object of Module
    ## 3D transposed convolution
    inChannels*: int
    outChannels*: int
    kernelSize*: tuple[d, h, w: int]
    stride*: tuple[d, h, w: int]
    padding*: tuple[d, h, w: int]
    outputPadding*: tuple[d, h, w: int]
    dilation*: tuple[d, h, w: int]
    groups*: int
    useBias*: bool
    weight*: Parameter
    bias*: Option[Parameter]

# =============================================================================
# Conv1d Implementation
# =============================================================================

proc newConv1d*(inChannels, outChannels: int,
                kernelSize: int,
                stride: int = 1,
                padding: int = 0,
                dilation: int = 1,
                groups: int = 1,
                useBias: bool = true,
                paddingMode: PaddingMode = pmZeros,
                dtype: DType = dtFloat32): Conv1d =
  ## Create a Conv1d layer
  ##
  ## Args:
  ##   inChannels: Number of input channels
  ##   outChannels: Number of output channels (filters)
  ##   kernelSize: Size of the convolving kernel
  ##   stride: Stride of the convolution
  ##   padding: Zero-padding added to both sides
  ##   dilation: Spacing between kernel elements
  ##   groups: Number of blocked connections from input to output
  ##   useBias: Whether to add a learnable bias
  ##   paddingMode: Padding mode
  assert inChannels mod groups == 0, "inChannels must be divisible by groups"
  assert outChannels mod groups == 0, "outChannels must be divisible by groups"

  result = Conv1d()
  result.initModule("Conv1d")
  result.inChannels = inChannels
  result.outChannels = outChannels
  result.kernelSize = kernelSize
  result.stride = stride
  result.padding = padding
  result.dilation = dilation
  result.groups = groups
  result.useBias = useBias
  result.paddingMode = paddingMode

  # Weight shape: (outChannels, inChannels/groups, kernelSize)
  result.weight = result.registerParameter(
    "weight",
    newShape(outChannels, inChannels div groups, kernelSize),
    dtype
  )

  if useBias:
    let biasParam = result.registerParameter("bias", newShape(outChannels), dtype)
    result.bias = some(biasParam)
  else:
    result.bias = none(Parameter)

proc calcConv1dOutputSize(inputLen: int, kernelSize, stride, padding, dilation: int): int =
  ## Calculate output length for 1D convolution
  (inputLen + 2 * padding - dilation * (kernelSize - 1) - 1) div stride + 1

method forward*(conv: Conv1d, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass for Conv1d
  if inputs.len == 0:
    raise newException(ModuleError, "Conv1d forward requires at least one input")

  let input = inputs[0]
  let dims = input.shape.dims

  # Expect (N, C, L) or (C, L)
  if dims.len < 2 or dims.len > 3:
    raise newException(ModuleError,
      fmt"Conv1d expects 2D or 3D input, got {dims.len}D")

  let hasBatch = dims.len == 3
  let channels = if hasBatch: dims[1] else: dims[0]

  if channels != conv.inChannels:
    raise newException(ModuleError,
      fmt"Conv1d input channels mismatch: expected {conv.inChannels}, got {channels}")

  let biasData = if conv.bias.isSome: conv.bias.get.data else: nil
  computeConv1d(input, conv.weight.data, biasData,
                conv.stride, conv.padding, conv.dilation, conv.groups)

# =============================================================================
# Conv2d Implementation
# =============================================================================

proc newConv2d*(inChannels, outChannels: int,
                kernelSize: int | tuple[h, w: int],
                stride: int | tuple[h, w: int] = 1,
                padding: int | tuple[h, w: int] = 0,
                dilation: int | tuple[h, w: int] = 1,
                groups: int = 1,
                useBias: bool = true,
                paddingMode: PaddingMode = pmZeros,
                dtype: DType = dtFloat32): Conv2d =
  ## Create a Conv2d layer
  ##
  ## Args:
  ##   inChannels: Number of input channels
  ##   outChannels: Number of output channels (filters)
  ##   kernelSize: Size of the convolving kernel (H, W)
  ##   stride: Stride of the convolution
  ##   padding: Zero-padding added to both sides
  ##   dilation: Spacing between kernel elements
  ##   groups: Number of blocked connections
  ##   useBias: Whether to add a learnable bias
  ##   paddingMode: Padding mode
  assert inChannels mod groups == 0, "inChannels must be divisible by groups"
  assert outChannels mod groups == 0, "outChannels must be divisible by groups"

  result = Conv2d()
  result.initModule("Conv2d")
  result.inChannels = inChannels
  result.outChannels = outChannels
  result.groups = groups
  result.useBias = useBias
  result.paddingMode = paddingMode

  # Handle int or tuple arguments
  when kernelSize is int:
    result.kernelSize = (h: kernelSize, w: kernelSize)
  else:
    result.kernelSize = kernelSize

  when stride is int:
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

  # Weight shape: (outChannels, inChannels/groups, kH, kW)
  result.weight = result.registerParameter(
    "weight",
    newShape(outChannels, inChannels div groups,
             result.kernelSize.h, result.kernelSize.w),
    dtype
  )

  if useBias:
    let biasParam = result.registerParameter("bias", newShape(outChannels), dtype)
    result.bias = some(biasParam)
  else:
    result.bias = none(Parameter)

proc calcConv2dOutputSize(inputH, inputW: int,
                          kernelH, kernelW: int,
                          strideH, strideW: int,
                          paddingH, paddingW: int,
                          dilationH, dilationW: int): tuple[h, w: int] =
  ## Calculate output dimensions for 2D convolution
  let outH = (inputH + 2 * paddingH - dilationH * (kernelH - 1) - 1) div strideH + 1
  let outW = (inputW + 2 * paddingW - dilationW * (kernelW - 1) - 1) div strideW + 1
  (h: outH, w: outW)

method forward*(conv: Conv2d, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass for Conv2d
  ##
  ## Input shape: (N, C_in, H, W)
  ## Output shape: (N, C_out, H_out, W_out)
  if inputs.len == 0:
    raise newException(ModuleError, "Conv2d forward requires at least one input")

  let input = inputs[0]
  let dims = input.shape.dims

  # Expect (N, C, H, W)
  if dims.len != 4:
    raise newException(ModuleError,
      fmt"Conv2d expects 4D input (N,C,H,W), got {dims.len}D")

  let channels = dims[1]

  if channels != conv.inChannels:
    raise newException(ModuleError,
      fmt"Conv2d input channels mismatch: expected {conv.inChannels}, got {channels}")

  let biasData = if conv.bias.isSome: conv.bias.get.data else: nil
  computeConv2d(input, conv.weight.data, biasData,
                conv.stride.h, conv.stride.w,
                conv.padding.h, conv.padding.w,
                conv.dilation.h, conv.dilation.w,
                conv.groups)

# =============================================================================
# ConvTranspose2d Implementation
# =============================================================================

proc newConvTranspose2d*(inChannels, outChannels: int,
                         kernelSize: int | tuple[h, w: int],
                         stride: int | tuple[h, w: int] = 1,
                         padding: int | tuple[h, w: int] = 0,
                         outputPadding: int | tuple[h, w: int] = 0,
                         dilation: int | tuple[h, w: int] = 1,
                         groups: int = 1,
                         useBias: bool = true,
                         dtype: DType = dtFloat32): ConvTranspose2d =
  ## Create a ConvTranspose2d layer (transposed/fractionally-strided convolution)
  ##
  ## Also known as "deconvolution", though it's not a true deconvolution.
  ## Used for upsampling in autoencoders, GANs, etc.
  assert inChannels mod groups == 0, "inChannels must be divisible by groups"
  assert outChannels mod groups == 0, "outChannels must be divisible by groups"

  result = ConvTranspose2d()
  result.initModule("ConvTranspose2d")
  result.inChannels = inChannels
  result.outChannels = outChannels
  result.groups = groups
  result.useBias = useBias

  when kernelSize is int:
    result.kernelSize = (h: kernelSize, w: kernelSize)
  else:
    result.kernelSize = kernelSize

  when stride is int:
    result.stride = (h: stride, w: stride)
  else:
    result.stride = stride

  when padding is int:
    result.padding = (h: padding, w: padding)
  else:
    result.padding = padding

  when outputPadding is int:
    result.outputPadding = (h: outputPadding, w: outputPadding)
  else:
    result.outputPadding = outputPadding

  when dilation is int:
    result.dilation = (h: dilation, w: dilation)
  else:
    result.dilation = dilation

  # Weight shape: (inChannels, outChannels/groups, kH, kW)
  # Note: transposed from Conv2d
  result.weight = result.registerParameter(
    "weight",
    newShape(inChannels, outChannels div groups,
             result.kernelSize.h, result.kernelSize.w),
    dtype
  )

  if useBias:
    let biasParam = result.registerParameter("bias", newShape(outChannels), dtype)
    result.bias = some(biasParam)
  else:
    result.bias = none(Parameter)

proc calcConvTranspose2dOutputSize(inputH, inputW: int,
                                    kernelH, kernelW: int,
                                    strideH, strideW: int,
                                    paddingH, paddingW: int,
                                    outputPaddingH, outputPaddingW: int,
                                    dilationH, dilationW: int): tuple[h, w: int] =
  ## Calculate output dimensions for transposed 2D convolution
  let outH = (inputH - 1) * strideH - 2 * paddingH +
             dilationH * (kernelH - 1) + outputPaddingH + 1
  let outW = (inputW - 1) * strideW - 2 * paddingW +
             dilationW * (kernelW - 1) + outputPaddingW + 1
  (h: outH, w: outW)

method forward*(conv: ConvTranspose2d, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass for ConvTranspose2d
  if inputs.len == 0:
    raise newException(ModuleError, "ConvTranspose2d forward requires at least one input")

  let input = inputs[0]
  let dims = input.shape.dims

  if dims.len != 4:
    raise newException(ModuleError,
      fmt"ConvTranspose2d expects 4D input (N,C,H,W), got {dims.len}D")

  let batchSize = dims[0]
  let channels = dims[1]
  let inH = dims[2]
  let inW = dims[3]

  if channels != conv.inChannels:
    raise newException(ModuleError,
      fmt"ConvTranspose2d input channels mismatch: expected {conv.inChannels}, got {channels}")

  let (outH, outW) = calcConvTranspose2dOutputSize(
    inH, inW,
    conv.kernelSize.h, conv.kernelSize.w,
    conv.stride.h, conv.stride.w,
    conv.padding.h, conv.padding.w,
    conv.outputPadding.h, conv.outputPadding.w,
    conv.dilation.h, conv.dilation.w
  )

  let outShape = newShape(batchSize, conv.outChannels, outH, outW)
  newTensorRef(outShape, input.dtype)

# =============================================================================
# Conv3d Implementation
# =============================================================================

proc newConv3d*(inChannels, outChannels: int,
                kernelSize: int | tuple[d, h, w: int],
                stride: int | tuple[d, h, w: int] = 1,
                padding: int | tuple[d, h, w: int] = 0,
                dilation: int | tuple[d, h, w: int] = 1,
                groups: int = 1,
                useBias: bool = true,
                paddingMode: PaddingMode = pmZeros,
                dtype: DType = dtFloat32): Conv3d =
  ## Create a Conv3d layer for volumetric data
  assert inChannels mod groups == 0, "inChannels must be divisible by groups"
  assert outChannels mod groups == 0, "outChannels must be divisible by groups"

  result = Conv3d()
  result.initModule("Conv3d")
  result.inChannels = inChannels
  result.outChannels = outChannels
  result.groups = groups
  result.useBias = useBias
  result.paddingMode = paddingMode

  when kernelSize is int:
    result.kernelSize = (d: kernelSize, h: kernelSize, w: kernelSize)
  else:
    result.kernelSize = kernelSize

  when stride is int:
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

  # Weight shape: (outChannels, inChannels/groups, kD, kH, kW)
  result.weight = result.registerParameter(
    "weight",
    newShape(outChannels, inChannels div groups,
             result.kernelSize.d, result.kernelSize.h, result.kernelSize.w),
    dtype
  )

  if useBias:
    let biasParam = result.registerParameter("bias", newShape(outChannels), dtype)
    result.bias = some(biasParam)
  else:
    result.bias = none(Parameter)

method forward*(conv: Conv3d, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass for Conv3d
  if inputs.len == 0:
    raise newException(ModuleError, "Conv3d forward requires at least one input")

  let input = inputs[0]
  let dims = input.shape.dims

  if dims.len != 5:
    raise newException(ModuleError,
      fmt"Conv3d expects 5D input (N,C,D,H,W), got {dims.len}D")

  let batchSize = dims[0]
  let channels = dims[1]
  let inD = dims[2]
  let inH = dims[3]
  let inW = dims[4]

  if channels != conv.inChannels:
    raise newException(ModuleError,
      fmt"Conv3d input channels mismatch: expected {conv.inChannels}, got {channels}")

  # Calculate output size for each dimension
  let outD = (inD + 2 * conv.padding.d - conv.dilation.d * (conv.kernelSize.d - 1) - 1) div conv.stride.d + 1
  let outH = (inH + 2 * conv.padding.h - conv.dilation.h * (conv.kernelSize.h - 1) - 1) div conv.stride.h + 1
  let outW = (inW + 2 * conv.padding.w - conv.dilation.w * (conv.kernelSize.w - 1) - 1) div conv.stride.w + 1

  let outShape = newShape(batchSize, conv.outChannels, outD, outH, outW)
  newTensorRef(outShape, input.dtype)

# =============================================================================
# Depthwise Separable Convolution
# =============================================================================

type
  DepthwiseSeparableConv2d* = ref object of Module
    ## Depthwise separable 2D convolution
    ## Used in MobileNet, EfficientNet, etc.
    ## Consists of depthwise + pointwise convolutions
    inChannels*: int
    outChannels*: int
    depthwiseConv*: Conv2d
    pointwiseConv*: Conv2d

proc newDepthwiseSeparableConv2d*(inChannels, outChannels: int,
                                   kernelSize: int,
                                   stride: int = 1,
                                   padding: int = 0,
                                   useBias: bool = true,
                                   dtype: DType = dtFloat32): DepthwiseSeparableConv2d =
  ## Create a depthwise separable 2D convolution
  ##
  ## More efficient than standard convolution:
  ##   Standard: inChannels * outChannels * kH * kW
  ##   Depthwise separable: inChannels * kH * kW + inChannels * outChannels
  result = DepthwiseSeparableConv2d()
  result.initModule("DepthwiseSeparableConv2d")
  result.inChannels = inChannels
  result.outChannels = outChannels

  # Depthwise: groups = inChannels (one filter per input channel)
  result.depthwiseConv = newConv2d(
    inChannels, inChannels,
    kernelSize = kernelSize,
    stride = stride,
    padding = padding,
    groups = inChannels,
    useBias = false,
    dtype = dtype
  )
  result.registerModule("depthwise", result.depthwiseConv)

  # Pointwise: 1x1 convolution to combine channels
  result.pointwiseConv = newConv2d(
    inChannels, outChannels,
    kernelSize = 1,
    useBias = useBias,
    dtype = dtype
  )
  result.registerModule("pointwise", result.pointwiseConv)

method forward*(dsc: DepthwiseSeparableConv2d, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass for depthwise separable convolution
  let depthwise = dsc.depthwiseConv.forward(inputs)
  dsc.pointwiseConv.forward(depthwise)

# =============================================================================
# Dilated (Atrous) Convolution Helper
# =============================================================================

proc newDilatedConv2d*(inChannels, outChannels: int,
                       kernelSize: int,
                       dilation: int,
                       padding: int = -1,
                       useBias: bool = true,
                       dtype: DType = dtFloat32): Conv2d =
  ## Create a dilated (atrous) convolution
  ##
  ## If padding is -1, calculate "same" padding for the dilation
  var actualPadding = padding
  if actualPadding == -1:
    # Calculate padding for "same" output size with dilation
    actualPadding = (kernelSize - 1) * dilation div 2

  newConv2d(
    inChannels, outChannels,
    kernelSize = kernelSize,
    dilation = dilation,
    padding = actualPadding,
    useBias = useBias,
    dtype = dtype
  )
