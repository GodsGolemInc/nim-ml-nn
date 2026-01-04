## Normalization Layers
##
## Normalization operations for neural networks.
## Includes BatchNorm, LayerNorm, GroupNorm, InstanceNorm, and RMSNorm.

import std/[options, strformat]
import ml_core
import ../module

type
  # =============================================================================
  # Batch Normalization
  # =============================================================================

  BatchNorm1d* = ref object of Module
    ## Batch normalization over 2D or 3D input
    ## Normalizes over the batch dimension
    numFeatures*: int
    eps*: float
    momentum*: float
    affine*: bool
    trackRunningStats*: bool
    weight*: Option[Parameter]  # gamma
    bias*: Option[Parameter]    # beta
    runningMean*: Option[TensorData]
    runningVar*: Option[TensorData]
    numBatchesTracked*: int

  BatchNorm2d* = ref object of Module
    ## Batch normalization over 4D input (N, C, H, W)
    numFeatures*: int
    eps*: float
    momentum*: float
    affine*: bool
    trackRunningStats*: bool
    weight*: Option[Parameter]
    bias*: Option[Parameter]
    runningMean*: Option[TensorData]
    runningVar*: Option[TensorData]
    numBatchesTracked*: int

  BatchNorm3d* = ref object of Module
    ## Batch normalization over 5D input (N, C, D, H, W)
    numFeatures*: int
    eps*: float
    momentum*: float
    affine*: bool
    trackRunningStats*: bool
    weight*: Option[Parameter]
    bias*: Option[Parameter]
    runningMean*: Option[TensorData]
    runningVar*: Option[TensorData]
    numBatchesTracked*: int

  # =============================================================================
  # Layer Normalization
  # =============================================================================

  LayerNorm* = ref object of Module
    ## Layer normalization
    ## Normalizes over the last D dimensions
    normalizedShape*: seq[int]
    eps*: float
    elementwiseAffine*: bool
    weight*: Option[Parameter]  # gamma
    bias*: Option[Parameter]    # beta

  # =============================================================================
  # Group Normalization
  # =============================================================================

  GroupNorm* = ref object of Module
    ## Group normalization
    ## Divides channels into groups and normalizes within each group
    numGroups*: int
    numChannels*: int
    eps*: float
    affine*: bool
    weight*: Option[Parameter]
    bias*: Option[Parameter]

  # =============================================================================
  # Instance Normalization
  # =============================================================================

  InstanceNorm1d* = ref object of Module
    ## Instance normalization over 3D input
    numFeatures*: int
    eps*: float
    momentum*: float
    affine*: bool
    trackRunningStats*: bool
    weight*: Option[Parameter]
    bias*: Option[Parameter]
    runningMean*: Option[TensorData]
    runningVar*: Option[TensorData]

  InstanceNorm2d* = ref object of Module
    ## Instance normalization over 4D input
    numFeatures*: int
    eps*: float
    momentum*: float
    affine*: bool
    trackRunningStats*: bool
    weight*: Option[Parameter]
    bias*: Option[Parameter]
    runningMean*: Option[TensorData]
    runningVar*: Option[TensorData]

  # =============================================================================
  # RMS Normalization
  # =============================================================================

  RMSNorm* = ref object of Module
    ## Root Mean Square Layer Normalization
    ## Used in LLaMA, T5, etc.
    normalizedShape*: seq[int]
    eps*: float
    weight*: Option[Parameter]  # Only scale, no bias

# =============================================================================
# BatchNorm1d Implementation
# =============================================================================

proc newBatchNorm1d*(numFeatures: int,
                     eps: float = 1e-5,
                     momentum: float = 0.1,
                     affine: bool = true,
                     trackRunningStats: bool = true,
                     dtype: DType = dtFloat32): BatchNorm1d =
  ## Create a BatchNorm1d layer
  ##
  ## Args:
  ##   numFeatures: Number of features/channels
  ##   eps: Small constant for numerical stability
  ##   momentum: Momentum for running stats update
  ##   affine: Whether to learn scale and shift
  ##   trackRunningStats: Whether to track running mean/var
  result = BatchNorm1d()
  result.initModule("BatchNorm1d")
  result.numFeatures = numFeatures
  result.eps = eps
  result.momentum = momentum
  result.affine = affine
  result.trackRunningStats = trackRunningStats
  result.numBatchesTracked = 0

  if affine:
    let weightParam = result.registerParameter("weight", newShape(numFeatures), dtype)
    result.weight = some(weightParam)
    let biasParam = result.registerParameter("bias", newShape(numFeatures), dtype)
    result.bias = some(biasParam)
  else:
    result.weight = none(Parameter)
    result.bias = none(Parameter)

  if trackRunningStats:
    result.runningMean = some(newTensorData(newShape(numFeatures), dtype))
    result.runningVar = some(newTensorData(newShape(numFeatures), dtype))
  else:
    result.runningMean = none(TensorData)
    result.runningVar = none(TensorData)

method forward*(bn: BatchNorm1d, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass for BatchNorm1d
  if inputs.len == 0:
    raise newException(ModuleError, "BatchNorm1d forward requires at least one input")

  let input = inputs[0]
  let dims = input.shape.dims

  # Expect (N, C) or (N, C, L)
  if dims.len < 2 or dims.len > 3:
    raise newException(ModuleError,
      fmt"BatchNorm1d expects 2D or 3D input, got {dims.len}D")

  if dims[1] != bn.numFeatures:
    raise newException(ModuleError,
      fmt"BatchNorm1d feature size mismatch: expected {bn.numFeatures}, got {dims[1]}")

  # Output has same shape as input
  newTensorRef(input.shape, input.dtype)

# =============================================================================
# BatchNorm2d Implementation
# =============================================================================

proc newBatchNorm2d*(numFeatures: int,
                     eps: float = 1e-5,
                     momentum: float = 0.1,
                     affine: bool = true,
                     trackRunningStats: bool = true,
                     dtype: DType = dtFloat32): BatchNorm2d =
  ## Create a BatchNorm2d layer
  ##
  ## Args:
  ##   numFeatures: Number of channels (C in N,C,H,W)
  ##   eps: Small constant for numerical stability
  ##   momentum: Momentum for running stats update
  ##   affine: Whether to learn scale and shift
  ##   trackRunningStats: Whether to track running mean/var
  result = BatchNorm2d()
  result.initModule("BatchNorm2d")
  result.numFeatures = numFeatures
  result.eps = eps
  result.momentum = momentum
  result.affine = affine
  result.trackRunningStats = trackRunningStats
  result.numBatchesTracked = 0

  if affine:
    let weightParam = result.registerParameter("weight", newShape(numFeatures), dtype)
    result.weight = some(weightParam)
    let biasParam = result.registerParameter("bias", newShape(numFeatures), dtype)
    result.bias = some(biasParam)
  else:
    result.weight = none(Parameter)
    result.bias = none(Parameter)

  if trackRunningStats:
    result.runningMean = some(newTensorData(newShape(numFeatures), dtype))
    result.runningVar = some(newTensorData(newShape(numFeatures), dtype))
  else:
    result.runningMean = none(TensorData)
    result.runningVar = none(TensorData)

method forward*(bn: BatchNorm2d, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass for BatchNorm2d
  if inputs.len == 0:
    raise newException(ModuleError, "BatchNorm2d forward requires at least one input")

  let input = inputs[0]
  let dims = input.shape.dims

  # Expect (N, C, H, W)
  if dims.len != 4:
    raise newException(ModuleError,
      fmt"BatchNorm2d expects 4D input (N,C,H,W), got {dims.len}D")

  if dims[1] != bn.numFeatures:
    raise newException(ModuleError,
      fmt"BatchNorm2d feature size mismatch: expected {bn.numFeatures}, got {dims[1]}")

  # Output has same shape as input
  newTensorRef(input.shape, input.dtype)

# =============================================================================
# BatchNorm3d Implementation
# =============================================================================

proc newBatchNorm3d*(numFeatures: int,
                     eps: float = 1e-5,
                     momentum: float = 0.1,
                     affine: bool = true,
                     trackRunningStats: bool = true,
                     dtype: DType = dtFloat32): BatchNorm3d =
  ## Create a BatchNorm3d layer
  result = BatchNorm3d()
  result.initModule("BatchNorm3d")
  result.numFeatures = numFeatures
  result.eps = eps
  result.momentum = momentum
  result.affine = affine
  result.trackRunningStats = trackRunningStats
  result.numBatchesTracked = 0

  if affine:
    let weightParam = result.registerParameter("weight", newShape(numFeatures), dtype)
    result.weight = some(weightParam)
    let biasParam = result.registerParameter("bias", newShape(numFeatures), dtype)
    result.bias = some(biasParam)
  else:
    result.weight = none(Parameter)
    result.bias = none(Parameter)

  if trackRunningStats:
    result.runningMean = some(newTensorData(newShape(numFeatures), dtype))
    result.runningVar = some(newTensorData(newShape(numFeatures), dtype))
  else:
    result.runningMean = none(TensorData)
    result.runningVar = none(TensorData)

method forward*(bn: BatchNorm3d, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass for BatchNorm3d
  if inputs.len == 0:
    raise newException(ModuleError, "BatchNorm3d forward requires at least one input")

  let input = inputs[0]
  let dims = input.shape.dims

  # Expect (N, C, D, H, W)
  if dims.len != 5:
    raise newException(ModuleError,
      fmt"BatchNorm3d expects 5D input (N,C,D,H,W), got {dims.len}D")

  if dims[1] != bn.numFeatures:
    raise newException(ModuleError,
      fmt"BatchNorm3d feature size mismatch: expected {bn.numFeatures}, got {dims[1]}")

  newTensorRef(input.shape, input.dtype)

# =============================================================================
# LayerNorm Implementation
# =============================================================================

proc newLayerNorm*(normalizedShape: seq[int],
                   eps: float = 1e-5,
                   elementwiseAffine: bool = true,
                   dtype: DType = dtFloat32): LayerNorm =
  ## Create a LayerNorm layer
  ##
  ## Args:
  ##   normalizedShape: Shape of the dimensions to normalize over
  ##   eps: Small constant for numerical stability
  ##   elementwiseAffine: Whether to learn scale and shift
  ##
  ## Example:
  ##   LayerNorm([768]) for transformer hidden states
  ##   LayerNorm([512, 512]) for image features
  result = LayerNorm()
  result.initModule("LayerNorm")
  result.normalizedShape = normalizedShape
  result.eps = eps
  result.elementwiseAffine = elementwiseAffine

  if elementwiseAffine:
    let weightParam = result.registerParameter("weight", newShape(normalizedShape), dtype)
    result.weight = some(weightParam)
    let biasParam = result.registerParameter("bias", newShape(normalizedShape), dtype)
    result.bias = some(biasParam)
  else:
    result.weight = none(Parameter)
    result.bias = none(Parameter)

proc newLayerNorm*(normalizedSize: int,
                   eps: float = 1e-5,
                   elementwiseAffine: bool = true,
                   dtype: DType = dtFloat32): LayerNorm =
  ## Convenience constructor for 1D normalization
  newLayerNorm(@[normalizedSize], eps, elementwiseAffine, dtype)

method forward*(ln: LayerNorm, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass for LayerNorm
  if inputs.len == 0:
    raise newException(ModuleError, "LayerNorm forward requires at least one input")

  let input = inputs[0]
  let dims = input.shape.dims
  let normDims = ln.normalizedShape.len

  # Input must have at least as many dimensions as normalized shape
  if dims.len < normDims:
    raise newException(ModuleError,
      fmt"LayerNorm input has fewer dims ({dims.len}) than normalized shape ({normDims})")

  # Check that trailing dimensions match normalized shape
  for i in 0..<normDims:
    let inputDim = dims[dims.len - normDims + i]
    let normDim = ln.normalizedShape[i]
    if inputDim != normDim:
      raise newException(ModuleError,
        fmt"LayerNorm shape mismatch at dim {i}: expected {normDim}, got {inputDim}")

  # Output has same shape as input
  newTensorRef(input.shape, input.dtype)

# =============================================================================
# GroupNorm Implementation
# =============================================================================

proc newGroupNorm*(numGroups: int,
                   numChannels: int,
                   eps: float = 1e-5,
                   affine: bool = true,
                   dtype: DType = dtFloat32): GroupNorm =
  ## Create a GroupNorm layer
  ##
  ## Args:
  ##   numGroups: Number of groups to divide channels into
  ##   numChannels: Total number of channels
  ##   eps: Small constant for numerical stability
  ##   affine: Whether to learn scale and shift
  ##
  ## Note: numChannels must be divisible by numGroups
  if numChannels mod numGroups != 0:
    raise newException(ModuleError,
      fmt"numChannels ({numChannels}) must be divisible by numGroups ({numGroups})")

  result = GroupNorm()
  result.initModule("GroupNorm")
  result.numGroups = numGroups
  result.numChannels = numChannels
  result.eps = eps
  result.affine = affine

  if affine:
    let weightParam = result.registerParameter("weight", newShape(numChannels), dtype)
    result.weight = some(weightParam)
    let biasParam = result.registerParameter("bias", newShape(numChannels), dtype)
    result.bias = some(biasParam)
  else:
    result.weight = none(Parameter)
    result.bias = none(Parameter)

method forward*(gn: GroupNorm, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass for GroupNorm
  if inputs.len == 0:
    raise newException(ModuleError, "GroupNorm forward requires at least one input")

  let input = inputs[0]
  let dims = input.shape.dims

  # Expect at least 2D input (N, C, ...)
  if dims.len < 2:
    raise newException(ModuleError,
      fmt"GroupNorm expects at least 2D input, got {dims.len}D")

  if dims[1] != gn.numChannels:
    raise newException(ModuleError,
      fmt"GroupNorm channel mismatch: expected {gn.numChannels}, got {dims[1]}")

  # Output has same shape as input
  newTensorRef(input.shape, input.dtype)

# =============================================================================
# InstanceNorm1d Implementation
# =============================================================================

proc newInstanceNorm1d*(numFeatures: int,
                        eps: float = 1e-5,
                        momentum: float = 0.1,
                        affine: bool = false,
                        trackRunningStats: bool = false,
                        dtype: DType = dtFloat32): InstanceNorm1d =
  ## Create an InstanceNorm1d layer
  ##
  ## Instance normalization normalizes over spatial dimensions per instance
  result = InstanceNorm1d()
  result.initModule("InstanceNorm1d")
  result.numFeatures = numFeatures
  result.eps = eps
  result.momentum = momentum
  result.affine = affine
  result.trackRunningStats = trackRunningStats

  if affine:
    let weightParam = result.registerParameter("weight", newShape(numFeatures), dtype)
    result.weight = some(weightParam)
    let biasParam = result.registerParameter("bias", newShape(numFeatures), dtype)
    result.bias = some(biasParam)
  else:
    result.weight = none(Parameter)
    result.bias = none(Parameter)

  if trackRunningStats:
    result.runningMean = some(newTensorData(newShape(numFeatures), dtype))
    result.runningVar = some(newTensorData(newShape(numFeatures), dtype))
  else:
    result.runningMean = none(TensorData)
    result.runningVar = none(TensorData)

method forward*(inst: InstanceNorm1d, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass for InstanceNorm1d
  if inputs.len == 0:
    raise newException(ModuleError, "InstanceNorm1d forward requires at least one input")

  let input = inputs[0]
  let dims = input.shape.dims

  if dims.len != 3:
    raise newException(ModuleError,
      fmt"InstanceNorm1d expects 3D input (N,C,L), got {dims.len}D")

  if dims[1] != inst.numFeatures:
    raise newException(ModuleError,
      fmt"InstanceNorm1d feature mismatch: expected {inst.numFeatures}, got {dims[1]}")

  newTensorRef(input.shape, input.dtype)

# =============================================================================
# InstanceNorm2d Implementation
# =============================================================================

proc newInstanceNorm2d*(numFeatures: int,
                        eps: float = 1e-5,
                        momentum: float = 0.1,
                        affine: bool = false,
                        trackRunningStats: bool = false,
                        dtype: DType = dtFloat32): InstanceNorm2d =
  ## Create an InstanceNorm2d layer
  result = InstanceNorm2d()
  result.initModule("InstanceNorm2d")
  result.numFeatures = numFeatures
  result.eps = eps
  result.momentum = momentum
  result.affine = affine
  result.trackRunningStats = trackRunningStats

  if affine:
    let weightParam = result.registerParameter("weight", newShape(numFeatures), dtype)
    result.weight = some(weightParam)
    let biasParam = result.registerParameter("bias", newShape(numFeatures), dtype)
    result.bias = some(biasParam)
  else:
    result.weight = none(Parameter)
    result.bias = none(Parameter)

  if trackRunningStats:
    result.runningMean = some(newTensorData(newShape(numFeatures), dtype))
    result.runningVar = some(newTensorData(newShape(numFeatures), dtype))
  else:
    result.runningMean = none(TensorData)
    result.runningVar = none(TensorData)

method forward*(inst: InstanceNorm2d, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass for InstanceNorm2d
  if inputs.len == 0:
    raise newException(ModuleError, "InstanceNorm2d forward requires at least one input")

  let input = inputs[0]
  let dims = input.shape.dims

  if dims.len != 4:
    raise newException(ModuleError,
      fmt"InstanceNorm2d expects 4D input (N,C,H,W), got {dims.len}D")

  if dims[1] != inst.numFeatures:
    raise newException(ModuleError,
      fmt"InstanceNorm2d feature mismatch: expected {inst.numFeatures}, got {dims[1]}")

  newTensorRef(input.shape, input.dtype)

# =============================================================================
# RMSNorm Implementation
# =============================================================================

proc newRMSNorm*(normalizedShape: seq[int],
                 eps: float = 1e-6,
                 dtype: DType = dtFloat32): RMSNorm =
  ## Create an RMSNorm layer
  ##
  ## RMSNorm is simpler than LayerNorm:
  ##   RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
  ##
  ## Used in LLaMA, T5, and other modern architectures.
  ## Does not subtract mean or have a bias term.
  result = RMSNorm()
  result.initModule("RMSNorm")
  result.normalizedShape = normalizedShape
  result.eps = eps

  let weightParam = result.registerParameter("weight", newShape(normalizedShape), dtype)
  result.weight = some(weightParam)

proc newRMSNorm*(normalizedSize: int,
                 eps: float = 1e-6,
                 dtype: DType = dtFloat32): RMSNorm =
  ## Convenience constructor for 1D normalization
  newRMSNorm(@[normalizedSize], eps, dtype)

method forward*(rms: RMSNorm, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass for RMSNorm
  ##
  ## Computes: x / sqrt(mean(x^2) + eps) * weight
  if inputs.len == 0:
    raise newException(ModuleError, "RMSNorm forward requires at least one input")

  let input = inputs[0]
  let dims = input.shape.dims
  let normDims = rms.normalizedShape.len

  # Input must have at least as many dimensions as normalized shape
  if dims.len < normDims:
    raise newException(ModuleError,
      fmt"RMSNorm input has fewer dims ({dims.len}) than normalized shape ({normDims})")

  # Check that trailing dimensions match normalized shape
  for i in 0..<normDims:
    let inputDim = dims[dims.len - normDims + i]
    let normDim = rms.normalizedShape[i]
    if inputDim != normDim:
      raise newException(ModuleError,
        fmt"RMSNorm shape mismatch at dim {i}: expected {normDim}, got {inputDim}")

  # Output has same shape as input
  newTensorRef(input.shape, input.dtype)

# =============================================================================
# LocalResponseNorm (LRN)
# =============================================================================

type
  LocalResponseNorm* = ref object of Module
    ## Local Response Normalization
    ## Used in AlexNet, etc.
    size*: int
    alpha*: float
    beta*: float
    k*: float

proc newLocalResponseNorm*(size: int,
                           alpha: float = 0.0001,
                           beta: float = 0.75,
                           k: float = 1.0): LocalResponseNorm =
  ## Create a LocalResponseNorm layer
  ##
  ## Args:
  ##   size: Amount of neighbouring channels used for normalization
  ##   alpha: Multiplicative factor
  ##   beta: Exponent
  ##   k: Additive factor
  result = LocalResponseNorm()
  result.initModule("LocalResponseNorm")
  result.size = size
  result.alpha = alpha
  result.beta = beta
  result.k = k

method forward*(lrn: LocalResponseNorm, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass for LocalResponseNorm
  if inputs.len == 0:
    raise newException(ModuleError, "LocalResponseNorm forward requires at least one input")

  let input = inputs[0]
  newTensorRef(input.shape, input.dtype)

# =============================================================================
# SyncBatchNorm (for distributed training)
# =============================================================================

type
  SyncBatchNorm* = ref object of BatchNorm2d
    ## Synchronized Batch Normalization across processes
    ## Uses all processes to compute mean and variance
    processGroup*: string  # Process group identifier

proc newSyncBatchNorm*(numFeatures: int,
                       eps: float = 1e-5,
                       momentum: float = 0.1,
                       affine: bool = true,
                       trackRunningStats: bool = true,
                       processGroup: string = "default",
                       dtype: DType = dtFloat32): SyncBatchNorm =
  ## Create a SyncBatchNorm layer
  result = SyncBatchNorm()
  result.initModule("SyncBatchNorm")
  result.numFeatures = numFeatures
  result.eps = eps
  result.momentum = momentum
  result.affine = affine
  result.trackRunningStats = trackRunningStats
  result.numBatchesTracked = 0
  result.processGroup = processGroup

  if affine:
    let weightParam = result.registerParameter("weight", newShape(numFeatures), dtype)
    result.weight = some(weightParam)
    let biasParam = result.registerParameter("bias", newShape(numFeatures), dtype)
    result.bias = some(biasParam)
  else:
    result.weight = none(Parameter)
    result.bias = none(Parameter)

  if trackRunningStats:
    result.runningMean = some(newTensorData(newShape(numFeatures), dtype))
    result.runningVar = some(newTensorData(newShape(numFeatures), dtype))
  else:
    result.runningMean = none(TensorData)
    result.runningVar = none(TensorData)

method forward*(sbn: SyncBatchNorm, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass for SyncBatchNorm
  if inputs.len == 0:
    raise newException(ModuleError, "SyncBatchNorm forward requires at least one input")

  let input = inputs[0]
  let dims = input.shape.dims

  if dims.len != 4:
    raise newException(ModuleError,
      fmt"SyncBatchNorm expects 4D input (N,C,H,W), got {dims.len}D")

  if dims[1] != sbn.numFeatures:
    raise newException(ModuleError,
      fmt"SyncBatchNorm feature size mismatch: expected {sbn.numFeatures}, got {dims[1]}")

  newTensorRef(input.shape, input.dtype)
