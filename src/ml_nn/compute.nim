## Computation Bridge
##
## Bridges nn.Module layers with actual tensor computations.
## Provides functions that:
##   1. Get TensorData from TensorRef
##   2. Call kernels for computation
##   3. Return new TensorRef with computed data

import std/[tables, options]
import ml_core
import ml_core/kernels

# Global tensor store for computed results
var tensorDataStore* {.global.}: Table[Hash256, TensorData] = initTable[Hash256, TensorData]()

# Counter for generating unique hashes
var computeCounter {.global.}: int = 0

proc generateComputeHash*(): Hash256 =
  ## Generate a unique hash for computed tensors
  inc computeCounter
  let counter = computeCounter
  result[0] = byte(counter and 0xFF)
  result[1] = byte((counter shr 8) and 0xFF)
  result[2] = byte((counter shr 16) and 0xFF)
  result[3] = byte((counter shr 24) and 0xFF)
  # Mark as computed result
  result[30] = 0xC0  # "CO" for computed
  result[31] = 0xDE

proc registerTensorData*(tr: TensorRef, td: TensorData) =
  ## Register tensor data in global store
  tensorDataStore[tr.hash] = td

proc getTensorData*(tr: TensorRef): TensorData =
  ## Get tensor data from global store, or return zeros if not found
  if tr.hash in tensorDataStore:
    return tensorDataStore[tr.hash]
  # Return zeros if not registered
  result = newTensorDataZeros(tr.shape, tr.dtype)

proc hasTensorData*(tr: TensorRef): bool =
  ## Check if tensor has data in store
  tr.hash in tensorDataStore

proc newComputedTensor*(td: TensorData): TensorRef =
  ## Create a new tensor ref from computed data
  let h = generateComputeHash()
  result = newTensorRef(h, td.shape, td.dtype)
  registerTensorData(result, td)

# =============================================================================
# Linear Layer Computation
# =============================================================================

proc computeLinear*(input: TensorRef, weight: TensorData, bias: TensorData): TensorRef =
  ## Compute linear layer: output = input @ weight^T + bias
  let inputData = getTensorData(input)

  # Handle batched input
  let inputDims = inputData.shape.dims
  let outFeatures = weight.shape.dims[0]
  let inFeatures = weight.shape.dims[1]

  if inputDims.len == 2:
    # (batch, in) @ (out, in)^T = (batch, out)
    let batchSize = inputDims[0]

    # Transpose weight (out, in) -> (in, out)
    let weightT = newTensorDataZeros(newShape(inFeatures, outFeatures), weight.dtype)
    case weight.dtype
    of dtFloat32:
      let wArr = weight.asFloat32
      let wtArr = weightT.asFloat32
      for i in 0..<outFeatures:
        for j in 0..<inFeatures:
          wtArr[j * outFeatures + i] = wArr[i * inFeatures + j]
    of dtFloat64:
      let wArr = weight.asFloat64
      let wtArr = weightT.asFloat64
      for i in 0..<outFeatures:
        for j in 0..<inFeatures:
          wtArr[j * outFeatures + i] = wArr[i * inFeatures + j]
    else:
      discard

    # Matrix multiply: (batch, in) @ (in, out) = (batch, out)
    let outputData = mmKernel(inputData, weightT)

    # Add bias
    if not bias.isNil and bias.size > 0:
      case outputData.dtype
      of dtFloat32:
        let outArr = outputData.asFloat32
        let biasArr = bias.asFloat32
        for b in 0..<batchSize:
          for o in 0..<outFeatures:
            outArr[b * outFeatures + o] += biasArr[o]
      of dtFloat64:
        let outArr = outputData.asFloat64
        let biasArr = bias.asFloat64
        for b in 0..<batchSize:
          for o in 0..<outFeatures:
            outArr[b * outFeatures + o] += biasArr[o]
      else:
        discard

    return newComputedTensor(outputData)

  else:
    # Flatten to 2D, compute, reshape back
    var batchDims: seq[int] = @[]
    var batchSize = 1
    for i in 0..<(inputDims.len - 1):
      batchDims.add(inputDims[i])
      batchSize *= inputDims[i]

    # Reshape to (batchSize, inFeatures)
    let flatInput = newTensorDataZeros(newShape(batchSize, inFeatures), inputData.dtype)
    case inputData.dtype
    of dtFloat32:
      let inArr = inputData.asFloat32
      let flatArr = flatInput.asFloat32
      for i in 0..<inputData.size:
        flatArr[i] = inArr[i]
    of dtFloat64:
      let inArr = inputData.asFloat64
      let flatArr = flatInput.asFloat64
      for i in 0..<inputData.size:
        flatArr[i] = inArr[i]
    else:
      discard

    # Compute with flat input
    let flatRef = newComputedTensor(flatInput)
    let flatOutput = computeLinear(flatRef, weight, bias)
    let flatOutputData = getTensorData(flatOutput)

    # Reshape back to (*batch, outFeatures)
    var outputDims = batchDims
    outputDims.add(outFeatures)
    let outputData = newTensorDataZeros(newShape(outputDims), flatOutputData.dtype)

    case flatOutputData.dtype
    of dtFloat32:
      let srcArr = flatOutputData.asFloat32
      let dstArr = outputData.asFloat32
      for i in 0..<flatOutputData.size:
        dstArr[i] = srcArr[i]
    of dtFloat64:
      let srcArr = flatOutputData.asFloat64
      let dstArr = outputData.asFloat64
      for i in 0..<flatOutputData.size:
        dstArr[i] = srcArr[i]
    else:
      discard

    return newComputedTensor(outputData)

# =============================================================================
# Activation Computations
# =============================================================================

proc computeRelu*(input: TensorRef): TensorRef =
  ## Compute ReLU activation
  let inputData = getTensorData(input)
  let outputData = reluKernel(inputData)
  newComputedTensor(outputData)

proc computeLeakyRelu*(input: TensorRef, negativeSlope: float = 0.01): TensorRef =
  ## Compute Leaky ReLU activation
  let inputData = getTensorData(input)
  let outputData = leakyReluKernel(inputData, negativeSlope)
  newComputedTensor(outputData)

proc computeSigmoid*(input: TensorRef): TensorRef =
  ## Compute Sigmoid activation
  let inputData = getTensorData(input)
  let outputData = sigmoidKernel(inputData)
  newComputedTensor(outputData)

proc computeTanh*(input: TensorRef): TensorRef =
  ## Compute Tanh activation
  let inputData = getTensorData(input)
  let outputData = tanhKernel(inputData)
  newComputedTensor(outputData)

proc computeGelu*(input: TensorRef): TensorRef =
  ## Compute GELU activation
  let inputData = getTensorData(input)
  let outputData = geluKernel(inputData)
  newComputedTensor(outputData)

proc computeSilu*(input: TensorRef): TensorRef =
  ## Compute SiLU/Swish activation
  let inputData = getTensorData(input)
  let outputData = siluKernel(inputData)
  newComputedTensor(outputData)

proc computeSoftmax*(input: TensorRef, dim: int = -1): TensorRef =
  ## Compute Softmax
  let inputData = getTensorData(input)
  let outputData = softmaxKernel(inputData, dim)
  newComputedTensor(outputData)

# =============================================================================
# Convolution Computations
# =============================================================================

proc computeConv2d*(input: TensorRef, weight: TensorData, bias: TensorData,
                    strideH, strideW: int,
                    padH, padW: int,
                    dilationH, dilationW: int,
                    groups: int): TensorRef =
  ## Compute 2D convolution
  let inputData = getTensorData(input)
  let outputData = conv2dKernel(inputData, weight, bias,
                                strideH, strideW, padH, padW,
                                dilationH, dilationW, groups)
  newComputedTensor(outputData)

proc computeConv1d*(input: TensorRef, weight: TensorData, bias: TensorData,
                    stride, padding, dilation, groups: int): TensorRef =
  ## Compute 1D convolution
  let inputData = getTensorData(input)
  let outputData = conv1dKernel(inputData, weight, bias,
                                stride, padding, dilation, groups)
  newComputedTensor(outputData)

# =============================================================================
# Normalization Computations
# =============================================================================

proc computeBatchNorm2d*(input: TensorRef, gamma, beta: TensorData,
                         runningMean, runningVar: TensorData,
                         eps: float, training: bool, momentum: float): TensorRef =
  ## Compute Batch Normalization 2D
  let inputData = getTensorData(input)
  let (outputData, _, _) = batchNorm2dKernel(inputData, gamma, beta,
                                              runningMean, runningVar,
                                              eps, training, momentum)
  newComputedTensor(outputData)

proc computeLayerNorm*(input: TensorRef, normalizedShape: seq[int],
                       gamma, beta: TensorData, eps: float): TensorRef =
  ## Compute Layer Normalization
  let inputData = getTensorData(input)
  let outputData = layerNormKernel(inputData, normalizedShape, gamma, beta, eps)
  newComputedTensor(outputData)

proc computeRmsNorm*(input: TensorRef, normalizedShape: seq[int],
                     gamma: TensorData, eps: float): TensorRef =
  ## Compute RMS Normalization
  let inputData = getTensorData(input)
  let outputData = rmsNormKernel(inputData, normalizedShape, gamma, eps)
  newComputedTensor(outputData)

proc computeGroupNorm*(input: TensorRef, numGroups: int,
                       gamma, beta: TensorData, eps: float): TensorRef =
  ## Compute Group Normalization
  let inputData = getTensorData(input)
  let outputData = groupNormKernel(inputData, numGroups, gamma, beta, eps)
  newComputedTensor(outputData)

# =============================================================================
# Pooling Computations
# =============================================================================

proc computeMaxPool2d*(input: TensorRef, kernelH, kernelW: int,
                       strideH, strideW: int,
                       padH, padW: int,
                       dilationH, dilationW: int): TensorRef =
  ## Compute Max Pooling 2D
  let inputData = getTensorData(input)
  let (outputData, _) = maxPool2dKernel(inputData, kernelH, kernelW,
                                         strideH, strideW, padH, padW,
                                         dilationH, dilationW)
  newComputedTensor(outputData)

proc computeAvgPool2d*(input: TensorRef, kernelH, kernelW: int,
                       strideH, strideW: int,
                       padH, padW: int,
                       countIncludePad: bool): TensorRef =
  ## Compute Average Pooling 2D
  let inputData = getTensorData(input)
  let outputData = avgPool2dKernel(inputData, kernelH, kernelW,
                                   strideH, strideW, padH, padW,
                                   countIncludePad)
  newComputedTensor(outputData)

proc computeAdaptiveAvgPool2d*(input: TensorRef, outputH, outputW: int): TensorRef =
  ## Compute Adaptive Average Pooling 2D
  let inputData = getTensorData(input)
  let outputData = adaptiveAvgPool2dKernel(inputData, outputH, outputW)
  newComputedTensor(outputData)

proc computeMaxPool1d*(input: TensorRef, kernelSize: int,
                       stride, padding, dilation: int): TensorRef =
  ## Compute Max Pooling 1D
  let inputData = getTensorData(input)
  let (outputData, _) = maxPool1dKernel(inputData, kernelSize,
                                         stride, padding, dilation)
  newComputedTensor(outputData)

# =============================================================================
# Dropout Computation
# =============================================================================

proc computeDropout*(input: TensorRef, p: float, training: bool): TensorRef =
  ## Compute Dropout
  let inputData = getTensorData(input)
  let outputData = dropoutKernel(inputData, p, training)
  newComputedTensor(outputData)
