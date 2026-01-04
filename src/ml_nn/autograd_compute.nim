## Autograd-aware Computation Bridge
##
## Wraps compute functions with gradient tape recording for automatic differentiation.
## Enables gradient-based pruning methods (Taylor, Movement, Equilibrium).

import std/[tables, options, math, algorithm]
import ml_core
import ml_autograd except getTensorData
import ./compute

# Helper: sum along axis (axis=0 means sum columns)
proc sumAxisData*(input: TensorData, axis: int): TensorData =
  ## Sum tensor along specified axis
  if input.shape.rank != 2:
    return input

  let rows = input.shape.dims[0]
  let cols = input.shape.dims[1]

  if axis == 0:
    # Sum along rows -> result is (cols,)
    result = newTensorDataZeros(newShape(cols), input.dtype)
    case input.dtype
    of dtFloat32:
      let inArr = input.asFloat32
      let outArr = result.asFloat32
      for j in 0..<cols:
        var sum = 0.0'f32
        for i in 0..<rows:
          sum += inArr[i * cols + j]
        outArr[j] = sum
    of dtFloat64:
      let inArr = input.asFloat64
      let outArr = result.asFloat64
      for j in 0..<cols:
        var sum = 0.0'f64
        for i in 0..<rows:
          sum += inArr[i * cols + j]
        outArr[j] = sum
    else:
      discard
  else:
    # Sum along columns -> result is (rows,)
    result = newTensorDataZeros(newShape(rows), input.dtype)
    case input.dtype
    of dtFloat32:
      let inArr = input.asFloat32
      let outArr = result.asFloat32
      for i in 0..<rows:
        var sum = 0.0'f32
        for j in 0..<cols:
          sum += inArr[i * cols + j]
        outArr[i] = sum
    of dtFloat64:
      let inArr = input.asFloat64
      let outArr = result.asFloat64
      for i in 0..<rows:
        var sum = 0.0'f64
        for j in 0..<cols:
          sum += inArr[i * cols + j]
        outArr[i] = sum
    else:
      discard

proc sumAxis*(input: TensorRef, axis: int): TensorRef =
  ## Sum tensor along specified axis
  let inputData = getTensorData(input)
  let resultData = sumAxisData(inputData, axis)
  newComputedTensorRef(resultData)

# =============================================================================
# Gradient Functions for Neural Network Layers
# =============================================================================

proc gradLinear*(grad: TensorRef, inputs: seq[TensorRef],
                 output: TensorRef): seq[TensorRef] =
  ## Gradient for linear layer: y = x @ W^T + b
  ## dL/dx = dL/dy @ W
  ## dL/dW = x^T @ dL/dy (transposed for weight shape)
  ## dL/db = sum(dL/dy, axis=0)
  if inputs.len < 2:
    return @[grad]

  let x = inputs[0]      # Input
  let weight = inputs[1] # Weight (out, in)

  # dL/dx = grad @ weight
  let gradInput = matmul(grad, weight)

  # dL/dW = x^T @ grad, then transpose to match weight shape
  let gradWeight = matmul(transpose(x), grad)

  # dL/db = sum over batch dimension
  let gradBias = if inputs.len > 2:
    # Sum along batch dimension
    sumAxis(grad, 0)
  else:
    nil

  if gradBias.isNil:
    @[gradInput, transpose(gradWeight)]
  else:
    @[gradInput, transpose(gradWeight), gradBias]

proc gradRelu*(grad: TensorRef, inputs: seq[TensorRef],
               output: TensorRef): seq[TensorRef] =
  ## Gradient for ReLU: 1 if x > 0 else 0
  if inputs.len == 0:
    return @[grad]
  @[mul(grad, reluMask(inputs[0]))]

proc gradSigmoid*(grad: TensorRef, inputs: seq[TensorRef],
                  output: TensorRef): seq[TensorRef] =
  ## Gradient for sigmoid: sigmoid(x) * (1 - sigmoid(x))
  let onesT = ones(output.shape, output.dtype)
  let oneMinusOutput = sub(onesT, output)
  @[mul(grad, mul(output, oneMinusOutput))]

proc gradTanh*(grad: TensorRef, inputs: seq[TensorRef],
               output: TensorRef): seq[TensorRef] =
  ## Gradient for tanh: 1 - tanh(x)^2
  let outputSquared = square(output)
  let onesT = ones(output.shape, output.dtype)
  @[mul(grad, sub(onesT, outputSquared))]

proc gradGelu*(grad: TensorRef, inputs: seq[TensorRef],
               output: TensorRef): seq[TensorRef] =
  ## Gradient for GELU (approximate)
  if inputs.len == 0:
    return @[grad]
  let x = inputs[0]
  let scaled = scale(x, 1.702)
  let sig = sigmoidRef(scaled)
  let onesT = ones(x.shape, x.dtype)
  let oneMinusSig = sub(onesT, sig)
  let term1 = sig
  let term2 = mul(scale(mul(mul(x, sig), oneMinusSig), 1.702), onesT)
  @[mul(grad, add(term1, term2))]

proc gradSoftmax*(grad: TensorRef, inputs: seq[TensorRef],
                  output: TensorRef): seq[TensorRef] =
  ## Gradient for softmax (simplified diagonal Jacobian approximation)
  let onesT = ones(output.shape, output.dtype)
  let oneMinusOutput = sub(onesT, output)
  @[mul(grad, mul(output, oneMinusOutput))]

proc gradDropout*(grad: TensorRef, inputs: seq[TensorRef],
                  output: TensorRef): seq[TensorRef] =
  ## Gradient for dropout: pass through gradient (mask applied during forward)
  @[grad]

# =============================================================================
# Autograd-aware Compute Functions
# =============================================================================

proc computeLinearWithGrad*(tape: GradientTape, input: TensorRef,
                            weight: TensorData, bias: TensorData): TensorRef =
  ## Linear layer with gradient recording
  let output = computeLinear(input, weight, bias)

  if tape.isWatching(input):
    # Create TensorRefs for weight and bias for gradient computation
    let weightRef = newComputedTensorRef(weight)
    let biasRef = if not bias.isNil: newComputedTensorRef(bias) else: nil

    let inputs = if biasRef.isNil:
      @[input, weightRef]
    else:
      @[input, weightRef, biasRef]

    tape.record(opMatMul, inputs, output, gradLinear)

  output

proc computeReluWithGrad*(tape: GradientTape, input: TensorRef): TensorRef =
  ## ReLU activation with gradient recording
  let output = computeRelu(input)

  if tape.isWatching(input):
    tape.record(opRelu, @[input], output, gradRelu)

  output

proc computeSigmoidWithGrad*(tape: GradientTape, input: TensorRef): TensorRef =
  ## Sigmoid activation with gradient recording
  let output = computeSigmoid(input)

  if tape.isWatching(input):
    tape.record(opSigmoid, @[input], output, gradSigmoid)

  output

proc computeTanhWithGrad*(tape: GradientTape, input: TensorRef): TensorRef =
  ## Tanh activation with gradient recording
  let output = computeTanh(input)

  if tape.isWatching(input):
    tape.record(opTanh, @[input], output, gradTanh)

  output

proc computeGeluWithGrad*(tape: GradientTape, input: TensorRef): TensorRef =
  ## GELU activation with gradient recording
  let output = computeGelu(input)

  if tape.isWatching(input):
    # Use custom op kind or map to existing
    tape.record(opGelu, @[input], output, gradGelu)

  output

proc computeSoftmaxWithGrad*(tape: GradientTape, input: TensorRef,
                             dim: int = -1): TensorRef =
  ## Softmax with gradient recording
  let output = computeSoftmax(input, dim)

  if tape.isWatching(input):
    tape.record(opSoftmax, @[input], output, gradSoftmax)

  output

proc computeDropoutWithGrad*(tape: GradientTape, input: TensorRef,
                             p: float, training: bool): TensorRef =
  ## Dropout with gradient recording
  let output = computeDropout(input, p, training)

  if tape.isWatching(input) and training:
    tape.record(opDropout, @[input], output, gradDropout)

  output

# =============================================================================
# Gradient Computation for Pruning
# =============================================================================

type
  GradientCache* = ref object
    ## Cache for storing gradients during pruning analysis
    tape*: GradientTape
    ctx*: GradientContext
    paramGrads*: Table[string, TensorRef]
    activationGrads*: Table[string, TensorRef]

proc newGradientCache*(): GradientCache =
  ## Create a new gradient cache
  result = GradientCache()
  result.tape = newGradientTape(persistent = true)
  result.ctx = newGradientContext(result.tape)
  result.paramGrads = initTable[string, TensorRef]()
  result.activationGrads = initTable[string, TensorRef]()

proc watchParameters*(cache: GradientCache, params: Table[string, TensorRef]) =
  ## Watch all parameters for gradient computation
  for name, param in params.pairs:
    cache.tape.watch(param)

proc computeAndCacheGradients*(cache: GradientCache, loss: TensorRef) =
  ## Compute gradients and cache them
  # Set initial gradient (1.0 for scalar loss)
  let onesGrad = ones(loss.shape, loss.dtype)
  cache.ctx.setGrad(loss, onesGrad)

  # Run backward pass
  cache.ctx.backward(onesGrad)

proc getParamGradient*(cache: GradientCache, paramHash: Hash256): Option[TensorRef] =
  ## Get cached gradient for a parameter
  let gradOpt = cache.ctx.getGrad(newTensorRef(paramHash, newShape(1), dtFloat32))
  if gradOpt.isSome:
    some(gradOpt.get)
  else:
    none(TensorRef)

proc clearCache*(cache: GradientCache) =
  ## Clear cached gradients
  cache.ctx.clearGrads()
  cache.tape.clear()
  cache.paramGrads.clear()
  cache.activationGrads.clear()

# =============================================================================
# Taylor Importance Computation
# =============================================================================

proc computeTaylorImportance*(weight: TensorRef, grad: TensorRef): TensorData =
  ## Compute first-order Taylor importance: |weight * gradient|
  ## This approximates the change in loss if the weight is removed
  let weightData = getTensorData(weight)
  let gradData = getTensorData(grad)

  result = newTensorDataZeros(weight.shape, weight.dtype)

  case weight.dtype
  of dtFloat32:
    let wArr = weightData.asFloat32
    let gArr = gradData.asFloat32
    let outArr = result.asFloat32
    for i in 0..<weight.size:
      outArr[i] = abs(wArr[i] * gArr[i])
  of dtFloat64:
    let wArr = weightData.asFloat64
    let gArr = gradData.asFloat64
    let outArr = result.asFloat64
    for i in 0..<weight.size:
      outArr[i] = abs(wArr[i] * gArr[i])
  else:
    discard

proc computeTaylorRanking*(importance: TensorData): seq[int] =
  ## Rank weights by Taylor importance (ascending - lowest importance first)
  var indexed: seq[(float, int)] = @[]

  case importance.dtype
  of dtFloat32:
    let arr = importance.asFloat32
    for i in 0..<importance.size:
      indexed.add((arr[i].float, i))
  of dtFloat64:
    let arr = importance.asFloat64
    for i in 0..<importance.size:
      indexed.add((arr[i], i))
  else:
    discard

  # Sort by importance (ascending)
  proc compareTuples(a, b: (float, int)): int = cmp(a[0], b[0])
  indexed.sort(compareTuples)

  result = @[]
  for item in indexed:
    result.add(item[1])

# =============================================================================
# Movement Pruning Support
# =============================================================================

type
  MovementTracker* = ref object
    ## Tracks weight movement during fine-tuning
    initialWeights*: Table[string, TensorData]
    currentWeights*: Table[string, TensorData]
    movementScores*: Table[string, TensorData]
    steps*: int

proc newMovementTracker*(): MovementTracker =
  ## Create a new movement tracker
  MovementTracker(
    initialWeights: initTable[string, TensorData](),
    currentWeights: initTable[string, TensorData](),
    movementScores: initTable[string, TensorData](),
    steps: 0
  )

proc recordInitialWeights*(tracker: MovementTracker,
                           params: Table[string, TensorData]) =
  ## Record initial weights before fine-tuning
  for name, data in params.pairs:
    # Clone the data
    let cloned = newTensorDataZeros(data.shape, data.dtype)
    case data.dtype
    of dtFloat32:
      let src = data.asFloat32
      let dst = cloned.asFloat32
      for i in 0..<data.size:
        dst[i] = src[i]
    of dtFloat64:
      let src = data.asFloat64
      let dst = cloned.asFloat64
      for i in 0..<data.size:
        dst[i] = src[i]
    else:
      discard
    tracker.initialWeights[name] = cloned

proc updateMovementScores*(tracker: MovementTracker,
                           params: Table[string, TensorData]) =
  ## Update movement scores based on weight changes
  tracker.steps += 1

  for name, currentData in params.pairs:
    if name notin tracker.initialWeights:
      continue

    let initialData = tracker.initialWeights[name]

    # Initialize movement scores if needed
    if name notin tracker.movementScores:
      tracker.movementScores[name] = newTensorDataZeros(currentData.shape, currentData.dtype)

    let scores = tracker.movementScores[name]

    # Accumulate absolute movement
    case currentData.dtype
    of dtFloat32:
      let initArr = initialData.asFloat32
      let currArr = currentData.asFloat32
      let scoreArr = scores.asFloat32
      for i in 0..<currentData.size:
        scoreArr[i] += abs(currArr[i] - initArr[i])
    of dtFloat64:
      let initArr = initialData.asFloat64
      let currArr = currentData.asFloat64
      let scoreArr = scores.asFloat64
      for i in 0..<currentData.size:
        scoreArr[i] += abs(currArr[i] - initArr[i])
    else:
      discard

proc getMovementImportance*(tracker: MovementTracker, name: string): TensorData =
  ## Get accumulated movement importance for a parameter
  if name in tracker.movementScores:
    tracker.movementScores[name]
  else:
    nil

# =============================================================================
# Equilibrium (Game-Theory) Pruning Support
# =============================================================================

type
  EquilibriumState* = ref object
    ## State for game-theoretic equilibrium pruning
    playerUtilities*: Table[string, TensorData]  # Utility for keeping each weight
    equilibriumMask*: Table[string, TensorData]  # Current equilibrium state
    temperature*: float
    learningRate*: float
    steps*: int

proc newEquilibriumState*(temperature: float = 1.0,
                          learningRate: float = 0.1): EquilibriumState =
  ## Create equilibrium state for game-theoretic pruning
  EquilibriumState(
    playerUtilities: initTable[string, TensorData](),
    equilibriumMask: initTable[string, TensorData](),
    temperature: temperature,
    learningRate: learningRate,
    steps: 0
  )

proc initializeUtilities*(state: EquilibriumState,
                          params: Table[string, TensorData]) =
  ## Initialize utilities based on weight magnitudes
  for name, data in params.pairs:
    let utility = newTensorDataZeros(data.shape, data.dtype)
    let mask = newTensorDataZeros(data.shape, data.dtype)

    case data.dtype
    of dtFloat32:
      let wArr = data.asFloat32
      let uArr = utility.asFloat32
      let mArr = mask.asFloat32
      for i in 0..<data.size:
        uArr[i] = abs(wArr[i])  # Initial utility = magnitude
        mArr[i] = 1.0'f32       # Start with all weights kept
    of dtFloat64:
      let wArr = data.asFloat64
      let uArr = utility.asFloat64
      let mArr = mask.asFloat64
      for i in 0..<data.size:
        uArr[i] = abs(wArr[i])
        mArr[i] = 1.0'f64
    else:
      discard

    state.playerUtilities[name] = utility
    state.equilibriumMask[name] = mask

proc updateEquilibrium*(state: EquilibriumState,
                        gradients: Table[string, TensorData],
                        targetSparsity: float) =
  ## Update equilibrium state using gradient information
  ## Game-theoretic update: each weight "plays" to maximize utility
  state.steps += 1

  for name, utility in state.playerUtilities.mpairs:
    if name notin gradients:
      continue

    let grad = gradients[name]
    let mask = state.equilibriumMask[name]

    case utility.dtype
    of dtFloat32:
      let uArr = utility.asFloat32
      let gArr = grad.asFloat32
      let mArr = mask.asFloat32

      # Update utility based on gradient (Taylor-like importance)
      for i in 0..<utility.size:
        let gradMag = abs(gArr[i])
        uArr[i] = uArr[i] * 0.9 + gradMag * 0.1  # EMA update

      # Compute soft mask using temperature-scaled softmax-like function
      var totalUtil = 0.0'f32
      for i in 0..<utility.size:
        totalUtil += exp(uArr[i] / state.temperature.float32)

      # Update mask towards equilibrium
      for i in 0..<utility.size:
        let prob = exp(uArr[i] / state.temperature.float32) / totalUtil
        mArr[i] = mArr[i] * (1.0 - state.learningRate.float32) +
                  prob * state.learningRate.float32 * utility.size.float32

    of dtFloat64:
      let uArr = utility.asFloat64
      let gArr = grad.asFloat64
      let mArr = mask.asFloat64

      for i in 0..<utility.size:
        let gradMag = abs(gArr[i])
        uArr[i] = uArr[i] * 0.9 + gradMag * 0.1

      var totalUtil = 0.0'f64
      for i in 0..<utility.size:
        totalUtil += exp(uArr[i] / state.temperature)

      for i in 0..<utility.size:
        let prob = exp(uArr[i] / state.temperature) / totalUtil
        mArr[i] = mArr[i] * (1.0 - state.learningRate) +
                  prob * state.learningRate * utility.size.float
    else:
      discard

proc getEquilibriumMask*(state: EquilibriumState, name: string,
                         sparsity: float): TensorData =
  ## Get binary mask from equilibrium state at target sparsity
  if name notin state.equilibriumMask:
    return nil

  let softMask = state.equilibriumMask[name]
  result = newTensorDataZeros(softMask.shape, softMask.dtype)

  # Convert soft mask to hard mask at target sparsity
  var values: seq[(float, int)] = @[]

  case softMask.dtype
  of dtFloat32:
    let mArr = softMask.asFloat32
    for i in 0..<softMask.size:
      values.add((mArr[i].float, i))
  of dtFloat64:
    let mArr = softMask.asFloat64
    for i in 0..<softMask.size:
      values.add((mArr[i], i))
  else:
    discard

  # Sort by mask value (descending - keep highest)
  proc compareDesc(a, b: (float, int)): int = cmp(b[0], a[0])
  values.sort(compareDesc)

  # Keep top (1 - sparsity) fraction
  let keepCount = int(softMask.size.float * (1.0 - sparsity))

  case result.dtype
  of dtFloat32:
    let outArr = result.asFloat32
    for i in 0..<softMask.size:
      outArr[i] = 0.0'f32
    for i in 0..<keepCount:
      outArr[values[i][1]] = 1.0'f32
  of dtFloat64:
    let outArr = result.asFloat64
    for i in 0..<softMask.size:
      outArr[i] = 0.0'f64
    for i in 0..<keepCount:
      outArr[values[i][1]] = 1.0'f64
  else:
    discard
