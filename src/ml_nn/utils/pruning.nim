## Pruning Utilities
##
## Neural network pruning methods for model compression.
## Supports structured, unstructured, and game-theory based pruning.

import std/[tables, algorithm, math, strformat, strutils]
import ml_core
import ../module

type
  PruningMethod* = enum
    ## Pruning strategies
    pmMagnitude        ## L1/L2 magnitude-based pruning
    pmRandom           ## Random pruning
    pmGradient         ## Gradient-based importance
    pmTaylor           ## First-order Taylor expansion
    pmSecondOrder      ## Second-order (Hessian-based)
    pmMovementPruning  ## Movement pruning (for fine-tuning)
    pmGameTheory       ## Game-theoretic equilibrium
    pmSensitivity      ## Layer sensitivity analysis
    pmWanda            ## Wanda (Weights and Activations)

  PruningScope* = enum
    ## Scope of pruning
    psGlobal    ## Global pruning across all layers
    psLocal     ## Per-layer pruning
    psStructured ## Structured (row/column/filter) pruning

  PruningSchedule* = enum
    ## Pruning schedule
    pschOneShot     ## One-shot pruning
    pschIterative   ## Gradual magnitude pruning
    pschCubic       ## Cubic sparsity schedule
    pschExponential ## Exponential decay

  # =============================================================================
  # Pruning Mask
  # =============================================================================

  PruningMask* = ref object
    ## Binary mask indicating which weights to keep (1) or prune (0)
    shape*: Shape
    data*: TensorData
    sparsity*: float  # Current sparsity level

  # =============================================================================
  # Pruning Context
  # =============================================================================

  PruningContext* = ref object
    ## Context for managing pruning state across a model
    module*: Module
    masks*: Table[string, PruningMask]
    targetSparsity*: float
    currentSparsity*: float
    pruningMethod*: PruningMethod
    pruningScope*: PruningScope
    pruningSchedule*: PruningSchedule
    currentStep*: int
    totalSteps*: int

  # =============================================================================
  # Importance Score
  # =============================================================================

  ImportanceScore* = ref object
    ## Importance scores for weights/neurons/filters
    paramName*: string
    scores*: TensorData
    ranking*: seq[int]  # Indices sorted by importance

  # =============================================================================
  # Pruning Result
  # =============================================================================

  PruningResult* = object
    ## Result of a pruning operation
    paramsPruned*: int
    totalParams*: int
    sparsity*: float
    compressionRatio*: float
    layerSparsities*: Table[string, float]

# =============================================================================
# Pruning Mask Operations
# =============================================================================

proc newPruningMask*(shape: Shape, dtype: DType = dtFloat32): PruningMask =
  ## Create a new pruning mask initialized to all ones (no pruning)
  result = PruningMask()
  result.shape = shape
  result.data = newTensorData(shape, dtype)
  result.sparsity = 0.0
  # Initialize all values to 1.0 (keep all weights)
  # In actual implementation, would fill with 1.0

proc calculateSparsity*(mask: PruningMask): float =
  ## Calculate the sparsity (ratio of zeros) in a mask
  let arr = mask.data.asFloat32
  var zeros = 0
  let n = mask.shape.size
  for i in 0..<n:
    if arr[i] == 0.0:
      zeros += 1
  if n > 0: zeros.float / n.float else: 0.0

proc applyMask*(param: Parameter, mask: PruningMask) =
  ## Apply a pruning mask to a parameter (element-wise multiplication)
  assert param.data.shape == mask.shape, "Parameter and mask shapes must match"
  # In actual implementation: param.data = param.data * mask.data

proc countNonzero*(mask: PruningMask): int =
  ## Count non-zero elements in mask
  let arr = mask.data.asFloat32
  result = 0
  for i in 0..<mask.shape.size:
    if arr[i] != 0.0:
      result += 1

# =============================================================================
# Magnitude-Based Pruning
# =============================================================================

proc computeMagnitudeScores*(param: Parameter): ImportanceScore =
  ## Compute L1 magnitude importance scores
  result = ImportanceScore()
  result.paramName = param.name
  result.scores = newTensorData(param.data.shape, dtFloat32)

  let src = param.data.asFloat32
  var dst = result.scores.asFloat32
  let n = param.data.shape.size

  # Compute absolute values
  for i in 0..<n:
    dst[i] = abs(src[i])

  # Compute ranking (indices sorted by importance, lowest first)
  var indices = newSeq[int](n)
  for i in 0..<n:
    indices[i] = i

  # Sort by score ascending (lowest importance first for pruning)
  indices.sort do (a, b: int) -> int:
    cmp(dst[a], dst[b])

  result.ranking = indices

proc magnitudePrune*(param: Parameter, sparsity: float): PruningMask =
  ## Prune weights by magnitude
  ##
  ## Args:
  ##   param: Parameter to prune
  ##   sparsity: Target sparsity (0.0 = no pruning, 1.0 = prune all)
  ##
  ## Returns:
  ##   Binary mask (1 = keep, 0 = prune)
  result = newPruningMask(param.data.shape)

  let scores = computeMagnitudeScores(param)
  let n = param.data.shape.size
  let numToPrune = int(sparsity * n.float)

  var maskArr = result.data.asFloat32
  # Initialize to 1.0
  for i in 0..<n:
    maskArr[i] = 1.0

  # Set lowest magnitude weights to 0
  for i in 0..<numToPrune:
    maskArr[scores.ranking[i]] = 0.0

  result.sparsity = sparsity

# =============================================================================
# Structured Pruning
# =============================================================================

proc computeFilterImportance*(conv: Module): seq[float] =
  ## Compute importance of each filter in a conv layer
  ## Uses L2 norm of filter weights
  if "weight" notin conv.parameters:
    return @[]

  let weight = conv.parameters["weight"]
  let dims = weight.data.shape.dims

  if dims.len != 4:
    return @[]

  let numFilters = dims[0]
  result = newSeq[float](numFilters)

  let arr = weight.data.asFloat32
  let filterSize = dims[1] * dims[2] * dims[3]

  for f in 0..<numFilters:
    var sumSq = 0.0
    for i in 0..<filterSize:
      let val = arr[f * filterSize + i]
      sumSq += val * val
    result[f] = sqrt(sumSq)

proc pruneFilters*(conv: Module, numToKeep: int): seq[int] =
  ## Prune filters by L2 norm, return indices of filters to keep
  let importance = computeFilterImportance(conv)
  if importance.len == 0:
    return @[]

  # Get indices sorted by importance (descending)
  var indices = newSeq[int](importance.len)
  for i in 0..<importance.len:
    indices[i] = i

  indices.sort do (a, b: int) -> int:
    -cmp(importance[a], importance[b])  # Descending

  # Return top numToKeep indices
  result = indices[0..<min(numToKeep, indices.len)]
  result.sort()  # Sort back to original order

proc computeNeuronImportance*(linear: Module): seq[float] =
  ## Compute importance of each output neuron in a linear layer
  ## Uses L1 norm of weights connected to each neuron
  if "weight" notin linear.parameters:
    return @[]

  let weight = linear.parameters["weight"]
  let dims = weight.data.shape.dims

  if dims.len != 2:
    return @[]

  let numNeurons = dims[0]
  let numInputs = dims[1]
  result = newSeq[float](numNeurons)

  let arr = weight.data.asFloat32

  for n in 0..<numNeurons:
    var sumAbs = 0.0
    for i in 0..<numInputs:
      sumAbs += abs(arr[n * numInputs + i])
    result[n] = sumAbs

proc pruneNeurons*(linear: Module, numToKeep: int): seq[int] =
  ## Prune neurons by L1 norm, return indices to keep
  let importance = computeNeuronImportance(linear)
  if importance.len == 0:
    return @[]

  var indices = newSeq[int](importance.len)
  for i in 0..<importance.len:
    indices[i] = i

  indices.sort do (a, b: int) -> int:
    -cmp(importance[a], importance[b])

  result = indices[0..<min(numToKeep, indices.len)]
  result.sort()

# =============================================================================
# Attention Head Pruning
# =============================================================================

proc computeHeadImportance*(attn: Module, numHeads: int): seq[float] =
  ## Compute importance of each attention head
  ## Can be based on gradient magnitude or activation patterns
  result = newSeq[float](numHeads)

  # Use L2 norm of query/key/value projections per head
  for name, param in attn.parameters.pairs:
    if "q_proj" in name or "k_proj" in name or "v_proj" in name:
      let dims = param.data.shape.dims
      if dims.len >= 2:
        let embedDim = dims[0]
        let headDim = embedDim div numHeads
        let arr = param.data.asFloat32

        for h in 0..<numHeads:
          var sumSq = 0.0
          for i in 0..<headDim:
            for j in 0..<dims[1]:
              let idx = (h * headDim + i) * dims[1] + j
              sumSq += arr[idx] * arr[idx]
          result[h] += sqrt(sumSq)

proc pruneAttentionHeads*(attn: Module, numHeads: int, numToKeep: int): seq[int] =
  ## Prune attention heads, return indices to keep
  let importance = computeHeadImportance(attn, numHeads)

  var indices = newSeq[int](numHeads)
  for i in 0..<numHeads:
    indices[i] = i

  indices.sort do (a, b: int) -> int:
    -cmp(importance[a], importance[b])

  result = indices[0..<min(numToKeep, numHeads)]
  result.sort()

# =============================================================================
# Game-Theoretic Pruning
# =============================================================================

type
  GameTheoreticPruner* = ref object
    ## Game-theoretic pruning based on Nash equilibrium
    ##
    ## Models weight importance as a cooperative game where
    ## Shapley values represent each weight's contribution.
    numSamples*: int  # Monte Carlo samples for Shapley approximation
    coalition_values*: Table[string, float]

proc newGameTheoreticPruner*(numSamples: int = 1000): GameTheoreticPruner =
  ## Create a game-theoretic pruner
  result = GameTheoreticPruner()
  result.numSamples = numSamples
  result.coalition_values = initTable[string, float]()

proc approximateShapleyValues*(pruner: GameTheoreticPruner,
                                param: Parameter,
                                valueFn: proc(mask: seq[bool]): float): seq[float] =
  ## Approximate Shapley values using Monte Carlo sampling
  ##
  ## Args:
  ##   param: Parameter whose elements to evaluate
  ##   valueFn: Coalition value function v(S) that evaluates a subset
  ##
  ## Returns:
  ##   Shapley value for each element
  let n = param.data.shape.size
  result = newSeq[float](n)

  # Monte Carlo approximation of Shapley values
  for _ in 0..<pruner.numSamples:
    # Random permutation
    var perm = newSeq[int](n)
    for i in 0..<n:
      perm[i] = i
    # Fisher-Yates shuffle would go here

    var mask = newSeq[bool](n)
    var prevValue = valueFn(mask)

    for idx in perm:
      mask[idx] = true
      let newValue = valueFn(mask)
      result[idx] += (newValue - prevValue)
      prevValue = newValue

  # Average over samples
  for i in 0..<n:
    result[i] /= pruner.numSamples.float

proc gameTheoreticPrune*(param: Parameter, sparsity: float,
                         valueFn: proc(mask: seq[bool]): float): PruningMask =
  ## Prune using game-theoretic Shapley values
  let pruner = newGameTheoreticPruner()
  let shapleyValues = pruner.approximateShapleyValues(param, valueFn)

  result = newPruningMask(param.data.shape)
  let n = param.data.shape.size
  let numToPrune = int(sparsity * n.float)

  # Sort by Shapley value ascending
  var indices = newSeq[int](n)
  for i in 0..<n:
    indices[i] = i
  indices.sort do (a, b: int) -> int:
    cmp(shapleyValues[a], shapleyValues[b])

  var maskArr = result.data.asFloat32
  for i in 0..<n:
    maskArr[i] = 1.0
  for i in 0..<numToPrune:
    maskArr[indices[i]] = 0.0

  result.sparsity = sparsity

# =============================================================================
# Gradual Magnitude Pruning
# =============================================================================

proc computeTargetSparsity*(schedule: PruningSchedule,
                             currentStep: int,
                             totalSteps: int,
                             initialSparsity: float,
                             finalSparsity: float): float =
  ## Compute target sparsity at current step based on schedule
  let t = currentStep.float / totalSteps.float

  case schedule
  of pschOneShot:
    if currentStep >= totalSteps:
      finalSparsity
    else:
      initialSparsity

  of pschIterative:
    # Linear interpolation
    initialSparsity + t * (finalSparsity - initialSparsity)

  of pschCubic:
    # Cubic schedule (from "To Prune or Not to Prune" paper)
    finalSparsity + (initialSparsity - finalSparsity) *
      pow(1.0 - t, 3)

  of pschExponential:
    # Exponential decay
    initialSparsity * pow(finalSparsity / initialSparsity, t)

type
  GradualMagnitudePruner* = ref object
    ## Gradual magnitude pruning (GMP)
    ##
    ## Prunes gradually during training according to a schedule.
    initialSparsity*: float
    finalSparsity*: float
    beginStep*: int
    endStep*: int
    frequency*: int
    schedule*: PruningSchedule

proc newGradualMagnitudePruner*(initialSparsity: float = 0.0,
                                 finalSparsity: float = 0.9,
                                 beginStep: int = 0,
                                 endStep: int = 1000,
                                 frequency: int = 100,
                                 schedule: PruningSchedule = pschCubic): GradualMagnitudePruner =
  ## Create a gradual magnitude pruner
  result = GradualMagnitudePruner()
  result.initialSparsity = initialSparsity
  result.finalSparsity = finalSparsity
  result.beginStep = beginStep
  result.endStep = endStep
  result.frequency = frequency
  result.schedule = schedule

proc step*(gmp: GradualMagnitudePruner, module: Module, currentStep: int) =
  ## Update pruning masks at current training step
  if currentStep < gmp.beginStep or currentStep > gmp.endStep:
    return

  if (currentStep - gmp.beginStep) mod gmp.frequency != 0:
    return

  let totalSteps = gmp.endStep - gmp.beginStep
  let relativeStep = currentStep - gmp.beginStep
  let targetSparsity = computeTargetSparsity(
    gmp.schedule, relativeStep, totalSteps,
    gmp.initialSparsity, gmp.finalSparsity
  )

  # Apply magnitude pruning to all parameters
  for param in module.parameters():
    let mask = magnitudePrune(param, targetSparsity)
    applyMask(param, mask)

# =============================================================================
# Layer-wise Sensitivity Analysis
# =============================================================================

proc analyzeLayerSensitivity*(module: Module,
                               evalFn: proc(): float,
                               sparsityLevels: seq[float] = @[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]): Table[string, seq[float]] =
  ## Analyze sensitivity of each layer to pruning
  ##
  ## Returns degradation at each sparsity level for each layer
  result = initTable[string, seq[float]]()

  let baseline = evalFn()

  for (name, param) in module.namedParameters():
    var sensitivities = newSeq[float]()

    for sparsity in sparsityLevels:
      let mask = magnitudePrune(param, sparsity)
      # Would temporarily apply mask, evaluate, restore
      let degradedPerf = evalFn()  # Placeholder
      sensitivities.add(baseline - degradedPerf)

    result[name] = sensitivities

proc optimalLayerSparsities*(sensitivity: Table[string, seq[float]],
                              sparsityLevels: seq[float],
                              totalSparsityBudget: float): Table[string, float] =
  ## Compute optimal per-layer sparsities given sensitivity analysis
  ##
  ## Uses a greedy algorithm to allocate sparsity budget
  result = initTable[string, float]()

  # Initialize all layers to 0 sparsity
  for name in sensitivity.keys:
    result[name] = 0.0

  # Greedy allocation - increase sparsity where sensitivity is lowest
  # This is a simplified version; optimal would use dynamic programming
  var remainingBudget = totalSparsityBudget
  let stepSize = 0.1

  while remainingBudget > 0:
    var bestLayer = ""
    var minSensitivity = Inf

    for name, sens in sensitivity.pairs:
      let currentIdx = int(result[name] / stepSize)
      if currentIdx < sens.len:
        if sens[currentIdx] < minSensitivity:
          minSensitivity = sens[currentIdx]
          bestLayer = name

    if bestLayer.len > 0:
      result[bestLayer] += stepSize
      remainingBudget -= stepSize
    else:
      break

# =============================================================================
# Wanda Pruning (Weights and Activations)
# =============================================================================

type
  WandaPruner* = ref object
    ## Wanda: Pruning by Weights and activations
    ##
    ## Combines weight magnitude with activation importance.
    ## More effective than pure magnitude pruning for LLMs.
    activationScales*: Table[string, TensorData]

proc newWandaPruner*(): WandaPruner =
  ## Create a Wanda pruner
  result = WandaPruner()
  result.activationScales = initTable[string, TensorData]()

proc recordActivationScales*(wanda: WandaPruner, name: string, activations: TensorData) =
  ## Record activation magnitudes for a layer
  ## Call this during a calibration forward pass
  let arr = activations.asFloat32
  let n = activations.shape.size

  # Compute mean absolute activation per feature
  # Simplified: just track overall scale
  var sumAbs = 0.0
  for i in 0..<n:
    sumAbs += abs(arr[i])

  wanda.activationScales[name] = activations  # Would store actual scales

proc wandaPrune*(wanda: WandaPruner, param: Parameter, paramName: string,
                 sparsity: float): PruningMask =
  ## Prune using Wanda metric: |W| * ||X||
  result = newPruningMask(param.data.shape)

  let weightArr = param.data.asFloat32
  let n = param.data.shape.size

  # Compute importance scores combining weight and activation
  var scores = newSeq[float](n)
  for i in 0..<n:
    # In actual implementation, would combine with activation scales
    scores[i] = abs(weightArr[i])

  # Sort by score ascending
  var indices = newSeq[int](n)
  for i in 0..<n:
    indices[i] = i
  indices.sort do (a, b: int) -> int:
    cmp(scores[a], scores[b])

  let numToPrune = int(sparsity * n.float)
  var maskArr = result.data.asFloat32
  for i in 0..<n:
    maskArr[i] = 1.0
  for i in 0..<numToPrune:
    maskArr[indices[i]] = 0.0

  result.sparsity = sparsity

# =============================================================================
# Model-Wide Pruning Functions
# =============================================================================

proc globalMagnitudePrune*(module: Module, sparsity: float): PruningResult =
  ## Globally prune model by magnitude across all parameters
  var allWeights: seq[(string, int, float)] = @[]  # (name, idx, magnitude)

  # Collect all weights with their magnitudes
  for (name, param) in module.namedParameters():
    let arr = param.data.asFloat32
    for i in 0..<param.data.shape.size:
      allWeights.add((name, i, abs(arr[i]).float))

  # Sort by magnitude ascending
  allWeights.sort do (a, b: (string, int, float)) -> int:
    cmp(a[2], b[2])

  let totalParams = allWeights.len
  let numToPrune = int(sparsity * totalParams.float)

  # Create masks for each parameter
  var masks = initTable[string, PruningMask]()
  for (name, param) in module.namedParameters():
    masks[name] = newPruningMask(param.data.shape)
    let maskArr = masks[name].data.asFloat32
    for i in 0..<param.data.shape.size:
      maskArr[i] = 1.0

  # Apply pruning
  for i in 0..<numToPrune:
    let (name, idx, _) = allWeights[i]
    let maskArr = masks[name].data.asFloat32
    maskArr[idx] = 0.0

  # Apply masks
  for (name, param) in module.namedParameters():
    applyMask(param, masks[name])

  # Compute layer sparsities
  var layerSparsities = initTable[string, float]()
  for name, mask in masks.pairs:
    layerSparsities[name] = calculateSparsity(mask)

  result = PruningResult(
    paramsPruned: numToPrune,
    totalParams: totalParams,
    sparsity: numToPrune.float / totalParams.float,
    compressionRatio: 1.0 / (1.0 - sparsity),
    layerSparsities: layerSparsities
  )

proc localMagnitudePrune*(module: Module, sparsity: float): PruningResult =
  ## Prune each layer independently to the same sparsity level
  var totalPruned = 0
  var totalParams = 0
  var layerSparsities = initTable[string, float]()

  for (name, param) in module.namedParameters():
    let mask = magnitudePrune(param, sparsity)
    applyMask(param, mask)

    let n = param.data.shape.size
    let pruned = int(n.float * sparsity)
    totalPruned += pruned
    totalParams += n
    layerSparsities[name] = sparsity

  result = PruningResult(
    paramsPruned: totalPruned,
    totalParams: totalParams,
    sparsity: totalPruned.float / totalParams.float,
    compressionRatio: 1.0 / (1.0 - sparsity),
    layerSparsities: layerSparsities
  )

# =============================================================================
# Pruning Context Management
# =============================================================================

proc newPruningContext*(module: Module,
                         targetSparsity: float,
                         pruningMethod: PruningMethod = pmMagnitude,
                         pruningScope: PruningScope = psLocal,
                         pruningSchedule: PruningSchedule = pschOneShot,
                         totalSteps: int = 1): PruningContext =
  ## Create a pruning context for managing pruning state
  result = PruningContext()
  result.module = module
  result.masks = initTable[string, PruningMask]()
  result.targetSparsity = targetSparsity
  result.currentSparsity = 0.0
  result.pruningMethod = pruningMethod
  result.pruningScope = pruningScope
  result.pruningSchedule = pruningSchedule
  result.currentStep = 0
  result.totalSteps = totalSteps

  # Initialize masks for all parameters
  for (name, param) in module.namedParameters():
    result.masks[name] = newPruningMask(param.data.shape)

proc step*(ctx: PruningContext) =
  ## Perform one pruning step
  ctx.currentStep += 1

  let targetSparsity = computeTargetSparsity(
    ctx.pruningSchedule, ctx.currentStep, ctx.totalSteps,
    0.0, ctx.targetSparsity
  )

  case ctx.pruningScope
  of psGlobal:
    discard globalMagnitudePrune(ctx.module, targetSparsity)
  of psLocal:
    discard localMagnitudePrune(ctx.module, targetSparsity)
  of psStructured:
    # Would implement structured pruning
    discard localMagnitudePrune(ctx.module, targetSparsity)

  ctx.currentSparsity = targetSparsity

proc finalize*(ctx: PruningContext): PruningResult =
  ## Finalize pruning and return results
  var totalPruned = 0
  var totalParams = 0
  var layerSparsities = initTable[string, float]()

  for name, mask in ctx.masks.pairs:
    let n = mask.shape.size
    let pruned = n - countNonzero(mask)
    totalPruned += pruned
    totalParams += n
    layerSparsities[name] = pruned.float / n.float

  result = PruningResult(
    paramsPruned: totalPruned,
    totalParams: totalParams,
    sparsity: totalPruned.float / totalParams.float,
    compressionRatio: 1.0 / (1.0 - ctx.currentSparsity),
    layerSparsities: layerSparsities
  )

# =============================================================================
# Utilities
# =============================================================================

proc printPruningReport*(res: PruningResult) =
  ## Print a human-readable pruning report
  echo "=".repeat(50)
  echo "Pruning Report"
  echo "=".repeat(50)
  echo fmt"Total Parameters: {res.totalParams}"
  echo fmt"Parameters Pruned: {res.paramsPruned}"
  echo fmt"Overall Sparsity: {res.sparsity * 100.0:.2f}%"
  echo fmt"Compression Ratio: {res.compressionRatio:.2f}x"
  echo ""
  echo "Per-Layer Sparsities:"
  for name, sparsity in res.layerSparsities.pairs:
    echo fmt"  {name}: {sparsity * 100.0:.2f}%"
  echo "=".repeat(50)

proc estimateSpeedupTheoretical*(sparsity: float): float =
  ## Estimate theoretical speedup from sparsity
  ## Assumes dense compute can be converted to sparse
  1.0 / (1.0 - sparsity)

proc estimateMemorySavings*(sparsity: float, useSparseStorage: bool): float =
  ## Estimate memory savings from pruning
  if useSparseStorage:
    # CSR format: needs values + column_indices + row_pointers
    # Rough estimate: 2x overhead for sparse vs dense at low sparsity
    let sparseOverhead = 2.0
    let denseCost = 1.0
    let sparseCost = (1.0 - sparsity) * sparseOverhead
    if sparseCost < denseCost:
      1.0 - sparseCost
    else:
      0.0
  else:
    # Dense storage with zeros still uses same memory
    0.0
