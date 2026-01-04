## Autograd-aware Module System
##
## Extends the Module system with gradient tape integration for automatic differentiation.
## Enables gradient-based pruning and training.

import std/[tables, options, hashes, algorithm]
import ml_core
import ml_autograd except getTensorData
import ./module
import ./compute
import ./autograd_compute

type
  GradientModule* = ref object of Module
    ## Module with gradient tape integration
    tape*: GradientTape
    gradCache*: GradientCache

# =============================================================================
# Gradient Module Creation
# =============================================================================

proc newGradientModule*(name: string = ""): GradientModule =
  ## Create a new gradient-aware module
  result = GradientModule()
  result.initModule(name)
  result.tape = newGradientTape(persistent = true)
  result.gradCache = newGradientCache()

proc initGradientModule*(m: GradientModule, name: string = "") =
  ## Initialize gradient module fields
  m.initModule(name)
  m.tape = newGradientTape(persistent = true)
  m.gradCache = newGradientCache()

# =============================================================================
# Tape Management
# =============================================================================

proc watchAllParameters*(m: GradientModule) =
  ## Watch all parameters for gradient computation
  for param in m.parameters():
    m.tape.watch(param.tensorRef)

proc watchParameter*(m: GradientModule, name: string) =
  ## Watch a specific parameter
  let paramOpt = m.getParameter(name)
  if paramOpt.isSome:
    m.tape.watch(paramOpt.get.tensorRef)

proc unwatchAllParameters*(m: GradientModule) =
  ## Stop watching all parameters
  for param in m.parameters():
    m.tape.unwatch(param.tensorRef)

proc clearTape*(m: GradientModule) =
  ## Clear the gradient tape
  m.tape.clear()

proc resetTape*(m: GradientModule) =
  ## Reset tape completely
  m.tape.reset()

proc getTape*(m: GradientModule): GradientTape =
  ## Get the gradient tape
  m.tape

# =============================================================================
# Gradient Computation
# =============================================================================

proc backward*(m: GradientModule, loss: TensorRef,
               retainGraph: bool = false): Table[Hash256, TensorRef] =
  ## Perform backward pass and compute gradients
  let options = newBackwardOptions(retainGraph = retainGraph)
  result = m.tape.backward(loss, nil, options)

  # Update parameter gradients
  for name, param in m.parameters.pairs:
    if param.tensorRef.hash in result:
      m.parameters[name].gradRef = some(result[param.tensorRef.hash])

proc computeGradients*(m: GradientModule, loss: TensorRef,
                       params: seq[string] = @[]): Table[string, TensorRef] =
  ## Compute gradients for named parameters
  let allGrads = m.backward(loss, retainGraph = true)
  result = initTable[string, TensorRef]()

  let paramList = if params.len == 0:
    var names: seq[string] = @[]
    for name in m.parameters.keys:
      names.add(name)
    names
  else:
    params

  for name in paramList:
    if name in m.parameters:
      let param = m.parameters[name]
      if param.tensorRef.hash in allGrads:
        result[name] = allGrads[param.tensorRef.hash]

proc getGradient*(m: GradientModule, name: string): Option[TensorRef] =
  ## Get gradient for a named parameter
  if name in m.parameters:
    m.parameters[name].gradRef
  else:
    none(TensorRef)

proc zeroGrad*(m: GradientModule) =
  ## Zero all gradients
  m.clearAllGradients()

# =============================================================================
# Forward with Gradient Recording
# =============================================================================

type
  GradientContext* = object
    ## Context for gradient-aware forward pass
    tape*: GradientTape
    recording*: bool

proc gradContext*(m: GradientModule, recording: bool = true): GradientContext =
  ## Create a gradient context for forward pass
  GradientContext(tape: m.tape, recording: recording)

proc noGrad*(m: GradientModule): GradientContext =
  ## Create a no-gradient context
  GradientContext(tape: m.tape, recording: false)

# =============================================================================
# Gradient-based Pruning Integration
# =============================================================================

type
  GradientPruningContext* = ref object
    ## Context for gradient-based pruning
    module*: GradientModule
    taylorScores*: Table[string, TensorData]
    movementTracker*: MovementTracker
    equilibriumState*: EquilibriumState

proc newGradientPruningContext*(m: GradientModule): GradientPruningContext =
  ## Create a gradient pruning context
  result = GradientPruningContext()
  result.module = m
  result.taylorScores = initTable[string, TensorData]()
  result.movementTracker = newMovementTracker()
  result.equilibriumState = newEquilibriumState()

proc initForPruning*(ctx: GradientPruningContext) =
  ## Initialize context for pruning
  # Watch all parameters
  ctx.module.watchAllParameters()

  # Record initial weights for movement pruning
  var initialWeights = initTable[string, TensorData]()
  for name, param in ctx.module.parameters.pairs:
    initialWeights[name] = param.data
  ctx.movementTracker.recordInitialWeights(initialWeights)

  # Initialize equilibrium state
  ctx.equilibriumState.initializeUtilities(initialWeights)

proc computeTaylorScores*(ctx: GradientPruningContext, loss: TensorRef) =
  ## Compute Taylor importance scores using gradients
  let grads = ctx.module.computeGradients(loss)

  for name, param in ctx.module.parameters.pairs:
    if name in grads:
      let importance = computeTaylorImportance(param.tensorRef, grads[name])
      ctx.taylorScores[name] = importance

proc updateMovement*(ctx: GradientPruningContext) =
  ## Update movement tracking after a training step
  var currentWeights = initTable[string, TensorData]()
  for name, param in ctx.module.parameters.pairs:
    currentWeights[name] = param.data
  ctx.movementTracker.updateMovementScores(currentWeights)

proc updateEquilibrium*(ctx: GradientPruningContext, loss: TensorRef,
                        targetSparsity: float) =
  ## Update equilibrium state with new gradients
  let grads = ctx.module.computeGradients(loss)

  var gradData = initTable[string, TensorData]()
  for name, gradRef in grads.pairs:
    gradData[name] = getTensorData(gradRef)

  ctx.equilibriumState.updateEquilibrium(gradData, targetSparsity)

proc getTaylorMask*(ctx: GradientPruningContext, name: string,
                    sparsity: float): TensorData =
  ## Get pruning mask based on Taylor scores
  if name notin ctx.taylorScores:
    return nil

  let scores = ctx.taylorScores[name]
  let ranking = computeTaylorRanking(scores)

  result = newTensorDataZeros(scores.shape, scores.dtype)
  let pruneCount = int(scores.size.float * sparsity)

  # Set mask: 1 for kept weights, 0 for pruned
  case result.dtype
  of dtFloat32:
    let outArr = result.asFloat32
    # Initialize all to 1
    for i in 0..<result.size:
      outArr[i] = 1.0'f32
    # Zero out the lowest importance weights
    for i in 0..<pruneCount:
      outArr[ranking[i]] = 0.0'f32
  of dtFloat64:
    let outArr = result.asFloat64
    for i in 0..<result.size:
      outArr[i] = 1.0'f64
    for i in 0..<pruneCount:
      outArr[ranking[i]] = 0.0'f64
  else:
    discard

proc getMovementMask*(ctx: GradientPruningContext, name: string,
                      sparsity: float): TensorData =
  ## Get pruning mask based on movement scores
  let scores = ctx.movementTracker.getMovementImportance(name)
  if scores.isNil:
    return nil

  # Higher movement = more important = keep
  var indexed: seq[(float, int)] = @[]
  case scores.dtype
  of dtFloat32:
    let arr = scores.asFloat32
    for i in 0..<scores.size:
      indexed.add((arr[i].float, i))
  of dtFloat64:
    let arr = scores.asFloat64
    for i in 0..<scores.size:
      indexed.add((arr[i], i))
  else:
    discard

  # Sort by movement (ascending - prune lowest movement first)
  proc compareAsc(a, b: (float, int)): int = cmp(a[0], b[0])
  indexed.sort(compareAsc)

  result = newTensorDataZeros(scores.shape, scores.dtype)
  let pruneCount = int(scores.size.float * sparsity)

  case result.dtype
  of dtFloat32:
    let outArr = result.asFloat32
    for i in 0..<result.size:
      outArr[i] = 1.0'f32
    for i in 0..<pruneCount:
      outArr[indexed[i][1]] = 0.0'f32
  of dtFloat64:
    let outArr = result.asFloat64
    for i in 0..<result.size:
      outArr[i] = 1.0'f64
    for i in 0..<pruneCount:
      outArr[indexed[i][1]] = 0.0'f64
  else:
    discard

proc getEquilibriumMask*(ctx: GradientPruningContext, name: string,
                         sparsity: float): TensorData =
  ## Get pruning mask based on equilibrium state
  ctx.equilibriumState.getEquilibriumMask(name, sparsity)

# =============================================================================
# Training Loop with Gradient Pruning
# =============================================================================

type
  PruningTrainer* = ref object
    ## Trainer with integrated pruning support
    module*: GradientModule
    pruningCtx*: GradientPruningContext
    optimizer*: proc(params: seq[Parameter], grads: Table[string, TensorRef])
    pruningMethod*: string  # "taylor", "movement", "equilibrium"
    targetSparsity*: float
    pruneEveryNSteps*: int
    currentStep*: int

proc newPruningTrainer*(m: GradientModule, pruningMethod: string = "taylor",
                        targetSparsity: float = 0.5,
                        pruneEveryNSteps: int = 100): PruningTrainer =
  ## Create a pruning trainer
  result = PruningTrainer()
  result.module = m
  result.pruningCtx = newGradientPruningContext(m)
  result.pruningMethod = pruningMethod
  result.targetSparsity = targetSparsity
  result.pruneEveryNSteps = pruneEveryNSteps
  result.currentStep = 0
  result.pruningCtx.initForPruning()

proc step*(trainer: PruningTrainer, loss: TensorRef) =
  ## Perform a training step with pruning
  trainer.currentStep += 1

  # Compute gradients
  let grads = trainer.module.computeGradients(loss)

  # Update pruning state based on method
  case trainer.pruningMethod
  of "taylor":
    trainer.pruningCtx.computeTaylorScores(loss)
  of "movement":
    trainer.pruningCtx.updateMovement()
  of "equilibrium":
    trainer.pruningCtx.updateEquilibrium(loss, trainer.targetSparsity)
  else:
    discard

  # Apply gradients (if optimizer is set)
  if not trainer.optimizer.isNil:
    var paramSeq: seq[Parameter] = @[]
    for p in trainer.module.parameters.values:
      paramSeq.add(p)
    trainer.optimizer(paramSeq, grads)

  # Clear tape for next iteration
  trainer.module.clearTape()

proc getPruningMasks*(trainer: PruningTrainer): Table[string, TensorData] =
  ## Get current pruning masks for all parameters
  result = initTable[string, TensorData]()

  for name in trainer.module.parameters.keys:
    let mask = case trainer.pruningMethod
    of "taylor":
      trainer.pruningCtx.getTaylorMask(name, trainer.targetSparsity)
    of "movement":
      trainer.pruningCtx.getMovementMask(name, trainer.targetSparsity)
    of "equilibrium":
      trainer.pruningCtx.getEquilibriumMask(name, trainer.targetSparsity)
    else:
      nil

    if not mask.isNil:
      result[name] = mask

proc applyPruning*(trainer: PruningTrainer) =
  ## Apply pruning masks to parameters
  let masks = trainer.getPruningMasks()

  for name, mask in masks.pairs:
    if name in trainer.module.parameters:
      let param = trainer.module.parameters[name]

      # Apply mask: weight = weight * mask
      case param.data.dtype
      of dtFloat32:
        let wArr = param.data.asFloat32
        let mArr = mask.asFloat32
        for i in 0..<param.data.size:
          wArr[i] = wArr[i] * mArr[i]
      of dtFloat64:
        let wArr = param.data.asFloat64
        let mArr = mask.asFloat64
        for i in 0..<param.data.size:
          wArr[i] = wArr[i] * mArr[i]
      else:
        discard
