## Optimizers and Learning Rate Schedulers
##
## Optimization algorithms for training neural networks.

import std/[options, tables, hashes, math, strformat]
import ml_core
import ./module

type
  OptimizerError* = object of CatchableError
    ## Error in optimizer operation

  OptimizerState* = object
    ## State for a single parameter
    step*: int
    tensors*: Table[string, TensorData]

  ParamGroup* = object
    ## Group of parameters with shared options
    params*: seq[Parameter]
    options*: Table[string, float]

  Optimizer* = ref object of RootObj
    ## Base optimizer class
    paramGroups*: seq[ParamGroup]
    defaults*: Table[string, float]
    state*: Table[string, OptimizerState]  # keyed by param name

  # SGD optimizer
  SGD* = ref object of Optimizer
    lr*: float
    momentum*: float
    dampening*: float
    weightDecay*: float
    nesterov*: bool

  # Adam optimizer
  Adam* = ref object of Optimizer
    lr*: float
    beta1*: float
    beta2*: float
    eps*: float
    weightDecay*: float
    amsgrad*: bool

  # AdamW optimizer (decoupled weight decay)
  AdamW* = ref object of Optimizer
    lr*: float
    beta1*: float
    beta2*: float
    eps*: float
    weightDecay*: float

  # Adagrad optimizer
  Adagrad* = ref object of Optimizer
    lr*: float
    lrDecay*: float
    weightDecay*: float
    initialAccumulatorValue*: float
    eps*: float

  # RMSprop optimizer
  RMSprop* = ref object of Optimizer
    lr*: float
    alpha*: float
    eps*: float
    weightDecay*: float
    momentum*: float
    centered*: bool

  # Adadelta optimizer
  Adadelta* = ref object of Optimizer
    lr*: float
    rho*: float
    eps*: float
    weightDecay*: float

  # Learning rate schedulers
  LRScheduler* = ref object of RootObj
    ## Base class for learning rate schedulers
    optimizer*: Optimizer
    lastEpoch*: int
    baseLRs*: seq[float]
    verbose*: bool

  StepLR* = ref object of LRScheduler
    ## Decay LR by gamma every stepSize epochs
    stepSize*: int
    gamma*: float

  MultiStepLR* = ref object of LRScheduler
    ## Decay LR at specified milestones
    milestones*: seq[int]
    gamma*: float

  ExponentialLR* = ref object of LRScheduler
    ## Decay LR by gamma every epoch
    gamma*: float

  CosineAnnealingLR* = ref object of LRScheduler
    ## Cosine annealing schedule
    tMax*: int
    etaMin*: float

  CosineAnnealingWarmRestarts* = ref object of LRScheduler
    ## Cosine annealing with warm restarts
    t0*: int
    tMult*: int
    etaMin*: float
    tCur*: int

  LinearLR* = ref object of LRScheduler
    ## Linear warmup/cooldown
    startFactor*: float
    endFactor*: float
    totalIters*: int

  PolynomialLR* = ref object of LRScheduler
    ## Polynomial decay
    totalIters*: int
    power*: float

  OneCycleLR* = ref object of LRScheduler
    ## One cycle policy
    maxLR*: float
    totalSteps*: int
    pctStart*: float
    annealStrategy*: string
    divFactor*: float
    finalDivFactor*: float

  ReduceLROnPlateau* = ref object of LRScheduler
    ## Reduce LR when metric stops improving
    mode*: string  # "min" or "max"
    factor*: float
    patience*: int
    threshold*: float
    thresholdMode*: string  # "rel" or "abs"
    cooldown*: int
    minLR*: float
    eps*: float
    # Internal state
    best*: float
    numBadEpochs*: int
    cooldownCounter*: int

# Helper to get param identifier
proc paramId(param: Parameter): string =
  param.name

# Optimizer base methods

proc initOptimizer*(o: Optimizer, params: seq[Parameter],
                    defaults: Table[string, float] = initTable[string, float]()) =
  ## Initialize optimizer with parameters
  o.paramGroups = @[ParamGroup(params: params, options: initTable[string, float]())]
  o.defaults = defaults
  o.state = initTable[string, OptimizerState]()

proc addParamGroup*(o: Optimizer, params: seq[Parameter],
                    options: Table[string, float] = initTable[string, float]()) =
  ## Add a parameter group with custom options
  o.paramGroups.add(ParamGroup(params: params, options: options))

proc getState*(o: Optimizer, param: Parameter): var OptimizerState =
  ## Get optimizer state for a parameter
  let id = paramId(param)
  if id notin o.state:
    o.state[id] = OptimizerState(step: 0, tensors: initTable[string, TensorData]())
  o.state[id]

method zeroGrad*(o: Optimizer) {.base.} =
  ## Zero all parameter gradients
  for i in 0..<o.paramGroups.len:
    for j in 0..<o.paramGroups[i].params.len:
      o.paramGroups[i].params[j].gradRef = none(TensorRef)

method step*(o: Optimizer) {.base.} =
  ## Perform optimization step - must be implemented by subclasses
  raise newException(OptimizerError, "step() not implemented")

proc getLR*(o: Optimizer): float =
  ## Get current learning rate
  if o.paramGroups.len > 0:
    if "lr" in o.paramGroups[0].options:
      return o.paramGroups[0].options["lr"]
  if "lr" in o.defaults:
    return o.defaults["lr"]
  0.0

proc setLR*(o: Optimizer, lr: float) =
  ## Set learning rate for all parameter groups
  for i in 0..<o.paramGroups.len:
    o.paramGroups[i].options["lr"] = lr

# SGD

proc newSGD*(params: seq[Parameter],
             lr: float,
             momentum: float = 0.0,
             dampening: float = 0.0,
             weightDecay: float = 0.0,
             nesterov: bool = false): SGD =
  ## Create SGD optimizer
  ##
  ## Args:
  ##   params: Parameters to optimize
  ##   lr: Learning rate
  ##   momentum: Momentum factor (default 0)
  ##   dampening: Dampening for momentum (default 0)
  ##   weightDecay: L2 penalty (default 0)
  ##   nesterov: Use Nesterov momentum (default false)
  if nesterov and (momentum <= 0 or dampening != 0):
    raise newException(OptimizerError,
      "Nesterov momentum requires momentum > 0 and dampening = 0")

  result = SGD()
  result.lr = lr
  result.momentum = momentum
  result.dampening = dampening
  result.weightDecay = weightDecay
  result.nesterov = nesterov

  var defaults = initTable[string, float]()
  defaults["lr"] = lr
  defaults["momentum"] = momentum
  defaults["dampening"] = dampening
  defaults["weight_decay"] = weightDecay
  result.initOptimizer(params, defaults)

method step*(o: SGD) =
  ## SGD optimization step
  ##
  ## v = momentum * v + (1 - dampening) * grad
  ## if nesterov: param = param - lr * (grad + momentum * v)
  ## else: param = param - lr * v
  for group in o.paramGroups:
    for param in group.params:
      if param.gradRef.isNone:
        continue

      var state = o.getState(param)
      state.step += 1

      # Actual update would be performed by executor
      # This is the optimization logic placeholder

# Adam

proc newAdam*(params: seq[Parameter],
              lr: float = 0.001,
              betas: (float, float) = (0.9, 0.999),
              eps: float = 1e-8,
              weightDecay: float = 0.0,
              amsgrad: bool = false): Adam =
  ## Create Adam optimizer
  ##
  ## Args:
  ##   params: Parameters to optimize
  ##   lr: Learning rate (default 0.001)
  ##   betas: Coefficients for running averages (default (0.9, 0.999))
  ##   eps: Term for numerical stability (default 1e-8)
  ##   weightDecay: L2 penalty (default 0)
  ##   amsgrad: Use AMSGrad variant (default false)
  result = Adam()
  result.lr = lr
  result.beta1 = betas[0]
  result.beta2 = betas[1]
  result.eps = eps
  result.weightDecay = weightDecay
  result.amsgrad = amsgrad

  var defaults = initTable[string, float]()
  defaults["lr"] = lr
  defaults["beta1"] = betas[0]
  defaults["beta2"] = betas[1]
  defaults["eps"] = eps
  defaults["weight_decay"] = weightDecay
  result.initOptimizer(params, defaults)

method step*(o: Adam) =
  ## Adam optimization step
  ##
  ## m = beta1 * m + (1 - beta1) * grad
  ## v = beta2 * v + (1 - beta2) * grad^2
  ## m_hat = m / (1 - beta1^t)
  ## v_hat = v / (1 - beta2^t)
  ## param = param - lr * m_hat / (sqrt(v_hat) + eps)
  for group in o.paramGroups:
    for param in group.params:
      if param.gradRef.isNone:
        continue

      var state = o.getState(param)
      state.step += 1

# AdamW

proc newAdamW*(params: seq[Parameter],
               lr: float = 0.001,
               betas: (float, float) = (0.9, 0.999),
               eps: float = 1e-8,
               weightDecay: float = 0.01): AdamW =
  ## Create AdamW optimizer (Adam with decoupled weight decay)
  ##
  ## Unlike Adam, weight decay is applied separately from gradient update.
  result = AdamW()
  result.lr = lr
  result.beta1 = betas[0]
  result.beta2 = betas[1]
  result.eps = eps
  result.weightDecay = weightDecay

  var defaults = initTable[string, float]()
  defaults["lr"] = lr
  defaults["beta1"] = betas[0]
  defaults["beta2"] = betas[1]
  defaults["eps"] = eps
  defaults["weight_decay"] = weightDecay
  result.initOptimizer(params, defaults)

method step*(o: AdamW) =
  ## AdamW optimization step
  ##
  ## Same as Adam but with decoupled weight decay:
  ## param = param - lr * weight_decay * param  (before Adam update)
  for group in o.paramGroups:
    for param in group.params:
      if param.gradRef.isNone:
        continue

      var state = o.getState(param)
      state.step += 1

# Adagrad

proc newAdagrad*(params: seq[Parameter],
                 lr: float = 0.01,
                 lrDecay: float = 0.0,
                 weightDecay: float = 0.0,
                 initialAccumulatorValue: float = 0.0,
                 eps: float = 1e-10): Adagrad =
  ## Create Adagrad optimizer
  result = Adagrad()
  result.lr = lr
  result.lrDecay = lrDecay
  result.weightDecay = weightDecay
  result.initialAccumulatorValue = initialAccumulatorValue
  result.eps = eps

  var defaults = initTable[string, float]()
  defaults["lr"] = lr
  result.initOptimizer(params, defaults)

method step*(o: Adagrad) =
  ## Adagrad optimization step
  for group in o.paramGroups:
    for param in group.params:
      if param.gradRef.isNone:
        continue

      var state = o.getState(param)
      state.step += 1

# RMSprop

proc newRMSprop*(params: seq[Parameter],
                 lr: float = 0.01,
                 alpha: float = 0.99,
                 eps: float = 1e-8,
                 weightDecay: float = 0.0,
                 momentum: float = 0.0,
                 centered: bool = false): RMSprop =
  ## Create RMSprop optimizer
  result = RMSprop()
  result.lr = lr
  result.alpha = alpha
  result.eps = eps
  result.weightDecay = weightDecay
  result.momentum = momentum
  result.centered = centered

  var defaults = initTable[string, float]()
  defaults["lr"] = lr
  result.initOptimizer(params, defaults)

method step*(o: RMSprop) =
  ## RMSprop optimization step
  for group in o.paramGroups:
    for param in group.params:
      if param.gradRef.isNone:
        continue

      var state = o.getState(param)
      state.step += 1

# Adadelta

proc newAdadelta*(params: seq[Parameter],
                  lr: float = 1.0,
                  rho: float = 0.9,
                  eps: float = 1e-6,
                  weightDecay: float = 0.0): Adadelta =
  ## Create Adadelta optimizer
  result = Adadelta()
  result.lr = lr
  result.rho = rho
  result.eps = eps
  result.weightDecay = weightDecay

  var defaults = initTable[string, float]()
  defaults["lr"] = lr
  result.initOptimizer(params, defaults)

method step*(o: Adadelta) =
  ## Adadelta optimization step
  for group in o.paramGroups:
    for param in group.params:
      if param.gradRef.isNone:
        continue

      var state = o.getState(param)
      state.step += 1

# LR Scheduler base

proc initScheduler*(s: LRScheduler, optimizer: Optimizer, lastEpoch: int = -1) =
  ## Initialize scheduler
  s.optimizer = optimizer
  s.lastEpoch = lastEpoch
  s.baseLRs = @[]

  # Capture base LRs
  for group in optimizer.paramGroups:
    if "lr" in group.options:
      s.baseLRs.add(group.options["lr"])
    elif "lr" in optimizer.defaults:
      s.baseLRs.add(optimizer.defaults["lr"])
    else:
      s.baseLRs.add(0.0)

method getLR*(s: LRScheduler): seq[float] {.base.} =
  ## Get current LRs for each param group
  s.baseLRs

method step*(s: LRScheduler) {.base.} =
  ## Advance scheduler by one epoch
  s.lastEpoch += 1
  let lrs = s.getLR()
  for i, lr in lrs:
    if i < s.optimizer.paramGroups.len:
      s.optimizer.paramGroups[i].options["lr"] = lr

# StepLR

proc newStepLR*(optimizer: Optimizer,
                stepSize: int,
                gamma: float = 0.1,
                lastEpoch: int = -1): StepLR =
  ## Create StepLR scheduler
  ##
  ## Decays LR by gamma every stepSize epochs.
  result = StepLR()
  result.stepSize = stepSize
  result.gamma = gamma
  result.initScheduler(optimizer, lastEpoch)

method getLR*(s: StepLR): seq[float] =
  ## Get current LRs
  let factor = pow(s.gamma, float(s.lastEpoch div s.stepSize))
  for baseLR in s.baseLRs:
    result.add(baseLR * factor)

# MultiStepLR

proc newMultiStepLR*(optimizer: Optimizer,
                     milestones: seq[int],
                     gamma: float = 0.1,
                     lastEpoch: int = -1): MultiStepLR =
  ## Create MultiStepLR scheduler
  ##
  ## Decays LR by gamma at each milestone.
  result = MultiStepLR()
  result.milestones = milestones
  result.gamma = gamma
  result.initScheduler(optimizer, lastEpoch)

method getLR*(s: MultiStepLR): seq[float] =
  ## Get current LRs
  var numDecays = 0
  for m in s.milestones:
    if s.lastEpoch >= m:
      numDecays += 1

  let factor = pow(s.gamma, float(numDecays))
  for baseLR in s.baseLRs:
    result.add(baseLR * factor)

# ExponentialLR

proc newExponentialLR*(optimizer: Optimizer,
                       gamma: float,
                       lastEpoch: int = -1): ExponentialLR =
  ## Create ExponentialLR scheduler
  ##
  ## Decays LR by gamma every epoch.
  result = ExponentialLR()
  result.gamma = gamma
  result.initScheduler(optimizer, lastEpoch)

method getLR*(s: ExponentialLR): seq[float] =
  ## Get current LRs
  let factor = pow(s.gamma, float(s.lastEpoch))
  for baseLR in s.baseLRs:
    result.add(baseLR * factor)

# CosineAnnealingLR

proc newCosineAnnealingLR*(optimizer: Optimizer,
                           tMax: int,
                           etaMin: float = 0.0,
                           lastEpoch: int = -1): CosineAnnealingLR =
  ## Create CosineAnnealingLR scheduler
  ##
  ## LR follows cosine curve from base to etaMin over tMax epochs.
  result = CosineAnnealingLR()
  result.tMax = tMax
  result.etaMin = etaMin
  result.initScheduler(optimizer, lastEpoch)

method getLR*(s: CosineAnnealingLR): seq[float] =
  ## Get current LRs using cosine annealing
  let progress = float(s.lastEpoch) / float(s.tMax)
  let cosValue = (1.0 + cos(PI * progress)) / 2.0

  for baseLR in s.baseLRs:
    result.add(s.etaMin + (baseLR - s.etaMin) * cosValue)

# CosineAnnealingWarmRestarts

proc newCosineAnnealingWarmRestarts*(optimizer: Optimizer,
                                     t0: int,
                                     tMult: int = 1,
                                     etaMin: float = 0.0,
                                     lastEpoch: int = -1): CosineAnnealingWarmRestarts =
  ## Create CosineAnnealingWarmRestarts scheduler
  ##
  ## Cosine annealing with warm restarts.
  result = CosineAnnealingWarmRestarts()
  result.t0 = t0
  result.tMult = tMult
  result.etaMin = etaMin
  result.tCur = 0
  result.initScheduler(optimizer, lastEpoch)

method getLR*(s: CosineAnnealingWarmRestarts): seq[float] =
  ## Get current LRs
  let progress = float(s.tCur) / float(s.t0)
  let cosValue = (1.0 + cos(PI * progress)) / 2.0

  for baseLR in s.baseLRs:
    result.add(s.etaMin + (baseLR - s.etaMin) * cosValue)

method step*(s: CosineAnnealingWarmRestarts) =
  ## Step with restart logic
  s.lastEpoch += 1
  s.tCur += 1

  if s.tCur >= s.t0:
    s.tCur = 0
    # Optionally increase period by tMult

  let lrs = s.getLR()
  for i, lr in lrs:
    if i < s.optimizer.paramGroups.len:
      s.optimizer.paramGroups[i].options["lr"] = lr

# LinearLR

proc newLinearLR*(optimizer: Optimizer,
                  startFactor: float = 1.0/3.0,
                  endFactor: float = 1.0,
                  totalIters: int = 5,
                  lastEpoch: int = -1): LinearLR =
  ## Create LinearLR scheduler
  ##
  ## Linear warmup from startFactor to endFactor over totalIters.
  result = LinearLR()
  result.startFactor = startFactor
  result.endFactor = endFactor
  result.totalIters = totalIters
  result.initScheduler(optimizer, lastEpoch)

method getLR*(s: LinearLR): seq[float] =
  ## Get current LRs
  let progress = min(float(s.lastEpoch) / float(s.totalIters), 1.0)
  let factor = s.startFactor + (s.endFactor - s.startFactor) * progress

  for baseLR in s.baseLRs:
    result.add(baseLR * factor)

# PolynomialLR

proc newPolynomialLR*(optimizer: Optimizer,
                      totalIters: int,
                      power: float = 1.0,
                      lastEpoch: int = -1): PolynomialLR =
  ## Create PolynomialLR scheduler
  ##
  ## Polynomial decay.
  result = PolynomialLR()
  result.totalIters = totalIters
  result.power = power
  result.initScheduler(optimizer, lastEpoch)

method getLR*(s: PolynomialLR): seq[float] =
  ## Get current LRs
  let progress = min(float(s.lastEpoch) / float(s.totalIters), 1.0)
  let factor = pow(1.0 - progress, s.power)

  for baseLR in s.baseLRs:
    result.add(baseLR * factor)

# OneCycleLR

proc newOneCycleLR*(optimizer: Optimizer,
                    maxLR: float,
                    totalSteps: int,
                    pctStart: float = 0.3,
                    annealStrategy: string = "cos",
                    divFactor: float = 25.0,
                    finalDivFactor: float = 1e4,
                    lastEpoch: int = -1): OneCycleLR =
  ## Create OneCycleLR scheduler
  ##
  ## One cycle learning rate policy.
  result = OneCycleLR()
  result.maxLR = maxLR
  result.totalSteps = totalSteps
  result.pctStart = pctStart
  result.annealStrategy = annealStrategy
  result.divFactor = divFactor
  result.finalDivFactor = finalDivFactor
  result.initScheduler(optimizer, lastEpoch)

method getLR*(s: OneCycleLR): seq[float] =
  ## Get current LRs following one cycle policy
  let progress = float(s.lastEpoch) / float(s.totalSteps)
  let warmupEnd = s.pctStart
  let initialLR = s.maxLR / s.divFactor
  let minLR = initialLR / s.finalDivFactor

  var lr: float
  if progress < warmupEnd:
    # Warmup phase
    let warmupProgress = progress / warmupEnd
    lr = initialLR + (s.maxLR - initialLR) * warmupProgress
  else:
    # Annealing phase
    let annealProgress = (progress - warmupEnd) / (1.0 - warmupEnd)
    if s.annealStrategy == "cos":
      lr = minLR + (s.maxLR - minLR) * (1.0 + cos(PI * annealProgress)) / 2.0
    else:
      lr = s.maxLR - (s.maxLR - minLR) * annealProgress

  for _ in s.baseLRs:
    result.add(lr)

# ReduceLROnPlateau

proc newReduceLROnPlateau*(optimizer: Optimizer,
                           mode: string = "min",
                           factor: float = 0.1,
                           patience: int = 10,
                           threshold: float = 1e-4,
                           thresholdMode: string = "rel",
                           cooldown: int = 0,
                           minLR: float = 0.0,
                           eps: float = 1e-8): ReduceLROnPlateau =
  ## Create ReduceLROnPlateau scheduler
  ##
  ## Reduce LR when a metric has stopped improving.
  result = ReduceLROnPlateau()
  result.mode = mode
  result.factor = factor
  result.patience = patience
  result.threshold = threshold
  result.thresholdMode = thresholdMode
  result.cooldown = cooldown
  result.minLR = minLR
  result.eps = eps
  result.best = if mode == "min": Inf else: NegInf
  result.numBadEpochs = 0
  result.cooldownCounter = 0
  result.initScheduler(optimizer)

proc isBetter(s: ReduceLROnPlateau, current: float): bool =
  ## Check if current metric is better than best
  if s.mode == "min":
    if s.thresholdMode == "rel":
      return current < s.best * (1.0 - s.threshold)
    else:
      return current < s.best - s.threshold
  else:
    if s.thresholdMode == "rel":
      return current > s.best * (1.0 + s.threshold)
    else:
      return current > s.best + s.threshold

proc step*(s: ReduceLROnPlateau, metric: float) =
  ## Step with metric value
  s.lastEpoch += 1

  if s.cooldownCounter > 0:
    s.cooldownCounter -= 1
    s.numBadEpochs = 0
    return

  if s.isBetter(metric):
    s.best = metric
    s.numBadEpochs = 0
  else:
    s.numBadEpochs += 1

  if s.numBadEpochs > s.patience:
    # Reduce LR
    for i in 0..<s.optimizer.paramGroups.len:
      let oldLR = s.optimizer.paramGroups[i].options.getOrDefault("lr", s.baseLRs[i])
      let newLR = max(oldLR * s.factor, s.minLR)
      if oldLR - newLR > s.eps:
        s.optimizer.paramGroups[i].options["lr"] = newLR

    s.cooldownCounter = s.cooldown
    s.numBadEpochs = 0

# Extra repr for optimizers

proc extraRepr*(o: SGD): string =
  result = fmt"lr={o.lr}"
  if o.momentum > 0:
    result &= fmt", momentum={o.momentum}"
  if o.weightDecay > 0:
    result &= fmt", weight_decay={o.weightDecay}"
  if o.nesterov:
    result &= ", nesterov=True"

proc extraRepr*(o: Adam): string =
  result = fmt"lr={o.lr}, betas=({o.beta1}, {o.beta2}), eps={o.eps}"
  if o.weightDecay > 0:
    result &= fmt", weight_decay={o.weightDecay}"
  if o.amsgrad:
    result &= ", amsgrad=True"

proc extraRepr*(o: AdamW): string =
  fmt"lr={o.lr}, betas=({o.beta1}, {o.beta2}), eps={o.eps}, weight_decay={o.weightDecay}"
