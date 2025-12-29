## Loss Functions
##
## Loss functions for training neural networks.

import std/[options, strformat]
import ml_core
import ./module

type
  Reduction* = enum
    ## Reduction mode for loss functions
    rNone     ## No reduction, return loss per element
    rMean     ## Mean of all losses
    rSum      ## Sum of all losses

  LossError* = object of CatchableError
    ## Error in loss computation

  Loss* = ref object of Module
    ## Base class for loss functions
    reduction*: Reduction

  # Classification losses

  CrossEntropyLoss* = ref object of Loss
    ## Cross-entropy loss for multi-class classification
    ignoreIndex*: Option[int]
    labelSmoothing*: float
    weight*: Option[TensorRef]  # Class weights

  NLLLoss* = ref object of Loss
    ## Negative log-likelihood loss
    ignoreIndex*: Option[int]
    weight*: Option[TensorRef]

  BCELoss* = ref object of Loss
    ## Binary cross-entropy loss
    weight*: Option[TensorRef]

  BCEWithLogitsLoss* = ref object of Loss
    ## BCE with sigmoid (numerically stable)
    weight*: Option[TensorRef]
    posWeight*: Option[TensorRef]

  # Regression losses

  MSELoss* = ref object of Loss
    ## Mean squared error (L2 loss)

  L1Loss* = ref object of Loss
    ## Mean absolute error (L1 loss)

  SmoothL1Loss* = ref object of Loss
    ## Smooth L1 loss (Huber loss)
    beta*: float

  HuberLoss* = ref object of Loss
    ## Huber loss
    delta*: float

  # Other losses

  KLDivLoss* = ref object of Loss
    ## Kullback-Leibler divergence
    logTarget*: bool

  MarginRankingLoss* = ref object of Loss
    ## Margin ranking loss
    margin*: float

  HingeEmbeddingLoss* = ref object of Loss
    ## Hinge embedding loss
    margin*: float

  CosineEmbeddingLoss* = ref object of Loss
    ## Cosine embedding loss
    margin*: float

  TripletMarginLoss* = ref object of Loss
    ## Triplet margin loss
    margin*: float
    p*: float  # Norm degree
    eps*: float
    swapEnabled*: bool

  MultiMarginLoss* = ref object of Loss
    ## Multi-class margin loss
    p*: int
    margin*: float
    weight*: Option[TensorRef]

  CTCLoss* = ref object of Loss
    ## Connectionist Temporal Classification loss
    blank*: int
    zeroInfinity*: bool

  PoissonNLLLoss* = ref object of Loss
    ## Negative log-likelihood for Poisson distribution
    logInput*: bool
    full*: bool
    eps*: float

# CrossEntropyLoss

proc newCrossEntropyLoss*(reduction: Reduction = rMean,
                          ignoreIndex: Option[int] = none(int),
                          labelSmoothing: float = 0.0,
                          weight: Option[TensorRef] = none(TensorRef)): CrossEntropyLoss =
  ## Create cross-entropy loss
  ##
  ## Args:
  ##   reduction: Reduction mode (mean, sum, none)
  ##   ignoreIndex: Index to ignore in loss computation (e.g., padding)
  ##   labelSmoothing: Smoothing factor (0.0 = no smoothing)
  ##   weight: Class weights for imbalanced data
  ##
  ## Input: (N, C) logits where C is number of classes
  ## Target: (N,) class indices
  result = CrossEntropyLoss()
  result.initModule("CrossEntropyLoss")
  result.reduction = reduction
  result.ignoreIndex = ignoreIndex
  result.labelSmoothing = labelSmoothing
  result.weight = weight

method forward*(l: CrossEntropyLoss, inputs: varargs[TensorRef]): TensorRef =
  ## Forward: -sum(target * log_softmax(input))
  if inputs.len < 2:
    raise newException(LossError, "CrossEntropyLoss requires input and target")

  let input = inputs[0]
  discard inputs[1]  # target - used in actual computation

  # Validate shapes
  if input.shape.dims.len < 1:
    raise newException(LossError, "CrossEntropyLoss input must have at least 1 dimension")

  # Output shape depends on reduction
  case l.reduction
  of rNone:
    # Return loss per sample
    if input.shape.dims.len >= 2:
      newTensorRef(newShape(input.shape.dims[0]), input.dtype)
    else:
      newTensorRef(newShape(1), input.dtype)
  of rMean, rSum:
    # Return scalar
    newTensorRef(newShape(1), input.dtype)

# NLLLoss

proc newNLLLoss*(reduction: Reduction = rMean,
                 ignoreIndex: Option[int] = none(int),
                 weight: Option[TensorRef] = none(TensorRef)): NLLLoss =
  ## Create negative log-likelihood loss
  ##
  ## Input: (N, C) log probabilities (from LogSoftmax)
  ## Target: (N,) class indices
  result = NLLLoss()
  result.initModule("NLLLoss")
  result.reduction = reduction
  result.ignoreIndex = ignoreIndex
  result.weight = weight

method forward*(l: NLLLoss, inputs: varargs[TensorRef]): TensorRef =
  ## Forward: -input[target]
  if inputs.len < 2:
    raise newException(LossError, "NLLLoss requires input and target")

  let input = inputs[0]

  case l.reduction
  of rNone:
    if input.shape.dims.len >= 2:
      newTensorRef(newShape(input.shape.dims[0]), input.dtype)
    else:
      newTensorRef(newShape(1), input.dtype)
  of rMean, rSum:
    newTensorRef(newShape(1), input.dtype)

# BCELoss

proc newBCELoss*(reduction: Reduction = rMean,
                 weight: Option[TensorRef] = none(TensorRef)): BCELoss =
  ## Create binary cross-entropy loss
  ##
  ## Input: (N, *) probabilities in [0, 1] (after sigmoid)
  ## Target: (N, *) binary labels {0, 1}
  result = BCELoss()
  result.initModule("BCELoss")
  result.reduction = reduction
  result.weight = weight

method forward*(l: BCELoss, inputs: varargs[TensorRef]): TensorRef =
  ## Forward: -weight * (target * log(input) + (1 - target) * log(1 - input))
  if inputs.len < 2:
    raise newException(LossError, "BCELoss requires input and target")

  let input = inputs[0]

  case l.reduction
  of rNone:
    newTensorRef(input.shape, input.dtype)
  of rMean, rSum:
    newTensorRef(newShape(1), input.dtype)

# BCEWithLogitsLoss

proc newBCEWithLogitsLoss*(reduction: Reduction = rMean,
                           weight: Option[TensorRef] = none(TensorRef),
                           posWeight: Option[TensorRef] = none(TensorRef)): BCEWithLogitsLoss =
  ## Create BCE with logits loss (sigmoid + BCE, numerically stable)
  ##
  ## Input: (N, *) raw logits
  ## Target: (N, *) binary labels {0, 1}
  result = BCEWithLogitsLoss()
  result.initModule("BCEWithLogitsLoss")
  result.reduction = reduction
  result.weight = weight
  result.posWeight = posWeight

method forward*(l: BCEWithLogitsLoss, inputs: varargs[TensorRef]): TensorRef =
  ## Forward: max(x, 0) - x * target + log(1 + exp(-abs(x)))
  if inputs.len < 2:
    raise newException(LossError, "BCEWithLogitsLoss requires input and target")

  let input = inputs[0]

  case l.reduction
  of rNone:
    newTensorRef(input.shape, input.dtype)
  of rMean, rSum:
    newTensorRef(newShape(1), input.dtype)

# MSELoss

proc newMSELoss*(reduction: Reduction = rMean): MSELoss =
  ## Create mean squared error loss
  ##
  ## Input: (N, *) predictions
  ## Target: (N, *) targets
  result = MSELoss()
  result.initModule("MSELoss")
  result.reduction = reduction

method forward*(l: MSELoss, inputs: varargs[TensorRef]): TensorRef =
  ## Forward: (input - target)^2
  if inputs.len < 2:
    raise newException(LossError, "MSELoss requires input and target")

  let input = inputs[0]

  case l.reduction
  of rNone:
    newTensorRef(input.shape, input.dtype)
  of rMean, rSum:
    newTensorRef(newShape(1), input.dtype)

# L1Loss

proc newL1Loss*(reduction: Reduction = rMean): L1Loss =
  ## Create L1 (mean absolute error) loss
  result = L1Loss()
  result.initModule("L1Loss")
  result.reduction = reduction

method forward*(l: L1Loss, inputs: varargs[TensorRef]): TensorRef =
  ## Forward: |input - target|
  if inputs.len < 2:
    raise newException(LossError, "L1Loss requires input and target")

  let input = inputs[0]

  case l.reduction
  of rNone:
    newTensorRef(input.shape, input.dtype)
  of rMean, rSum:
    newTensorRef(newShape(1), input.dtype)

# SmoothL1Loss

proc newSmoothL1Loss*(reduction: Reduction = rMean,
                      beta: float = 1.0): SmoothL1Loss =
  ## Create Smooth L1 loss (Huber-style)
  ##
  ## loss = 0.5 * (x - y)^2 / beta  if |x - y| < beta
  ##      = |x - y| - 0.5 * beta     otherwise
  result = SmoothL1Loss()
  result.initModule("SmoothL1Loss")
  result.reduction = reduction
  result.beta = beta

method forward*(l: SmoothL1Loss, inputs: varargs[TensorRef]): TensorRef =
  ## Forward: smooth L1 between input and target
  if inputs.len < 2:
    raise newException(LossError, "SmoothL1Loss requires input and target")

  let input = inputs[0]

  case l.reduction
  of rNone:
    newTensorRef(input.shape, input.dtype)
  of rMean, rSum:
    newTensorRef(newShape(1), input.dtype)

# HuberLoss

proc newHuberLoss*(reduction: Reduction = rMean,
                   delta: float = 1.0): HuberLoss =
  ## Create Huber loss
  ##
  ## loss = 0.5 * (x - y)^2              if |x - y| <= delta
  ##      = delta * (|x - y| - 0.5 * delta)  otherwise
  result = HuberLoss()
  result.initModule("HuberLoss")
  result.reduction = reduction
  result.delta = delta

method forward*(l: HuberLoss, inputs: varargs[TensorRef]): TensorRef =
  ## Forward: Huber loss between input and target
  if inputs.len < 2:
    raise newException(LossError, "HuberLoss requires input and target")

  let input = inputs[0]

  case l.reduction
  of rNone:
    newTensorRef(input.shape, input.dtype)
  of rMean, rSum:
    newTensorRef(newShape(1), input.dtype)

# KLDivLoss

proc newKLDivLoss*(reduction: Reduction = rMean,
                   logTarget: bool = false): KLDivLoss =
  ## Create KL divergence loss
  ##
  ## Input: (N, *) log probabilities
  ## Target: (N, *) probabilities (or log probs if logTarget=true)
  result = KLDivLoss()
  result.initModule("KLDivLoss")
  result.reduction = reduction
  result.logTarget = logTarget

method forward*(l: KLDivLoss, inputs: varargs[TensorRef]): TensorRef =
  ## Forward: target * (log(target) - input)
  if inputs.len < 2:
    raise newException(LossError, "KLDivLoss requires input and target")

  let input = inputs[0]

  case l.reduction
  of rNone:
    newTensorRef(input.shape, input.dtype)
  of rMean, rSum:
    newTensorRef(newShape(1), input.dtype)

# MarginRankingLoss

proc newMarginRankingLoss*(margin: float = 0.0,
                           reduction: Reduction = rMean): MarginRankingLoss =
  ## Create margin ranking loss
  ##
  ## loss = max(0, -y * (x1 - x2) + margin)
  result = MarginRankingLoss()
  result.initModule("MarginRankingLoss")
  result.reduction = reduction
  result.margin = margin

method forward*(l: MarginRankingLoss, inputs: varargs[TensorRef]): TensorRef =
  ## Forward: margin ranking loss
  if inputs.len < 3:
    raise newException(LossError, "MarginRankingLoss requires x1, x2, and y")

  let x1 = inputs[0]

  case l.reduction
  of rNone:
    newTensorRef(x1.shape, x1.dtype)
  of rMean, rSum:
    newTensorRef(newShape(1), x1.dtype)

# HingeEmbeddingLoss

proc newHingeEmbeddingLoss*(margin: float = 1.0,
                            reduction: Reduction = rMean): HingeEmbeddingLoss =
  ## Create hinge embedding loss
  ##
  ## loss = x                if y == 1
  ##      = max(0, margin - x) if y == -1
  result = HingeEmbeddingLoss()
  result.initModule("HingeEmbeddingLoss")
  result.reduction = reduction
  result.margin = margin

method forward*(l: HingeEmbeddingLoss, inputs: varargs[TensorRef]): TensorRef =
  ## Forward: hinge embedding loss
  if inputs.len < 2:
    raise newException(LossError, "HingeEmbeddingLoss requires input and target")

  let input = inputs[0]

  case l.reduction
  of rNone:
    newTensorRef(input.shape, input.dtype)
  of rMean, rSum:
    newTensorRef(newShape(1), input.dtype)

# CosineEmbeddingLoss

proc newCosineEmbeddingLoss*(margin: float = 0.0,
                             reduction: Reduction = rMean): CosineEmbeddingLoss =
  ## Create cosine embedding loss
  ##
  ## loss = 1 - cos(x1, x2)           if y == 1
  ##      = max(0, cos(x1, x2) - margin) if y == -1
  result = CosineEmbeddingLoss()
  result.initModule("CosineEmbeddingLoss")
  result.reduction = reduction
  result.margin = margin

method forward*(l: CosineEmbeddingLoss, inputs: varargs[TensorRef]): TensorRef =
  ## Forward: cosine embedding loss
  if inputs.len < 3:
    raise newException(LossError, "CosineEmbeddingLoss requires x1, x2, and y")

  let x1 = inputs[0]

  case l.reduction
  of rNone:
    newTensorRef(newShape(x1.shape.dims[0]), x1.dtype)
  of rMean, rSum:
    newTensorRef(newShape(1), x1.dtype)

# TripletMarginLoss

proc newTripletMarginLoss*(margin: float = 1.0,
                           p: float = 2.0,
                           eps: float = 1e-6,
                           swap: bool = false,
                           reduction: Reduction = rMean): TripletMarginLoss =
  ## Create triplet margin loss
  ##
  ## loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
  result = TripletMarginLoss()
  result.initModule("TripletMarginLoss")
  result.reduction = reduction
  result.margin = margin
  result.p = p
  result.eps = eps
  result.swapEnabled = swap

method forward*(l: TripletMarginLoss, inputs: varargs[TensorRef]): TensorRef =
  ## Forward: triplet margin loss
  if inputs.len < 3:
    raise newException(LossError, "TripletMarginLoss requires anchor, positive, and negative")

  let anchor = inputs[0]

  case l.reduction
  of rNone:
    newTensorRef(newShape(anchor.shape.dims[0]), anchor.dtype)
  of rMean, rSum:
    newTensorRef(newShape(1), anchor.dtype)

# MultiMarginLoss

proc newMultiMarginLoss*(p: int = 1,
                         margin: float = 1.0,
                         weight: Option[TensorRef] = none(TensorRef),
                         reduction: Reduction = rMean): MultiMarginLoss =
  ## Create multi-class margin loss
  result = MultiMarginLoss()
  result.initModule("MultiMarginLoss")
  result.reduction = reduction
  result.p = p
  result.margin = margin
  result.weight = weight

method forward*(l: MultiMarginLoss, inputs: varargs[TensorRef]): TensorRef =
  ## Forward: multi-margin loss
  if inputs.len < 2:
    raise newException(LossError, "MultiMarginLoss requires input and target")

  let input = inputs[0]

  case l.reduction
  of rNone:
    newTensorRef(newShape(input.shape.dims[0]), input.dtype)
  of rMean, rSum:
    newTensorRef(newShape(1), input.dtype)

# CTCLoss

proc newCTCLoss*(blank: int = 0,
                 zeroInfinity: bool = false,
                 reduction: Reduction = rMean): CTCLoss =
  ## Create CTC (Connectionist Temporal Classification) loss
  ##
  ## Used for sequence-to-sequence with unknown alignment
  result = CTCLoss()
  result.initModule("CTCLoss")
  result.reduction = reduction
  result.blank = blank
  result.zeroInfinity = zeroInfinity

method forward*(l: CTCLoss, inputs: varargs[TensorRef]): TensorRef =
  ## Forward: CTC loss
  ## Inputs: log_probs (T, N, C), targets, input_lengths, target_lengths
  if inputs.len < 4:
    raise newException(LossError,
      "CTCLoss requires log_probs, targets, input_lengths, target_lengths")

  let logProbs = inputs[0]

  case l.reduction
  of rNone:
    if logProbs.shape.dims.len >= 2:
      newTensorRef(newShape(logProbs.shape.dims[1]), logProbs.dtype)
    else:
      newTensorRef(newShape(1), logProbs.dtype)
  of rMean, rSum:
    newTensorRef(newShape(1), logProbs.dtype)

# PoissonNLLLoss

proc newPoissonNLLLoss*(logInput: bool = true,
                        full: bool = false,
                        eps: float = 1e-8,
                        reduction: Reduction = rMean): PoissonNLLLoss =
  ## Create Poisson NLL loss
  result = PoissonNLLLoss()
  result.initModule("PoissonNLLLoss")
  result.reduction = reduction
  result.logInput = logInput
  result.full = full
  result.eps = eps

method forward*(l: PoissonNLLLoss, inputs: varargs[TensorRef]): TensorRef =
  ## Forward: Poisson negative log-likelihood
  if inputs.len < 2:
    raise newException(LossError, "PoissonNLLLoss requires input and target")

  let input = inputs[0]

  case l.reduction
  of rNone:
    newTensorRef(input.shape, input.dtype)
  of rMean, rSum:
    newTensorRef(newShape(1), input.dtype)

# Extra repr

proc extraRepr*(l: Loss): string =
  ## Base extra repr
  fmt"reduction={l.reduction}"

proc extraRepr*(l: CrossEntropyLoss): string =
  result = fmt"reduction={l.reduction}"
  if l.ignoreIndex.isSome:
    result &= fmt", ignore_index={l.ignoreIndex.get}"
  if l.labelSmoothing > 0.0:
    result &= fmt", label_smoothing={l.labelSmoothing}"

proc extraRepr*(l: SmoothL1Loss): string =
  fmt"reduction={l.reduction}, beta={l.beta}"

proc extraRepr*(l: HuberLoss): string =
  fmt"reduction={l.reduction}, delta={l.delta}"

proc extraRepr*(l: TripletMarginLoss): string =
  result = fmt"margin={l.margin}, p={l.p}"
  if l.swapEnabled:
    result &= ", swap=True"
