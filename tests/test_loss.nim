## Tests for loss functions

import unittest
import std/[options, strutils]
import nimml_core
import ../src/nimml_nn/module
import ../src/nimml_nn/loss

suite "CrossEntropyLoss":
  test "create cross entropy loss":
    let loss = newCrossEntropyLoss()
    check loss.reduction == rMean
    check loss.ignoreIndex.isNone
    check loss.labelSmoothing == 0.0

  test "cross entropy with options":
    let loss = newCrossEntropyLoss(
      reduction = rSum,
      ignoreIndex = some(-100),
      labelSmoothing = 0.1
    )
    check loss.reduction == rSum
    check loss.ignoreIndex.get == -100
    check loss.labelSmoothing == 0.1

  test "cross entropy forward scalar output":
    let loss = newCrossEntropyLoss(reduction = rMean)
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let target = newTensorRef(newShape(32), dtInt64)
    let output = loss.forward(input, target)
    check output.shape == newShape(1)

  test "cross entropy forward no reduction":
    let loss = newCrossEntropyLoss(reduction = rNone)
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let target = newTensorRef(newShape(32), dtInt64)
    let output = loss.forward(input, target)
    check output.shape == newShape(32)

  test "cross entropy missing inputs":
    let loss = newCrossEntropyLoss()
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    expect(LossError):
      discard loss.forward(input)

  test "cross entropy extraRepr":
    let loss = newCrossEntropyLoss(labelSmoothing = 0.1)
    let repr = loss.extraRepr
    check "label_smoothing=0.1" in repr

suite "NLLLoss":
  test "create nll loss":
    let loss = newNLLLoss()
    check loss.reduction == rMean

  test "nll loss forward":
    let loss = newNLLLoss()
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let target = newTensorRef(newShape(32), dtInt64)
    let output = loss.forward(input, target)
    check output.shape == newShape(1)

suite "BCELoss":
  test "create bce loss":
    let loss = newBCELoss()
    check loss.reduction == rMean

  test "bce loss forward":
    let loss = newBCELoss()
    let input = newTensorRef(newShape(32, 1), dtFloat32)
    let target = newTensorRef(newShape(32, 1), dtFloat32)
    let output = loss.forward(input, target)
    check output.shape == newShape(1)

  test "bce loss no reduction":
    let loss = newBCELoss(reduction = rNone)
    let input = newTensorRef(newShape(32, 1), dtFloat32)
    let target = newTensorRef(newShape(32, 1), dtFloat32)
    let output = loss.forward(input, target)
    check output.shape == newShape(32, 1)

suite "BCEWithLogitsLoss":
  test "create bce with logits":
    let loss = newBCEWithLogitsLoss()
    check loss.reduction == rMean

  test "bce with logits forward":
    let loss = newBCEWithLogitsLoss()
    let input = newTensorRef(newShape(32, 1), dtFloat32)
    let target = newTensorRef(newShape(32, 1), dtFloat32)
    let output = loss.forward(input, target)
    check output.shape == newShape(1)

suite "MSELoss":
  test "create mse loss":
    let loss = newMSELoss()
    check loss.reduction == rMean

  test "mse loss forward":
    let loss = newMSELoss()
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let target = newTensorRef(newShape(32, 10), dtFloat32)
    let output = loss.forward(input, target)
    check output.shape == newShape(1)

  test "mse loss no reduction":
    let loss = newMSELoss(reduction = rNone)
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let target = newTensorRef(newShape(32, 10), dtFloat32)
    let output = loss.forward(input, target)
    check output.shape == newShape(32, 10)

  test "mse loss sum reduction":
    let loss = newMSELoss(reduction = rSum)
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let target = newTensorRef(newShape(32, 10), dtFloat32)
    let output = loss.forward(input, target)
    check output.shape == newShape(1)

suite "L1Loss":
  test "create l1 loss":
    let loss = newL1Loss()
    check loss.reduction == rMean

  test "l1 loss forward":
    let loss = newL1Loss()
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let target = newTensorRef(newShape(32, 10), dtFloat32)
    let output = loss.forward(input, target)
    check output.shape == newShape(1)

suite "SmoothL1Loss":
  test "create smooth l1 loss":
    let loss = newSmoothL1Loss()
    check loss.beta == 1.0

  test "smooth l1 custom beta":
    let loss = newSmoothL1Loss(beta = 0.5)
    check loss.beta == 0.5

  test "smooth l1 forward":
    let loss = newSmoothL1Loss()
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let target = newTensorRef(newShape(32, 10), dtFloat32)
    let output = loss.forward(input, target)
    check output.shape == newShape(1)

  test "smooth l1 extraRepr":
    let loss = newSmoothL1Loss(beta = 0.5)
    check "beta=0.5" in loss.extraRepr

suite "HuberLoss":
  test "create huber loss":
    let loss = newHuberLoss()
    check loss.delta == 1.0

  test "huber loss forward":
    let loss = newHuberLoss()
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let target = newTensorRef(newShape(32, 10), dtFloat32)
    let output = loss.forward(input, target)
    check output.shape == newShape(1)

  test "huber loss extraRepr":
    let loss = newHuberLoss(delta = 2.0)
    check "delta=2.0" in loss.extraRepr

suite "KLDivLoss":
  test "create kl div loss":
    let loss = newKLDivLoss()
    check not loss.logTarget

  test "kl div loss forward":
    let loss = newKLDivLoss()
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let target = newTensorRef(newShape(32, 10), dtFloat32)
    let output = loss.forward(input, target)
    check output.shape == newShape(1)

suite "MarginRankingLoss":
  test "create margin ranking loss":
    let loss = newMarginRankingLoss()
    check loss.margin == 0.0

  test "margin ranking forward":
    let loss = newMarginRankingLoss(margin = 1.0)
    let x1 = newTensorRef(newShape(32), dtFloat32)
    let x2 = newTensorRef(newShape(32), dtFloat32)
    let y = newTensorRef(newShape(32), dtFloat32)
    let output = loss.forward(x1, x2, y)
    check output.shape == newShape(1)

suite "HingeEmbeddingLoss":
  test "create hinge embedding loss":
    let loss = newHingeEmbeddingLoss()
    check loss.margin == 1.0

  test "hinge embedding forward":
    let loss = newHingeEmbeddingLoss()
    let input = newTensorRef(newShape(32), dtFloat32)
    let target = newTensorRef(newShape(32), dtFloat32)
    let output = loss.forward(input, target)
    check output.shape == newShape(1)

suite "CosineEmbeddingLoss":
  test "create cosine embedding loss":
    let loss = newCosineEmbeddingLoss()
    check loss.margin == 0.0

  test "cosine embedding forward":
    let loss = newCosineEmbeddingLoss()
    let x1 = newTensorRef(newShape(32, 128), dtFloat32)
    let x2 = newTensorRef(newShape(32, 128), dtFloat32)
    let y = newTensorRef(newShape(32), dtFloat32)
    let output = loss.forward(x1, x2, y)
    check output.shape == newShape(1)

suite "TripletMarginLoss":
  test "create triplet margin loss":
    let loss = newTripletMarginLoss()
    check loss.margin == 1.0
    check loss.p == 2.0

  test "triplet margin forward":
    let loss = newTripletMarginLoss()
    let anchor = newTensorRef(newShape(32, 128), dtFloat32)
    let positive = newTensorRef(newShape(32, 128), dtFloat32)
    let negative = newTensorRef(newShape(32, 128), dtFloat32)
    let output = loss.forward(anchor, positive, negative)
    check output.shape == newShape(1)

  test "triplet margin extraRepr":
    let loss = newTripletMarginLoss(margin = 0.5, swap = true)
    let repr = loss.extraRepr
    check "margin=0.5" in repr
    check "swap=True" in repr

suite "MultiMarginLoss":
  test "create multi margin loss":
    let loss = newMultiMarginLoss()
    check loss.p == 1
    check loss.margin == 1.0

  test "multi margin forward":
    let loss = newMultiMarginLoss()
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let target = newTensorRef(newShape(32), dtInt64)
    let output = loss.forward(input, target)
    check output.shape == newShape(1)

suite "CTCLoss":
  test "create ctc loss":
    let loss = newCTCLoss()
    check loss.blank == 0
    check not loss.zeroInfinity

  test "ctc loss forward":
    let loss = newCTCLoss()
    let logProbs = newTensorRef(newShape(50, 32, 100), dtFloat32)
    let targets = newTensorRef(newShape(32, 20), dtInt64)
    let inputLengths = newTensorRef(newShape(32), dtInt32)
    let targetLengths = newTensorRef(newShape(32), dtInt32)
    let output = loss.forward(logProbs, targets, inputLengths, targetLengths)
    check output.shape == newShape(1)

suite "PoissonNLLLoss":
  test "create poisson nll loss":
    let loss = newPoissonNLLLoss()
    check loss.logInput

  test "poisson nll forward":
    let loss = newPoissonNLLLoss()
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let target = newTensorRef(newShape(32, 10), dtFloat32)
    let output = loss.forward(input, target)
    check output.shape == newShape(1)

suite "Loss No Parameters":
  test "all losses have no trainable parameters":
    check newCrossEntropyLoss().numParameters() == 0
    check newNLLLoss().numParameters() == 0
    check newBCELoss().numParameters() == 0
    check newBCEWithLogitsLoss().numParameters() == 0
    check newMSELoss().numParameters() == 0
    check newL1Loss().numParameters() == 0
    check newSmoothL1Loss().numParameters() == 0
    check newHuberLoss().numParameters() == 0
    check newKLDivLoss().numParameters() == 0
    check newMarginRankingLoss().numParameters() == 0
    check newTripletMarginLoss().numParameters() == 0
    check newCTCLoss().numParameters() == 0
