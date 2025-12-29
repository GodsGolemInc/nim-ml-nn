## Tests for optimizers and learning rate schedulers

import unittest
import std/[options, strutils, tables]
import ml_core
import ../src/ml_nn/module
import ../src/ml_nn/optim

# Helper to create test parameters
proc createTestParams(): seq[Parameter] =
  @[
    newParameter("weight", newShape(10, 5), dtFloat32),
    newParameter("bias", newShape(5), dtFloat32)
  ]

suite "SGD Optimizer":
  test "create sgd":
    let params = createTestParams()
    let sgd = newSGD(params, lr = 0.01)
    check sgd.lr == 0.01
    check sgd.momentum == 0.0
    check sgd.weightDecay == 0.0

  test "sgd with momentum":
    let params = createTestParams()
    let sgd = newSGD(params, lr = 0.01, momentum = 0.9)
    check sgd.momentum == 0.9

  test "sgd with weight decay":
    let params = createTestParams()
    let sgd = newSGD(params, lr = 0.01, weightDecay = 0.0001)
    check sgd.weightDecay == 0.0001

  test "sgd nesterov requires momentum":
    let params = createTestParams()
    expect(OptimizerError):
      discard newSGD(params, lr = 0.01, nesterov = true)

  test "sgd nesterov valid":
    let params = createTestParams()
    let sgd = newSGD(params, lr = 0.01, momentum = 0.9, nesterov = true)
    check sgd.nesterov

  test "sgd zero grad":
    let params = createTestParams()
    let sgd = newSGD(params, lr = 0.01)

    # Set a gradient through the optimizer's param groups
    sgd.paramGroups[0].params[0].gradRef = some(newTensorRef(newShape(10, 5), dtFloat32))
    check sgd.paramGroups[0].params[0].gradRef.isSome

    sgd.zeroGrad()
    check sgd.paramGroups[0].params[0].gradRef.isNone

  test "sgd get and set lr":
    let params = createTestParams()
    let sgd = newSGD(params, lr = 0.01)
    check sgd.getLR() == 0.01

    sgd.setLR(0.001)
    check sgd.getLR() == 0.001

  test "sgd extraRepr":
    let params = createTestParams()
    let sgd = newSGD(params, lr = 0.01, momentum = 0.9, weightDecay = 0.0001)
    let repr = sgd.extraRepr
    check "lr=0.01" in repr
    check "momentum=0.9" in repr
    check "weight_decay=0.0001" in repr

suite "Adam Optimizer":
  test "create adam":
    let params = createTestParams()
    let adam = newAdam(params)
    check adam.lr == 0.001
    check adam.beta1 == 0.9
    check adam.beta2 == 0.999
    check adam.eps == 1e-8

  test "adam custom params":
    let params = createTestParams()
    let adam = newAdam(params, lr = 0.0001, betas = (0.95, 0.99))
    check adam.lr == 0.0001
    check adam.beta1 == 0.95
    check adam.beta2 == 0.99

  test "adam with weight decay":
    let params = createTestParams()
    let adam = newAdam(params, weightDecay = 0.01)
    check adam.weightDecay == 0.01

  test "adam amsgrad":
    let params = createTestParams()
    let adam = newAdam(params, amsgrad = true)
    check adam.amsgrad

  test "adam extraRepr":
    let params = createTestParams()
    let adam = newAdam(params, amsgrad = true)
    let repr = adam.extraRepr
    check "lr=0.001" in repr
    check "amsgrad=True" in repr

suite "AdamW Optimizer":
  test "create adamw":
    let params = createTestParams()
    let adamw = newAdamW(params)
    check adamw.lr == 0.001
    check adamw.weightDecay == 0.01

  test "adamw custom weight decay":
    let params = createTestParams()
    let adamw = newAdamW(params, weightDecay = 0.1)
    check adamw.weightDecay == 0.1

  test "adamw extraRepr":
    let params = createTestParams()
    let adamw = newAdamW(params)
    let repr = adamw.extraRepr
    check "weight_decay=0.01" in repr

suite "Adagrad Optimizer":
  test "create adagrad":
    let params = createTestParams()
    let adagrad = newAdagrad(params)
    check adagrad.lr == 0.01

  test "adagrad custom params":
    let params = createTestParams()
    let adagrad = newAdagrad(params, lr = 0.1, lrDecay = 0.01)
    check adagrad.lr == 0.1
    check adagrad.lrDecay == 0.01

suite "RMSprop Optimizer":
  test "create rmsprop":
    let params = createTestParams()
    let rmsprop = newRMSprop(params)
    check rmsprop.lr == 0.01
    check rmsprop.alpha == 0.99

  test "rmsprop centered":
    let params = createTestParams()
    let rmsprop = newRMSprop(params, centered = true)
    check rmsprop.centered

suite "Adadelta Optimizer":
  test "create adadelta":
    let params = createTestParams()
    let adadelta = newAdadelta(params)
    check adadelta.lr == 1.0
    check adadelta.rho == 0.9

suite "Add Parameter Group":
  test "add param group":
    let params1 = @[newParameter("w1", newShape(10), dtFloat32)]
    let params2 = @[newParameter("w2", newShape(5), dtFloat32)]

    let adam = newAdam(params1)
    check adam.paramGroups.len == 1

    adam.addParamGroup(params2)
    check adam.paramGroups.len == 2

suite "StepLR Scheduler":
  test "create step lr":
    let params = createTestParams()
    let optimizer = newSGD(params, lr = 0.1)
    let scheduler = newStepLR(optimizer, stepSize = 10, gamma = 0.1)
    check scheduler.stepSize == 10
    check scheduler.gamma == 0.1

  test "step lr decay":
    let params = createTestParams()
    let optimizer = newSGD(params, lr = 0.1)
    let scheduler = newStepLR(optimizer, stepSize = 2, gamma = 0.5)

    # Initially at epoch -1
    check scheduler.lastEpoch == -1

    scheduler.step()  # epoch 0
    var lrs = scheduler.getLR()
    check lrs[0] == 0.1

    scheduler.step()  # epoch 1
    lrs = scheduler.getLR()
    check lrs[0] == 0.1

    scheduler.step()  # epoch 2
    lrs = scheduler.getLR()
    check abs(lrs[0] - 0.05) < 1e-6

suite "MultiStepLR Scheduler":
  test "create multi step lr":
    let params = createTestParams()
    let optimizer = newSGD(params, lr = 0.1)
    let scheduler = newMultiStepLR(optimizer, milestones = @[5, 10, 15])
    check scheduler.milestones == @[5, 10, 15]

  test "multi step lr decay":
    let params = createTestParams()
    let optimizer = newSGD(params, lr = 0.1)
    let scheduler = newMultiStepLR(optimizer, milestones = @[2, 4], gamma = 0.5)

    for _ in 0..<5:
      scheduler.step()

    let lrs = scheduler.getLR()
    # After milestones 2 and 4: 0.1 * 0.5 * 0.5 = 0.025
    check abs(lrs[0] - 0.025) < 1e-6

suite "ExponentialLR Scheduler":
  test "create exponential lr":
    let params = createTestParams()
    let optimizer = newSGD(params, lr = 0.1)
    let scheduler = newExponentialLR(optimizer, gamma = 0.9)
    check scheduler.gamma == 0.9

  test "exponential lr decay":
    let params = createTestParams()
    let optimizer = newSGD(params, lr = 0.1)
    let scheduler = newExponentialLR(optimizer, gamma = 0.9)

    scheduler.step()
    scheduler.step()

    let lrs = scheduler.getLR()
    check abs(lrs[0] - 0.1 * 0.9) < 1e-6

suite "CosineAnnealingLR Scheduler":
  test "create cosine annealing":
    let params = createTestParams()
    let optimizer = newSGD(params, lr = 0.1)
    let scheduler = newCosineAnnealingLR(optimizer, tMax = 100)
    check scheduler.tMax == 100
    check scheduler.etaMin == 0.0

  test "cosine annealing decay":
    let params = createTestParams()
    let optimizer = newSGD(params, lr = 0.1)
    let scheduler = newCosineAnnealingLR(optimizer, tMax = 10, etaMin = 0.01)

    # At t=0: lr = 0.1
    scheduler.step()
    var lrs = scheduler.getLR()
    check lrs[0] > 0.09

    # At t=5: lr should be around midpoint
    for _ in 0..<4:
      scheduler.step()
    lrs = scheduler.getLR()
    check lrs[0] < 0.08
    check lrs[0] > 0.04

suite "CosineAnnealingWarmRestarts Scheduler":
  test "create warm restarts":
    let params = createTestParams()
    let optimizer = newSGD(params, lr = 0.1)
    let scheduler = newCosineAnnealingWarmRestarts(optimizer, t0 = 10)
    check scheduler.t0 == 10

suite "LinearLR Scheduler":
  test "create linear lr":
    let params = createTestParams()
    let optimizer = newSGD(params, lr = 0.1)
    let scheduler = newLinearLR(optimizer, startFactor = 0.1, totalIters = 10)
    check scheduler.startFactor == 0.1
    check scheduler.endFactor == 1.0

  test "linear lr warmup":
    let params = createTestParams()
    let optimizer = newSGD(params, lr = 0.1)
    let scheduler = newLinearLR(optimizer, startFactor = 0.1, endFactor = 1.0, totalIters = 10)

    scheduler.step()  # epoch 0
    var lrs = scheduler.getLR()
    # At epoch 0: 0.1 * 0.1 = 0.01
    check abs(lrs[0] - 0.01) < 1e-6

suite "PolynomialLR Scheduler":
  test "create polynomial lr":
    let params = createTestParams()
    let optimizer = newSGD(params, lr = 0.1)
    let scheduler = newPolynomialLR(optimizer, totalIters = 100, power = 2.0)
    check scheduler.power == 2.0

suite "OneCycleLR Scheduler":
  test "create one cycle lr":
    let params = createTestParams()
    let optimizer = newSGD(params, lr = 0.1)
    let scheduler = newOneCycleLR(optimizer, maxLR = 0.1, totalSteps = 100)
    check scheduler.maxLR == 0.1
    check scheduler.totalSteps == 100

  test "one cycle warmup":
    let params = createTestParams()
    let optimizer = newSGD(params, lr = 0.1)
    let scheduler = newOneCycleLR(optimizer, maxLR = 0.1, totalSteps = 100, pctStart = 0.3)

    # Initial LR should be maxLR / divFactor
    scheduler.step()
    var lrs = scheduler.getLR()
    check lrs[0] < 0.1
    check lrs[0] > 0.003  # maxLR / 25

suite "ReduceLROnPlateau Scheduler":
  test "create reduce on plateau":
    let params = createTestParams()
    let optimizer = newSGD(params, lr = 0.1)
    let scheduler = newReduceLROnPlateau(optimizer, patience = 5)
    check scheduler.patience == 5
    check scheduler.factor == 0.1

  test "reduce on plateau min mode":
    let params = createTestParams()
    let optimizer = newSGD(params, lr = 0.1)
    let scheduler = newReduceLROnPlateau(optimizer, mode = "min", patience = 2)

    # Loss keeps getting worse
    scheduler.step(1.0)  # best = 1.0
    scheduler.step(1.5)  # worse
    scheduler.step(1.6)  # worse
    scheduler.step(1.7)  # worse, should reduce LR

    let currentLR = optimizer.getLR()
    check currentLR < 0.1

  test "reduce on plateau max mode":
    let params = createTestParams()
    let optimizer = newSGD(params, lr = 0.1)
    let scheduler = newReduceLROnPlateau(optimizer, mode = "max", patience = 2)

    # Accuracy keeps getting worse
    scheduler.step(0.9)  # best = 0.9
    scheduler.step(0.8)  # worse
    scheduler.step(0.7)  # worse
    scheduler.step(0.6)  # worse, should reduce LR

    let currentLR = optimizer.getLR()
    check currentLR < 0.1

suite "Optimizer State":
  test "optimizer state initialization":
    let params = createTestParams()
    let adam = newAdam(params)

    # Initially no state
    check adam.state.len == 0

    # After accessing state, it gets initialized
    discard adam.getState(adam.paramGroups[0].params[0])
    check adam.state.len == 1

  test "optimizer param groups":
    let params = createTestParams()
    let adam = newAdam(params)

    check adam.paramGroups.len == 1
    check adam.paramGroups[0].params.len == 2
