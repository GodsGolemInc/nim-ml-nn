## Tests for dense layers

import unittest
import std/[options, strutils]
import ml_core
import ../src/ml_nn/module
import ../src/ml_nn/layers/dense

suite "Linear Layer":
  test "create linear layer":
    let linear = newLinear(10, 5)
    check linear.inFeatures == 10
    check linear.outFeatures == 5
    check linear.useBias
    check linear.hasParameter("weight")
    check linear.hasParameter("bias")

  test "linear without bias":
    let linear = newLinear(10, 5, useBias = false)
    check not linear.useBias
    check linear.hasParameter("weight")
    check not linear.hasParameter("bias")
    check linear.bias.isNone

  test "linear weight shape":
    let linear = newLinear(10, 5)
    let weightOpt = linear.getParameter("weight")
    check weightOpt.isSome
    check weightOpt.get.data.shape == newShape(5, 10)

  test "linear bias shape":
    let linear = newLinear(10, 5)
    check linear.bias.isSome
    check linear.bias.get.data.shape == newShape(5)

  test "linear forward 1D":
    let linear = newLinear(10, 5)
    let input = newTensorRef(newShape(10), dtFloat32)
    let output = linear.forward(input)
    check output.shape == newShape(5)

  test "linear forward 2D batch":
    let linear = newLinear(10, 5)
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let output = linear.forward(input)
    check output.shape == newShape(32, 5)

  test "linear forward 3D":
    let linear = newLinear(10, 5)
    let input = newTensorRef(newShape(2, 16, 10), dtFloat32)
    let output = linear.forward(input)
    check output.shape == newShape(2, 16, 5)

  test "linear forward size mismatch":
    let linear = newLinear(10, 5)
    let input = newTensorRef(newShape(8), dtFloat32)
    expect(ModuleError):
      discard linear.forward(input)

  test "linear parameter count":
    let linear = newLinear(10, 5)
    check linear.numParameters() == 55  # 10*5 + 5

  test "linear parameter count no bias":
    let linear = newLinear(10, 5, useBias = false)
    check linear.numParameters() == 50  # 10*5

  test "linear numInputs numOutputs":
    let linear = newLinear(100, 50)
    check linear.numInputs == 100
    check linear.numOutputs == 50

  test "linear extraRepr":
    let linear = newLinear(10, 5)
    let repr = linear.extraRepr
    check "in_features=10" in repr
    check "out_features=5" in repr

  test "linear extraRepr no bias":
    let linear = newLinear(10, 5, useBias = false)
    let repr = linear.extraRepr
    check "bias=False" in repr

suite "LazyLinear Layer":
  test "create lazy linear":
    let lazy = newLazyLinear(5)
    check lazy.outFeatures == 5
    check lazy.useBias
    check not lazy.initialized
    check lazy.weight.isNone
    check lazy.bias.isNone

  test "lazy linear materialize":
    let lazy = newLazyLinear(5)
    lazy.materialize(10)
    check lazy.initialized
    check lazy.weight.isSome
    check lazy.weight.get.data.shape == newShape(5, 10)

  test "lazy linear forward initializes":
    let lazy = newLazyLinear(5)
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let output = lazy.forward(input)
    check lazy.initialized
    check output.shape == newShape(32, 5)

  test "lazy linear without bias":
    let lazy = newLazyLinear(5, useBias = false)
    lazy.materialize(10)
    check lazy.bias.isNone

  test "lazy linear double materialize safe":
    let lazy = newLazyLinear(5)
    lazy.materialize(10)
    lazy.materialize(20)  # Should not change
    check lazy.weight.get.data.shape == newShape(5, 10)

suite "Bilinear Layer":
  test "create bilinear":
    let bilinear = newBilinear(10, 8, 5)
    check bilinear.inFeatures1 == 10
    check bilinear.inFeatures2 == 8
    check bilinear.outFeatures == 5
    check bilinear.useBias

  test "bilinear weight shape":
    let bilinear = newBilinear(10, 8, 5)
    let weightOpt = bilinear.getParameter("weight")
    check weightOpt.isSome
    check weightOpt.get.data.shape == newShape(5, 10, 8)

  test "bilinear forward":
    let bilinear = newBilinear(10, 8, 5)
    let input1 = newTensorRef(newShape(32, 10), dtFloat32)
    let input2 = newTensorRef(newShape(32, 8), dtFloat32)
    let output = bilinear.forward(input1, input2)
    check output.shape == newShape(32, 5)

  test "bilinear forward size mismatch":
    let bilinear = newBilinear(10, 8, 5)
    let input1 = newTensorRef(newShape(32, 5), dtFloat32)  # Wrong size
    let input2 = newTensorRef(newShape(32, 8), dtFloat32)
    expect(ModuleError):
      discard bilinear.forward(input1, input2)

  test "bilinear without bias":
    let bilinear = newBilinear(10, 8, 5, useBias = false)
    check not bilinear.useBias
    check bilinear.bias.isNone

  test "bilinear parameter count":
    let bilinear = newBilinear(10, 8, 5)
    check bilinear.numParameters() == 405  # 5*10*8 + 5

suite "Identity Layer":
  test "create identity":
    let identity = newIdentity()
    check identity.name == "Identity"

  test "identity forward":
    let identity = newIdentity()
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let output = identity.forward(input)
    check output == input

  test "identity no parameters":
    let identity = newIdentity()
    check identity.numParameters() == 0

suite "Flatten Layer":
  test "create flatten":
    let flatten = newFlatten()
    check flatten.startDim == 1
    check flatten.endDim == -1

  test "flatten 4D to 2D":
    let flatten = newFlatten()
    let input = newTensorRef(newShape(32, 3, 28, 28), dtFloat32)
    let output = flatten.forward(input)
    check output.shape == newShape(32, 2352)  # 3*28*28

  test "flatten with custom dims":
    let flatten = newFlatten(startDim = 2, endDim = 3)
    let input = newTensorRef(newShape(32, 3, 28, 28), dtFloat32)
    let output = flatten.forward(input)
    check output.shape == newShape(32, 3, 784)  # 28*28

  test "flatten all":
    let flatten = newFlatten(startDim = 0)
    let input = newTensorRef(newShape(2, 3, 4), dtFloat32)
    let output = flatten.forward(input)
    check output.shape == newShape(24)

  test "flatten no parameters":
    let flatten = newFlatten()
    check flatten.numParameters() == 0

suite "Unflatten Layer":
  test "create unflatten":
    let unflatten = newUnflatten(1, @[3, 28, 28])
    check unflatten.dim == 1
    check unflatten.unflattendSize == @[3, 28, 28]

  test "unflatten 2D to 4D":
    let unflatten = newUnflatten(1, @[3, 28, 28])
    let input = newTensorRef(newShape(32, 2352), dtFloat32)
    let output = unflatten.forward(input)
    check output.shape == newShape(32, 3, 28, 28)

  test "unflatten size mismatch":
    let unflatten = newUnflatten(1, @[3, 28, 28])
    let input = newTensorRef(newShape(32, 100), dtFloat32)  # Wrong size
    expect(ModuleError):
      discard unflatten.forward(input)

  test "unflatten no parameters":
    let unflatten = newUnflatten(1, @[4, 4])
    check unflatten.numParameters() == 0

suite "Dense in Sequential":
  test "sequential of linear layers":
    let seq1 = newSequential(
      newLinear(784, 256),
      newLinear(256, 10)
    )
    let input = newTensorRef(newShape(32, 784), dtFloat32)
    let output = seq1.forward(input)
    check output.shape == newShape(32, 10)

  test "sequential parameter count":
    let seq1 = newSequential(
      newLinear(784, 256),
      newLinear(256, 10)
    )
    # (784*256 + 256) + (256*10 + 10) = 200960 + 2570 = 203530
    check seq1.numParameters() == 203530

  test "flatten and linear":
    let seq1 = newSequential(
      newFlatten(),
      newLinear(2352, 10)
    )
    let input = newTensorRef(newShape(32, 3, 28, 28), dtFloat32)
    let output = seq1.forward(input)
    check output.shape == newShape(32, 10)
