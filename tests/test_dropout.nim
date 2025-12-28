## Tests for dropout layers

import unittest
import std/[strutils]
import nimml_core
import ../src/nimml_nn/module
import ../src/nimml_nn/layers/dropout

suite "Dropout":
  test "create dropout":
    let dropout = newDropout()
    check dropout.p == 0.5
    check not dropout.inplace

  test "dropout custom rate":
    let dropout = newDropout(p = 0.3)
    check dropout.p == 0.3

  test "dropout inplace":
    let dropout = newDropout(inplace = true)
    check dropout.inplace

  test "dropout invalid rate low":
    expect(ModuleError):
      discard newDropout(p = -0.1)

  test "dropout invalid rate high":
    expect(ModuleError):
      discard newDropout(p = 1.1)

  test "dropout forward training":
    let dropout = newDropout(p = 0.5)
    dropout.train()
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let output = dropout.forward(input)
    check output.shape == input.shape

  test "dropout forward eval returns input":
    let dropout = newDropout(p = 0.5)
    dropout.eval()
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let output = dropout.forward(input)
    check output == input

  test "dropout p=0 returns input":
    let dropout = newDropout(p = 0.0)
    dropout.train()
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let output = dropout.forward(input)
    check output == input

  test "dropout no parameters":
    let dropout = newDropout()
    check dropout.numParameters() == 0

  test "dropout extraRepr":
    let dropout = newDropout(p = 0.3, inplace = true)
    let repr = dropout.extraRepr
    check "p=0.3" in repr
    check "inplace=True" in repr

suite "Dropout1d":
  test "create dropout1d":
    let dropout = newDropout1d()
    check dropout.p == 0.5

  test "dropout1d forward":
    let dropout = newDropout1d(p = 0.3)
    dropout.train()
    let input = newTensorRef(newShape(32, 64, 100), dtFloat32)
    let output = dropout.forward(input)
    check output.shape == input.shape

  test "dropout1d wrong dims":
    let dropout = newDropout1d()
    dropout.train()
    let input = newTensorRef(newShape(32, 10), dtFloat32)  # 2D not 3D
    expect(ModuleError):
      discard dropout.forward(input)

  test "dropout1d eval returns input":
    let dropout = newDropout1d()
    dropout.eval()
    let input = newTensorRef(newShape(32, 64, 100), dtFloat32)
    let output = dropout.forward(input)
    check output == input

suite "Dropout2d":
  test "create dropout2d":
    let dropout = newDropout2d()
    check dropout.p == 0.5

  test "dropout2d forward":
    let dropout = newDropout2d(p = 0.3)
    dropout.train()
    let input = newTensorRef(newShape(32, 64, 28, 28), dtFloat32)
    let output = dropout.forward(input)
    check output.shape == input.shape

  test "dropout2d wrong dims":
    let dropout = newDropout2d()
    dropout.train()
    let input = newTensorRef(newShape(32, 64, 28), dtFloat32)  # 3D not 4D
    expect(ModuleError):
      discard dropout.forward(input)

  test "dropout2d eval returns input":
    let dropout = newDropout2d()
    dropout.eval()
    let input = newTensorRef(newShape(32, 64, 28, 28), dtFloat32)
    let output = dropout.forward(input)
    check output == input

suite "Dropout3d":
  test "create dropout3d":
    let dropout = newDropout3d()
    check dropout.p == 0.5

  test "dropout3d forward":
    let dropout = newDropout3d(p = 0.3)
    dropout.train()
    let input = newTensorRef(newShape(8, 64, 16, 28, 28), dtFloat32)
    let output = dropout.forward(input)
    check output.shape == input.shape

  test "dropout3d wrong dims":
    let dropout = newDropout3d()
    dropout.train()
    let input = newTensorRef(newShape(8, 64, 28, 28), dtFloat32)  # 4D not 5D
    expect(ModuleError):
      discard dropout.forward(input)

  test "dropout3d eval returns input":
    let dropout = newDropout3d()
    dropout.eval()
    let input = newTensorRef(newShape(8, 64, 16, 28, 28), dtFloat32)
    let output = dropout.forward(input)
    check output == input

suite "AlphaDropout":
  test "create alpha dropout":
    let dropout = newAlphaDropout()
    check dropout.p == 0.5

  test "alpha dropout forward":
    let dropout = newAlphaDropout(p = 0.1)
    dropout.train()
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let output = dropout.forward(input)
    check output.shape == input.shape

  test "alpha dropout eval returns input":
    let dropout = newAlphaDropout()
    dropout.eval()
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let output = dropout.forward(input)
    check output == input

  test "alpha dropout extraRepr":
    let dropout = newAlphaDropout(p = 0.1, inplace = true)
    let repr = dropout.extraRepr
    check "p=0.1" in repr
    check "inplace=True" in repr

suite "FeatureAlphaDropout":
  test "create feature alpha dropout":
    let dropout = newFeatureAlphaDropout()
    check dropout.p == 0.5

  test "feature alpha dropout forward":
    let dropout = newFeatureAlphaDropout(p = 0.1)
    dropout.train()
    let input = newTensorRef(newShape(32, 64), dtFloat32)
    let output = dropout.forward(input)
    check output.shape == input.shape

  test "feature alpha dropout eval":
    let dropout = newFeatureAlphaDropout()
    dropout.eval()
    let input = newTensorRef(newShape(32, 64), dtFloat32)
    let output = dropout.forward(input)
    check output == input

suite "Dropout Training Mode Propagation":
  test "dropout in module inherits training":
    let parent = newModule("parent")
    let dropout = newDropout(p = 0.5)
    parent.registerModule("dropout", dropout)

    parent.train()
    check parent.isTraining()
    check dropout.isTraining()

    parent.eval()
    check not parent.isTraining()
    check not dropout.isTraining()

  test "dropout in sequential":
    let seq1 = newSequential()
    let dropout = newDropout(p = 0.5)
    seq1.add(dropout)

    seq1.train()
    check dropout.isTraining()

    seq1.eval()
    check not dropout.isTraining()
