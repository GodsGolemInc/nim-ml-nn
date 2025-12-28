## Tests for activation layers

import unittest
import std/[strutils]
import nimml_core
import ../src/nimml_nn/module
import ../src/nimml_nn/layers/activation

suite "ReLU":
  test "create relu":
    let relu = newReLU()
    check relu.name == "ReLU"
    check not relu.inplace

  test "relu inplace":
    let relu = newReLU(inplace = true)
    check relu.inplace

  test "relu forward":
    let relu = newReLU()
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let output = relu.forward(input)
    check output.shape == input.shape

  test "relu no parameters":
    let relu = newReLU()
    check relu.numParameters() == 0

  test "relu extraRepr":
    let relu = newReLU(inplace = true)
    check "inplace=True" in relu.extraRepr

suite "LeakyReLU":
  test "create leaky relu":
    let leaky = newLeakyReLU()
    check leaky.negativeSlope == 0.01

  test "leaky relu custom slope":
    let leaky = newLeakyReLU(negativeSlope = 0.1)
    check leaky.negativeSlope == 0.1

  test "leaky relu forward":
    let leaky = newLeakyReLU()
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let output = leaky.forward(input)
    check output.shape == input.shape

  test "leaky relu extraRepr":
    let leaky = newLeakyReLU(negativeSlope = 0.2, inplace = true)
    let repr = leaky.extraRepr
    check "0.2" in repr
    check "inplace=True" in repr

suite "PReLU":
  test "create prelu":
    let prelu = newPReLU()
    check prelu.numParameters == 1
    check prelu.hasParameter("weight")

  test "prelu multi channel":
    let prelu = newPReLU(numParameters = 64)
    check prelu.weight.data.shape == newShape(64)

  test "prelu forward":
    let prelu = newPReLU()
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let output = prelu.forward(input)
    check output.shape == input.shape

suite "ELU":
  test "create elu":
    let elu = newELU()
    check elu.alpha == 1.0

  test "elu custom alpha":
    let elu = newELU(alpha = 0.5)
    check elu.alpha == 0.5

  test "elu forward":
    let elu = newELU()
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let output = elu.forward(input)
    check output.shape == input.shape

suite "SELU":
  test "create selu":
    let selu = newSELU()
    check selu.name == "SELU"

  test "selu forward":
    let selu = newSELU()
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let output = selu.forward(input)
    check output.shape == input.shape

suite "GELU":
  test "create gelu":
    let gelu = newGELU()
    check gelu.approximate == "none"

  test "gelu tanh approximation":
    let gelu = newGELU(approximate = "tanh")
    check gelu.approximate == "tanh"

  test "gelu forward":
    let gelu = newGELU()
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let output = gelu.forward(input)
    check output.shape == input.shape

suite "Sigmoid":
  test "create sigmoid":
    let sigmoid = newSigmoid()
    check sigmoid.name == "Sigmoid"

  test "sigmoid forward":
    let sigmoid = newSigmoid()
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let output = sigmoid.forward(input)
    check output.shape == input.shape

suite "Tanh":
  test "create tanh":
    let tanh = newTanh()
    check tanh.name == "Tanh"

  test "tanh forward":
    let tanh = newTanh()
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let output = tanh.forward(input)
    check output.shape == input.shape

suite "Softmax":
  test "create softmax":
    let softmax = newSoftmax()
    check softmax.dim == -1

  test "softmax custom dim":
    let softmax = newSoftmax(dim = 1)
    check softmax.dim == 1

  test "softmax forward":
    let softmax = newSoftmax()
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let output = softmax.forward(input)
    check output.shape == input.shape

  test "softmax extraRepr":
    let softmax = newSoftmax(dim = 1)
    check "dim=1" in softmax.extraRepr

suite "LogSoftmax":
  test "create log softmax":
    let logSoftmax = newLogSoftmax()
    check logSoftmax.dim == -1

  test "log softmax forward":
    let logSoftmax = newLogSoftmax()
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let output = logSoftmax.forward(input)
    check output.shape == input.shape

suite "Softplus":
  test "create softplus":
    let softplus = newSoftplus()
    check softplus.beta == 1.0
    check softplus.threshold == 20.0

  test "softplus forward":
    let softplus = newSoftplus()
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let output = softplus.forward(input)
    check output.shape == input.shape

suite "Softsign":
  test "create softsign":
    let softsign = newSoftsign()
    check softsign.name == "Softsign"

  test "softsign forward":
    let softsign = newSoftsign()
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let output = softsign.forward(input)
    check output.shape == input.shape

suite "Hardtanh":
  test "create hardtanh":
    let hardtanh = newHardtanh()
    check hardtanh.minVal == -1.0
    check hardtanh.maxVal == 1.0

  test "hardtanh custom range":
    let hardtanh = newHardtanh(minVal = 0.0, maxVal = 6.0)
    check hardtanh.minVal == 0.0
    check hardtanh.maxVal == 6.0

  test "hardtanh forward":
    let hardtanh = newHardtanh()
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let output = hardtanh.forward(input)
    check output.shape == input.shape

  test "hardtanh extraRepr":
    let hardtanh = newHardtanh(inplace = true)
    check "inplace=True" in hardtanh.extraRepr

suite "Hardswish":
  test "create hardswish":
    let hardswish = newHardswish()
    check hardswish.name == "Hardswish"

  test "hardswish forward":
    let hardswish = newHardswish()
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let output = hardswish.forward(input)
    check output.shape == input.shape

suite "Hardsigmoid":
  test "create hardsigmoid":
    let hardsigmoid = newHardsigmoid()
    check hardsigmoid.name == "Hardsigmoid"

  test "hardsigmoid forward":
    let hardsigmoid = newHardsigmoid()
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let output = hardsigmoid.forward(input)
    check output.shape == input.shape

suite "SiLU":
  test "create silu":
    let silu = newSiLU()
    check silu.name == "SiLU"

  test "silu forward":
    let silu = newSiLU()
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let output = silu.forward(input)
    check output.shape == input.shape

suite "Mish":
  test "create mish":
    let mish = newMish()
    check mish.name == "Mish"

  test "mish forward":
    let mish = newMish()
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let output = mish.forward(input)
    check output.shape == input.shape

suite "GLU":
  test "create glu":
    let glu = newGLU()
    check glu.dim == -1

  test "glu forward":
    let glu = newGLU()
    let input = newTensorRef(newShape(32, 20), dtFloat32)
    let output = glu.forward(input)
    check output.shape == newShape(32, 10)  # Half the last dim

  test "glu forward 3D":
    let glu = newGLU(dim = 1)
    let input = newTensorRef(newShape(32, 64, 10), dtFloat32)
    let output = glu.forward(input)
    check output.shape == newShape(32, 32, 10)

  test "glu odd size error":
    let glu = newGLU()
    let input = newTensorRef(newShape(32, 11), dtFloat32)
    expect(ModuleError):
      discard glu.forward(input)

suite "Activations in Sequential":
  test "linear with relu":
    let model = newSequential()
    # Would need linear import, just test activation chain
    let input = newTensorRef(newShape(32, 10), dtFloat32)

    let relu = newReLU()
    let output = relu.forward(input)
    check output.shape == newShape(32, 10)

  test "multiple activations":
    let relu = newReLU()
    let sigmoid = newSigmoid()

    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let h = relu.forward(input)
    let output = sigmoid.forward(h)

    check output.shape == newShape(32, 10)
