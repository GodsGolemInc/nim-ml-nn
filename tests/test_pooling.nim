## Pooling Layer Tests
##
## Tests for MaxPool, AvgPool, AdaptivePool, etc.

import std/options
import unittest
import ml_core
import ml_nn/module
import ml_nn/layers/pooling

# =============================================================================
# MaxPool1d Tests
# =============================================================================

suite "MaxPool1d":
  test "create MaxPool1d":
    let pool = newMaxPool1d(2)
    check pool.kernelSize == 2
    check pool.stride == 2  # Default stride = kernel_size
    check pool.padding == 0
    check pool.dilation == 1

  test "MaxPool1d custom stride":
    let pool = newMaxPool1d(3, stride = 1)
    check pool.stride == 1

  test "MaxPool1d forward":
    let pool = newMaxPool1d(2)
    let input = newTensorRef(newShape(32, 64, 100), dtFloat32)
    let output = pool.forward(input)
    # Output length = (100 - 2) / 2 + 1 = 50
    check output.shape.dims == @[32, 64, 50]

  test "MaxPool1d with padding":
    let pool = newMaxPool1d(3, stride = 1, padding = 1)
    let input = newTensorRef(newShape(32, 64, 100), dtFloat32)
    let output = pool.forward(input)
    check output.shape.dims == @[32, 64, 100]

  test "MaxPool1d ceil mode":
    let pool = newMaxPool1d(3, stride = 2, ceilMode = true)
    let input = newTensorRef(newShape(32, 64, 7), dtFloat32)
    let output = pool.forward(input)
    # With ceil: (7 + 2 - 3 - 1) / 2 + 1 = 3 (ceiling)
    check output.shape.dims[2] >= 3

# =============================================================================
# AvgPool1d Tests
# =============================================================================

suite "AvgPool1d":
  test "create AvgPool1d":
    let pool = newAvgPool1d(2)
    check pool.kernelSize == 2
    check pool.countIncludePad == true

  test "AvgPool1d forward":
    let pool = newAvgPool1d(2)
    let input = newTensorRef(newShape(32, 64, 100), dtFloat32)
    let output = pool.forward(input)
    check output.shape.dims == @[32, 64, 50]

# =============================================================================
# MaxPool2d Tests
# =============================================================================

suite "MaxPool2d":
  test "create MaxPool2d int":
    let pool = newMaxPool2d(2)
    check pool.kernelSize == (h: 2, w: 2)
    check pool.stride == (h: 2, w: 2)

  test "create MaxPool2d tuple":
    let pool = newMaxPool2d((2, 3))
    check pool.kernelSize == (h: 2, w: 3)

  test "MaxPool2d forward":
    let pool = newMaxPool2d(2)
    let input = newTensorRef(newShape(32, 64, 28, 28), dtFloat32)
    let output = pool.forward(input)
    check output.shape.dims == @[32, 64, 14, 14]

  test "MaxPool2d forward 3x3":
    let pool = newMaxPool2d(3, stride = 2, padding = 1)
    let input = newTensorRef(newShape(32, 64, 28, 28), dtFloat32)
    let output = pool.forward(input)
    check output.shape.dims == @[32, 64, 14, 14]

  test "MaxPool2d return indices":
    let pool = newMaxPool2d(2, returnIndices = true)
    check pool.returnIndices == true

  test "MaxPool2d with dilation":
    let pool = newMaxPool2d(3, dilation = 2)
    let input = newTensorRef(newShape(32, 64, 28, 28), dtFloat32)
    let output = pool.forward(input)
    # Dilated kernel: 2*(3-1)+1 = 5
    # Output: (28 - 5) / 3 + 1 = 8 (stride = kernel_size for dilation > 1)
    check output.shape.dims[0] == 32

  test "MaxPool2d wrong dims":
    let pool = newMaxPool2d(2)
    let input = newTensorRef(newShape(32, 64, 28), dtFloat32)
    expect(ModuleError):
      discard pool.forward(input)

# =============================================================================
# AvgPool2d Tests
# =============================================================================

suite "AvgPool2d":
  test "create AvgPool2d":
    let pool = newAvgPool2d(2)
    check pool.kernelSize == (h: 2, w: 2)
    check pool.countIncludePad == true

  test "AvgPool2d forward":
    let pool = newAvgPool2d(2)
    let input = newTensorRef(newShape(32, 64, 28, 28), dtFloat32)
    let output = pool.forward(input)
    check output.shape.dims == @[32, 64, 14, 14]

  test "AvgPool2d divisor override":
    let pool = newAvgPool2d(2, divisorOverride = some(4))
    check pool.divisorOverride.isSome
    check pool.divisorOverride.get == 4

# =============================================================================
# MaxPool3d Tests
# =============================================================================

suite "MaxPool3d":
  test "create MaxPool3d":
    let pool = newMaxPool3d(2)
    check pool.kernelSize == (d: 2, h: 2, w: 2)

  test "MaxPool3d forward":
    let pool = newMaxPool3d(2)
    let input = newTensorRef(newShape(4, 64, 16, 28, 28), dtFloat32)
    let output = pool.forward(input)
    check output.shape.dims == @[4, 64, 8, 14, 14]

# =============================================================================
# AdaptiveMaxPool Tests
# =============================================================================

suite "AdaptiveMaxPool1d":
  test "create AdaptiveMaxPool1d":
    let pool = newAdaptiveMaxPool1d(1)
    check pool.outputSize == 1

  test "AdaptiveMaxPool1d forward":
    let pool = newAdaptiveMaxPool1d(10)
    let input = newTensorRef(newShape(32, 64, 100), dtFloat32)
    let output = pool.forward(input)
    check output.shape.dims == @[32, 64, 10]

suite "AdaptiveMaxPool2d":
  test "create AdaptiveMaxPool2d int":
    let pool = newAdaptiveMaxPool2d(1)
    check pool.outputSize == (h: 1, w: 1)

  test "create AdaptiveMaxPool2d tuple":
    let pool = newAdaptiveMaxPool2d((7, 7))
    check pool.outputSize == (h: 7, w: 7)

  test "AdaptiveMaxPool2d forward":
    let pool = newAdaptiveMaxPool2d((7, 7))
    let input = newTensorRef(newShape(32, 64, 28, 28), dtFloat32)
    let output = pool.forward(input)
    check output.shape.dims == @[32, 64, 7, 7]

# =============================================================================
# AdaptiveAvgPool Tests
# =============================================================================

suite "AdaptiveAvgPool1d":
  test "create AdaptiveAvgPool1d":
    let pool = newAdaptiveAvgPool1d(1)
    check pool.outputSize == 1

  test "AdaptiveAvgPool1d forward":
    let pool = newAdaptiveAvgPool1d(10)
    let input = newTensorRef(newShape(32, 64, 100), dtFloat32)
    let output = pool.forward(input)
    check output.shape.dims == @[32, 64, 10]

suite "AdaptiveAvgPool2d":
  test "create AdaptiveAvgPool2d":
    let pool = newAdaptiveAvgPool2d(1)
    check pool.outputSize == (h: 1, w: 1)

  test "AdaptiveAvgPool2d forward":
    let pool = newAdaptiveAvgPool2d((7, 7))
    let input = newTensorRef(newShape(32, 512, 28, 28), dtFloat32)
    let output = pool.forward(input)
    check output.shape.dims == @[32, 512, 7, 7]

  test "AdaptiveAvgPool2d global":
    let pool = newAdaptiveAvgPool2d(1)
    let input = newTensorRef(newShape(32, 512, 28, 28), dtFloat32)
    let output = pool.forward(input)
    check output.shape.dims == @[32, 512, 1, 1]

suite "AdaptiveAvgPool3d":
  test "create AdaptiveAvgPool3d":
    let pool = newAdaptiveAvgPool3d((4, 4, 4))
    check pool.outputSize == (d: 4, h: 4, w: 4)

  test "AdaptiveAvgPool3d forward":
    let pool = newAdaptiveAvgPool3d((4, 4, 4))
    let input = newTensorRef(newShape(4, 64, 16, 28, 28), dtFloat32)
    let output = pool.forward(input)
    check output.shape.dims == @[4, 64, 4, 4, 4]

# =============================================================================
# Global Pooling Shortcuts
# =============================================================================

suite "Global Pooling":
  test "GlobalMaxPool2d":
    let pool = newGlobalMaxPool2d()
    check pool.outputSize == (h: 1, w: 1)

    let input = newTensorRef(newShape(32, 512, 7, 7), dtFloat32)
    let output = pool.forward(input)
    check output.shape.dims == @[32, 512, 1, 1]

  test "GlobalAvgPool2d":
    let pool = newGlobalAvgPool2d()
    check pool.outputSize == (h: 1, w: 1)

    let input = newTensorRef(newShape(32, 512, 7, 7), dtFloat32)
    let output = pool.forward(input)
    check output.shape.dims == @[32, 512, 1, 1]

  test "GlobalMaxPool1d":
    let pool = newGlobalMaxPool1d()
    check pool.outputSize == 1

  test "GlobalAvgPool1d":
    let pool = newGlobalAvgPool1d()
    check pool.outputSize == 1

# =============================================================================
# LPPool Tests
# =============================================================================

suite "LPPool1d":
  test "create LPPool1d":
    let pool = newLPPool1d(2.0, 3)
    check pool.normType == 2.0
    check pool.kernelSize == 3

  test "LPPool1d forward":
    let pool = newLPPool1d(2.0, 3, stride = 2)
    let input = newTensorRef(newShape(32, 64, 100), dtFloat32)
    let output = pool.forward(input)
    check output.shape.dims[0] == 32
    check output.shape.dims[1] == 64

suite "LPPool2d":
  test "create LPPool2d":
    let pool = newLPPool2d(2.0, 3)
    check pool.normType == 2.0

  test "LPPool2d forward":
    let pool = newLPPool2d(2.0, 3, stride = 2)
    let input = newTensorRef(newShape(32, 64, 28, 28), dtFloat32)
    let output = pool.forward(input)
    check output.shape.dims[0] == 32
    check output.shape.dims[1] == 64

# =============================================================================
# MaxUnpool Tests
# =============================================================================

suite "MaxUnpool2d":
  test "create MaxUnpool2d":
    let unpool = newMaxUnpool2d(2)
    check unpool.kernelSize == (h: 2, w: 2)

  test "MaxUnpool2d forward":
    let unpool = newMaxUnpool2d(2)
    let input = newTensorRef(newShape(32, 64, 14, 14), dtFloat32)
    let indices = newTensorRef(newShape(32, 64, 14, 14), dtInt64)
    let output = unpool.forward(input, indices)
    check output.shape.dims == @[32, 64, 28, 28]
