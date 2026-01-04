## Convolution Layer Tests
##
## Tests for Conv1d, Conv2d, Conv3d and transposed variants.

import std/options
import unittest
import ml_core
import ml_nn/module
import ml_nn/layers/conv

# =============================================================================
# Conv1d Tests
# =============================================================================

suite "Conv1d":
  test "create Conv1d":
    let conv = newConv1d(64, 128, 3)
    check conv.inChannels == 64
    check conv.outChannels == 128
    check conv.kernelSize == 3
    check conv.stride == 1
    check conv.padding == 0
    check conv.dilation == 1
    check conv.groups == 1
    check conv.useBias == true

  test "Conv1d weight shape":
    let conv = newConv1d(64, 128, 3)
    check conv.weight.data.shape.dims == @[128, 64, 3]

  test "Conv1d with padding":
    let conv = newConv1d(64, 128, 3, padding = 1)
    check conv.padding == 1

  test "Conv1d forward":
    let conv = newConv1d(64, 128, 3)
    let input = newTensorRef(newShape(32, 64, 100), dtFloat32)
    let output = conv.forward(input)
    # Output length = (100 + 2*0 - 1*(3-1) - 1) / 1 + 1 = 98
    check output.shape.dims == @[32, 128, 98]

  test "Conv1d forward with padding":
    let conv = newConv1d(64, 128, 3, padding = 1)
    let input = newTensorRef(newShape(32, 64, 100), dtFloat32)
    let output = conv.forward(input)
    # Output length = (100 + 2*1 - 3) / 1 + 1 = 100
    check output.shape.dims == @[32, 128, 100]

  test "Conv1d groups":
    let conv = newConv1d(64, 128, 3, groups = 4)
    check conv.groups == 4
    check conv.weight.data.shape.dims == @[128, 16, 3]  # 64/4 = 16

  test "Conv1d without bias":
    let conv = newConv1d(64, 128, 3, useBias = false)
    check conv.bias.isNone
    check conv.parameters().len == 1

  test "Conv1d wrong channels":
    let conv = newConv1d(64, 128, 3)
    let input = newTensorRef(newShape(32, 32, 100), dtFloat32)
    expect(ModuleError):
      discard conv.forward(input)

# =============================================================================
# Conv2d Tests
# =============================================================================

suite "Conv2d":
  test "create Conv2d int kernel":
    let conv = newConv2d(64, 128, 3)
    check conv.inChannels == 64
    check conv.outChannels == 128
    check conv.kernelSize == (h: 3, w: 3)

  test "create Conv2d tuple kernel":
    let conv = newConv2d(64, 128, (3, 5))
    check conv.kernelSize == (h: 3, w: 5)

  test "Conv2d weight shape":
    let conv = newConv2d(64, 128, 3)
    check conv.weight.data.shape.dims == @[128, 64, 3, 3]

  test "Conv2d forward":
    let conv = newConv2d(64, 128, 3)
    let input = newTensorRef(newShape(32, 64, 28, 28), dtFloat32)
    let output = conv.forward(input)
    # Output size = (28 - 3) / 1 + 1 = 26
    check output.shape.dims == @[32, 128, 26, 26]

  test "Conv2d forward with padding":
    let conv = newConv2d(64, 128, 3, padding = 1)
    let input = newTensorRef(newShape(32, 64, 28, 28), dtFloat32)
    let output = conv.forward(input)
    # Same size with padding=1
    check output.shape.dims == @[32, 128, 28, 28]

  test "Conv2d with stride":
    let conv = newConv2d(64, 128, 3, stride = 2, padding = 1)
    let input = newTensorRef(newShape(32, 64, 28, 28), dtFloat32)
    let output = conv.forward(input)
    # Output size = (28 + 2 - 3) / 2 + 1 = 14
    check output.shape.dims == @[32, 128, 14, 14]

  test "Conv2d with dilation":
    let conv = newConv2d(64, 128, 3, dilation = 2, padding = 2)
    let input = newTensorRef(newShape(32, 64, 28, 28), dtFloat32)
    let output = conv.forward(input)
    # Dilated kernel size = 2*(3-1)+1 = 5
    # Output size = (28 + 4 - 5) / 1 + 1 = 28
    check output.shape.dims == @[32, 128, 28, 28]

  test "Conv2d groups (depthwise)":
    let conv = newConv2d(64, 64, 3, groups = 64)  # Depthwise
    check conv.groups == 64
    check conv.weight.data.shape.dims == @[64, 1, 3, 3]

  test "Conv2d wrong dimensions":
    let conv = newConv2d(64, 128, 3)
    let input = newTensorRef(newShape(32, 64, 28), dtFloat32)
    expect(ModuleError):
      discard conv.forward(input)

# =============================================================================
# ConvTranspose2d Tests
# =============================================================================

suite "ConvTranspose2d":
  test "create ConvTranspose2d":
    let conv = newConvTranspose2d(128, 64, 3)
    check conv.inChannels == 128
    check conv.outChannels == 64
    check conv.kernelSize == (h: 3, w: 3)

  test "ConvTranspose2d weight shape":
    let conv = newConvTranspose2d(128, 64, 3)
    # Note: weight shape is (inChannels, outChannels/groups, kH, kW)
    check conv.weight.data.shape.dims == @[128, 64, 3, 3]

  test "ConvTranspose2d forward":
    let conv = newConvTranspose2d(128, 64, 3)
    let input = newTensorRef(newShape(32, 128, 14, 14), dtFloat32)
    let output = conv.forward(input)
    # Output = (14-1)*1 - 2*0 + 1*(3-1) + 0 + 1 = 16
    check output.shape.dims == @[32, 64, 16, 16]

  test "ConvTranspose2d upsample 2x":
    let conv = newConvTranspose2d(128, 64, 4, stride = 2, padding = 1)
    let input = newTensorRef(newShape(32, 128, 14, 14), dtFloat32)
    let output = conv.forward(input)
    # Output = (14-1)*2 - 2*1 + 1*(4-1) + 0 + 1 = 28
    check output.shape.dims == @[32, 64, 28, 28]

  test "ConvTranspose2d with output_padding":
    let conv = newConvTranspose2d(128, 64, 3, stride = 2, outputPadding = 1)
    let input = newTensorRef(newShape(32, 128, 14, 14), dtFloat32)
    let output = conv.forward(input)
    # Output = (14-1)*2 - 0 + 1*(3-1) + 1 + 1 = 30
    check output.shape.dims[2] == 30

# =============================================================================
# Conv3d Tests
# =============================================================================

suite "Conv3d":
  test "create Conv3d":
    let conv = newConv3d(64, 128, 3)
    check conv.inChannels == 64
    check conv.outChannels == 128
    check conv.kernelSize == (d: 3, h: 3, w: 3)

  test "Conv3d weight shape":
    let conv = newConv3d(64, 128, 3)
    check conv.weight.data.shape.dims == @[128, 64, 3, 3, 3]

  test "Conv3d forward":
    let conv = newConv3d(64, 128, 3)
    let input = newTensorRef(newShape(4, 64, 16, 28, 28), dtFloat32)
    let output = conv.forward(input)
    # Output size = (size - 3) / 1 + 1 = size - 2
    check output.shape.dims == @[4, 128, 14, 26, 26]

# =============================================================================
# DepthwiseSeparableConv2d Tests
# =============================================================================

suite "DepthwiseSeparableConv2d":
  test "create DepthwiseSeparableConv2d":
    let conv = newDepthwiseSeparableConv2d(64, 128, 3)
    check conv.inChannels == 64
    check conv.outChannels == 128

  test "DepthwiseSeparableConv2d submodules":
    let conv = newDepthwiseSeparableConv2d(64, 128, 3)
    check conv.hasModule("depthwise")
    check conv.hasModule("pointwise")

  test "DepthwiseSeparableConv2d forward":
    let conv = newDepthwiseSeparableConv2d(64, 128, 3, padding = 1)
    let input = newTensorRef(newShape(32, 64, 28, 28), dtFloat32)
    let output = conv.forward(input)
    check output.shape.dims == @[32, 128, 28, 28]

  test "DepthwiseSeparableConv2d parameter efficiency":
    let conv = newDepthwiseSeparableConv2d(64, 128, 3)
    let regularConv = newConv2d(64, 128, 3)

    let dscParams = conv.numParameters()
    let regParams = regularConv.numParameters()

    # DSC should have fewer parameters
    # DSC: 64*3*3 + 64 (depthwise) + 64*128 + 128 (pointwise)
    # Regular: 64*128*3*3 + 128
    check dscParams < regParams

# =============================================================================
# Dilated Convolution Tests
# =============================================================================

suite "Dilated Convolution":
  test "newDilatedConv2d":
    let conv = newDilatedConv2d(64, 128, 3, dilation = 2)
    check conv.dilation == (h: 2, w: 2)
    check conv.padding.h > 0  # Should have auto-calculated padding

  test "Dilated conv forward":
    let conv = newDilatedConv2d(64, 128, 3, dilation = 2)
    let input = newTensorRef(newShape(32, 64, 28, 28), dtFloat32)
    let output = conv.forward(input)
    check output.shape.dims[0] == 32
    check output.shape.dims[1] == 128

# =============================================================================
# Padding Mode Tests
# =============================================================================

suite "Padding Modes":
  test "Conv2d zeros padding":
    let conv = newConv2d(64, 128, 3, paddingMode = pmZeros)
    check conv.paddingMode == pmZeros

  test "Conv2d reflect padding":
    let conv = newConv2d(64, 128, 3, paddingMode = pmReflect)
    check conv.paddingMode == pmReflect

  test "Conv2d replicate padding":
    let conv = newConv2d(64, 128, 3, paddingMode = pmReplicate)
    check conv.paddingMode == pmReplicate

  test "Conv2d circular padding":
    let conv = newConv2d(64, 128, 3, paddingMode = pmCircular)
    check conv.paddingMode == pmCircular
