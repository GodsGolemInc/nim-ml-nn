## Normalization Layer Tests
##
## Tests for BatchNorm, LayerNorm, GroupNorm, RMSNorm, etc.

import std/options
import unittest
import ml_core
import ml_nn/module
import ml_nn/layers/norm

# =============================================================================
# BatchNorm Tests
# =============================================================================

suite "BatchNorm1d":
  test "create BatchNorm1d":
    let bn = newBatchNorm1d(64)
    check bn.numFeatures == 64
    check bn.eps == 1e-5
    check bn.momentum == 0.1
    check bn.affine == true
    check bn.trackRunningStats == true

  test "BatchNorm1d parameters":
    let bn = newBatchNorm1d(64)
    check bn.weight.isSome
    check bn.bias.isSome
    check bn.parameters().len == 2

  test "BatchNorm1d without affine":
    let bn = newBatchNorm1d(64, affine = false)
    check bn.weight.isNone
    check bn.bias.isNone
    check bn.parameters().len == 0

  test "BatchNorm1d forward 2D":
    let bn = newBatchNorm1d(64)
    let input = newTensorRef(newShape(32, 64), dtFloat32)
    let output = bn.forward(input)
    check output.shape.dims == @[32, 64]

  test "BatchNorm1d forward 3D":
    let bn = newBatchNorm1d(64)
    let input = newTensorRef(newShape(32, 64, 100), dtFloat32)
    let output = bn.forward(input)
    check output.shape.dims == @[32, 64, 100]

  test "BatchNorm1d wrong channels":
    let bn = newBatchNorm1d(64)
    let input = newTensorRef(newShape(32, 32), dtFloat32)
    expect(ModuleError):
      discard bn.forward(input)

suite "BatchNorm2d":
  test "create BatchNorm2d":
    let bn = newBatchNorm2d(64)
    check bn.numFeatures == 64

  test "BatchNorm2d forward":
    let bn = newBatchNorm2d(64)
    let input = newTensorRef(newShape(32, 64, 28, 28), dtFloat32)
    let output = bn.forward(input)
    check output.shape.dims == @[32, 64, 28, 28]

  test "BatchNorm2d wrong dims":
    let bn = newBatchNorm2d(64)
    let input = newTensorRef(newShape(32, 64, 28), dtFloat32)
    expect(ModuleError):
      discard bn.forward(input)

suite "BatchNorm3d":
  test "create BatchNorm3d":
    let bn = newBatchNorm3d(64)
    check bn.numFeatures == 64

  test "BatchNorm3d forward":
    let bn = newBatchNorm3d(64)
    let input = newTensorRef(newShape(32, 64, 8, 28, 28), dtFloat32)
    let output = bn.forward(input)
    check output.shape.dims == @[32, 64, 8, 28, 28]

# =============================================================================
# LayerNorm Tests
# =============================================================================

suite "LayerNorm":
  test "create LayerNorm 1D":
    let ln = newLayerNorm(768)
    check ln.normalizedShape == @[768]
    check ln.eps == 1e-5
    check ln.elementwiseAffine == true

  test "create LayerNorm multi-dim":
    let ln = newLayerNorm(@[512, 512])
    check ln.normalizedShape == @[512, 512]

  test "LayerNorm parameters":
    let ln = newLayerNorm(768)
    check ln.weight.isSome
    check ln.bias.isSome
    check ln.parameters().len == 2

  test "LayerNorm without affine":
    let ln = newLayerNorm(768, elementwiseAffine = false)
    check ln.weight.isNone
    check ln.bias.isNone

  test "LayerNorm forward":
    let ln = newLayerNorm(768)
    let input = newTensorRef(newShape(32, 100, 768), dtFloat32)
    let output = ln.forward(input)
    check output.shape.dims == @[32, 100, 768]

  test "LayerNorm forward 2D":
    let ln = newLayerNorm(768)
    let input = newTensorRef(newShape(32, 768), dtFloat32)
    let output = ln.forward(input)
    check output.shape.dims == @[32, 768]

  test "LayerNorm shape mismatch":
    let ln = newLayerNorm(768)
    let input = newTensorRef(newShape(32, 512), dtFloat32)
    expect(ModuleError):
      discard ln.forward(input)

# =============================================================================
# GroupNorm Tests
# =============================================================================

suite "GroupNorm":
  test "create GroupNorm":
    let gn = newGroupNorm(8, 64)
    check gn.numGroups == 8
    check gn.numChannels == 64
    check gn.eps == 1e-5
    check gn.affine == true

  test "GroupNorm not divisible":
    expect(ModuleError):
      discard newGroupNorm(8, 65)

  test "GroupNorm parameters":
    let gn = newGroupNorm(8, 64)
    check gn.weight.isSome
    check gn.bias.isSome

  test "GroupNorm forward":
    let gn = newGroupNorm(8, 64)
    let input = newTensorRef(newShape(32, 64, 28, 28), dtFloat32)
    let output = gn.forward(input)
    check output.shape.dims == @[32, 64, 28, 28]

  test "GroupNorm channel mismatch":
    let gn = newGroupNorm(8, 64)
    let input = newTensorRef(newShape(32, 32, 28, 28), dtFloat32)
    expect(ModuleError):
      discard gn.forward(input)

# =============================================================================
# InstanceNorm Tests
# =============================================================================

suite "InstanceNorm1d":
  test "create InstanceNorm1d":
    let inst = newInstanceNorm1d(64)
    check inst.numFeatures == 64
    check inst.affine == false  # Default is false for InstanceNorm

  test "InstanceNorm1d with affine":
    let inst = newInstanceNorm1d(64, affine = true)
    check inst.affine == true
    check inst.weight.isSome

  test "InstanceNorm1d forward":
    let inst = newInstanceNorm1d(64)
    let input = newTensorRef(newShape(32, 64, 100), dtFloat32)
    let output = inst.forward(input)
    check output.shape.dims == @[32, 64, 100]

suite "InstanceNorm2d":
  test "create InstanceNorm2d":
    let inst = newInstanceNorm2d(64)
    check inst.numFeatures == 64

  test "InstanceNorm2d forward":
    let inst = newInstanceNorm2d(64)
    let input = newTensorRef(newShape(32, 64, 28, 28), dtFloat32)
    let output = inst.forward(input)
    check output.shape.dims == @[32, 64, 28, 28]

# =============================================================================
# RMSNorm Tests
# =============================================================================

suite "RMSNorm":
  test "create RMSNorm":
    let rms = newRMSNorm(768)
    check rms.normalizedShape == @[768]
    check rms.eps == 1e-6

  test "RMSNorm multi-dim":
    let rms = newRMSNorm(@[512, 512])
    check rms.normalizedShape == @[512, 512]

  test "RMSNorm parameters":
    let rms = newRMSNorm(768)
    check rms.weight.isSome
    # RMSNorm has no bias
    check rms.parameters().len == 1

  test "RMSNorm forward":
    let rms = newRMSNorm(768)
    let input = newTensorRef(newShape(32, 100, 768), dtFloat32)
    let output = rms.forward(input)
    check output.shape.dims == @[32, 100, 768]

  test "RMSNorm shape mismatch":
    let rms = newRMSNorm(768)
    let input = newTensorRef(newShape(32, 512), dtFloat32)
    expect(ModuleError):
      discard rms.forward(input)

# =============================================================================
# LocalResponseNorm Tests
# =============================================================================

suite "LocalResponseNorm":
  test "create LocalResponseNorm":
    let lrn = newLocalResponseNorm(5)
    check lrn.size == 5
    check lrn.alpha == 0.0001
    check lrn.beta == 0.75
    check lrn.k == 1.0

  test "LocalResponseNorm custom params":
    let lrn = newLocalResponseNorm(3, alpha = 0.0002, beta = 0.5, k = 2.0)
    check lrn.size == 3
    check lrn.alpha == 0.0002
    check lrn.beta == 0.5
    check lrn.k == 2.0

  test "LocalResponseNorm forward":
    let lrn = newLocalResponseNorm(5)
    let input = newTensorRef(newShape(32, 64, 28, 28), dtFloat32)
    let output = lrn.forward(input)
    check output.shape == input.shape

# =============================================================================
# SyncBatchNorm Tests
# =============================================================================

suite "SyncBatchNorm":
  test "create SyncBatchNorm":
    let sbn = newSyncBatchNorm(64)
    check sbn.numFeatures == 64
    check sbn.processGroup == "default"

  test "SyncBatchNorm custom group":
    let sbn = newSyncBatchNorm(64, processGroup = "custom")
    check sbn.processGroup == "custom"

  test "SyncBatchNorm forward":
    let sbn = newSyncBatchNorm(64)
    let input = newTensorRef(newShape(32, 64, 28, 28), dtFloat32)
    let output = sbn.forward(input)
    check output.shape.dims == @[32, 64, 28, 28]

# =============================================================================
# Training Mode Tests
# =============================================================================

suite "Norm Training Mode":
  test "BatchNorm training mode":
    let bn = newBatchNorm2d(64)
    check bn.training == true

    bn.eval()
    check bn.training == false

    bn.train()
    check bn.training == true

  test "LayerNorm training mode":
    let ln = newLayerNorm(768)
    check ln.training == true
    ln.eval()
    check ln.training == false
