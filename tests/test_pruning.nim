## Pruning Utilities Tests
##
## Tests for magnitude pruning, structured pruning, game-theoretic pruning,
## Wanda pruning, and pruning context management.

import std/[tables, options]
import unittest
import ml_core
import ml_nn/module
import ml_nn/layers/dense
import ml_nn/layers/conv
import ml_nn/utils/pruning

# =============================================================================
# PruningMask Tests
# =============================================================================

suite "PruningMask":
  test "create PruningMask":
    let mask = newPruningMask(newShape(10, 10))
    check mask.shape.dims == @[10, 10]
    check mask.sparsity == 0.0

  test "calculate sparsity":
    let mask = newPruningMask(newShape(100))
    check calculateSparsity(mask) >= 0.0
    check calculateSparsity(mask) <= 1.0

  test "count nonzero":
    let mask = newPruningMask(newShape(100))
    let count = countNonzero(mask)
    check count >= 0
    check count <= 100

  test "apply mask":
    let linear = newLinear(10, 10)
    let mask = newPruningMask(linear.weight.data.shape)
    # Should not raise
    applyMask(linear.weight, mask)

  test "apply mask shape mismatch":
    let linear = newLinear(10, 10)
    let mask = newPruningMask(newShape(5, 5))
    expect(AssertionDefect):
      applyMask(linear.weight, mask)

# =============================================================================
# ImportanceScore Tests
# =============================================================================

suite "ImportanceScore":
  test "compute magnitude scores":
    let linear = newLinear(10, 5)
    let scores = computeMagnitudeScores(linear.weight)
    check scores.paramName == "weight"
    check scores.ranking.len == 50  # 10 * 5

  test "ranking order":
    let linear = newLinear(10, 5)
    let scores = computeMagnitudeScores(linear.weight)
    # Ranking should contain all indices 0..49
    var seen = newSeq[bool](50)
    for idx in scores.ranking:
      check idx >= 0 and idx < 50
      seen[idx] = true
    for i in 0..<50:
      check seen[i] == true

# =============================================================================
# Magnitude Pruning Tests
# =============================================================================

suite "MagnitudePruning":
  test "magnitude prune 0%":
    let linear = newLinear(10, 10)
    let mask = magnitudePrune(linear.weight, 0.0)
    check mask.sparsity == 0.0

  test "magnitude prune 50%":
    let linear = newLinear(10, 10)
    let mask = magnitudePrune(linear.weight, 0.5)
    check mask.sparsity == 0.5

  test "magnitude prune 90%":
    let linear = newLinear(10, 10)
    let mask = magnitudePrune(linear.weight, 0.9)
    check mask.sparsity == 0.9

  test "magnitude prune preserves shape":
    let linear = newLinear(64, 128)
    let mask = magnitudePrune(linear.weight, 0.5)
    check mask.shape == linear.weight.data.shape

# =============================================================================
# Structured Pruning Tests
# =============================================================================

suite "StructuredPruning":
  test "compute filter importance":
    let conv = newConv2d(64, 128, 3)
    let importance = computeFilterImportance(conv)
    check importance.len == 128  # Number of output filters

  test "compute filter importance non-conv":
    let linear = newLinear(10, 10)
    let importance = computeFilterImportance(linear)
    check importance.len == 0

  test "prune filters":
    let conv = newConv2d(64, 128, 3)
    let kept = pruneFilters(conv, 64)  # Keep half
    check kept.len == 64
    # Indices should be sorted
    for i in 0..<kept.len - 1:
      check kept[i] < kept[i + 1]

  test "prune filters keep all":
    let conv = newConv2d(64, 128, 3)
    let kept = pruneFilters(conv, 128)
    check kept.len == 128

  test "compute neuron importance":
    let linear = newLinear(64, 128)
    let importance = computeNeuronImportance(linear)
    check importance.len == 128

  test "prune neurons":
    let linear = newLinear(64, 128)
    let kept = pruneNeurons(linear, 64)
    check kept.len == 64

# =============================================================================
# Attention Head Pruning Tests
# =============================================================================

suite "AttentionHeadPruning":
  test "compute head importance":
    let linear = newLinear(512, 512)  # Simulated attention projection
    let importance = computeHeadImportance(linear, 8)
    check importance.len == 8

  test "prune attention heads":
    let linear = newLinear(512, 512)
    let kept = pruneAttentionHeads(linear, 8, 4)  # Keep half
    check kept.len == 4

# =============================================================================
# Game-Theoretic Pruning Tests
# =============================================================================

suite "GameTheoreticPruning":
  test "create game theoretic pruner":
    let pruner = newGameTheoreticPruner()
    check pruner.numSamples == 1000

  test "create game theoretic pruner custom samples":
    let pruner = newGameTheoreticPruner(500)
    check pruner.numSamples == 500

  test "approximate shapley values":
    let linear = newLinear(5, 5)  # Small for testing
    let pruner = newGameTheoreticPruner(10)  # Few samples for speed

    proc valueFn(mask: seq[bool]): float =
      var count = 0
      for m in mask:
        if m: count += 1
      count.float / mask.len.float

    let shapley = pruner.approximateShapleyValues(linear.weight, valueFn)
    check shapley.len == 25  # 5 * 5

  test "game theoretic prune":
    let linear = newLinear(5, 5)

    proc valueFn(mask: seq[bool]): float =
      var count = 0
      for m in mask:
        if m: count += 1
      count.float

    let mask = gameTheoreticPrune(linear.weight, 0.5, valueFn)
    check mask.sparsity == 0.5

# =============================================================================
# Gradual Magnitude Pruning Tests
# =============================================================================

suite "GradualMagnitudePruning":
  test "create gradual magnitude pruner":
    let gmp = newGradualMagnitudePruner()
    check gmp.initialSparsity == 0.0
    check gmp.finalSparsity == 0.9
    check gmp.beginStep == 0
    check gmp.endStep == 1000
    check gmp.frequency == 100
    check gmp.schedule == pschCubic

  test "create gradual magnitude pruner custom":
    let gmp = newGradualMagnitudePruner(
      initialSparsity = 0.0,
      finalSparsity = 0.8,
      beginStep = 100,
      endStep = 500,
      frequency = 50,
      schedule = pschIterative
    )
    check gmp.finalSparsity == 0.8
    check gmp.beginStep == 100
    check gmp.endStep == 500
    check gmp.schedule == pschIterative

  test "compute target sparsity one shot":
    let sparsity = computeTargetSparsity(pschOneShot, 100, 100, 0.0, 0.9)
    check sparsity == 0.9

  test "compute target sparsity one shot before end":
    let sparsity = computeTargetSparsity(pschOneShot, 50, 100, 0.0, 0.9)
    check sparsity == 0.0

  test "compute target sparsity iterative":
    let sparsity = computeTargetSparsity(pschIterative, 50, 100, 0.0, 0.9)
    # At t=0.5, should be 0.45
    check sparsity > 0.4
    check sparsity < 0.5

  test "compute target sparsity cubic":
    let sparsity = computeTargetSparsity(pschCubic, 50, 100, 0.0, 0.9)
    # Cubic schedule
    check sparsity >= 0.0
    check sparsity <= 0.9

  test "gradual magnitude pruner step before begin":
    let gmp = newGradualMagnitudePruner(beginStep = 100)
    let linear = newLinear(10, 10)
    gmp.step(linear, 50)  # Should do nothing

  test "gradual magnitude pruner step after end":
    let gmp = newGradualMagnitudePruner(endStep = 100)
    let linear = newLinear(10, 10)
    gmp.step(linear, 150)  # Should do nothing

  test "gradual magnitude pruner step at frequency":
    let gmp = newGradualMagnitudePruner(
      beginStep = 0,
      endStep = 100,
      frequency = 10
    )
    let linear = newLinear(10, 10)
    gmp.step(linear, 10)  # Should prune

# =============================================================================
# Layer Sensitivity Analysis Tests
# =============================================================================

suite "LayerSensitivityAnalysis":
  test "analyze layer sensitivity":
    let linear = newLinear(10, 10)

    proc evalFn(): float =
      0.95  # Dummy accuracy

    let sensitivity = analyzeLayerSensitivity(linear, evalFn, @[0.1, 0.2, 0.3])
    check sensitivity.len > 0

  test "optimal layer sparsities":
    var sensitivity = initTable[string, seq[float]]()
    sensitivity["layer1"] = @[0.01, 0.02, 0.05]  # Low sensitivity
    sensitivity["layer2"] = @[0.1, 0.2, 0.5]    # High sensitivity

    let optimal = optimalLayerSparsities(sensitivity, @[0.1, 0.2, 0.3], 0.3)
    check optimal.len == 2
    # Layer1 should have higher sparsity (lower sensitivity)
    check optimal["layer1"] >= optimal["layer2"]

# =============================================================================
# Wanda Pruning Tests
# =============================================================================

suite "WandaPruning":
  test "create wanda pruner":
    let wanda = newWandaPruner()
    check wanda.activationScales.len == 0

  test "record activation scales":
    let wanda = newWandaPruner()
    let activations = newTensorData(newShape(32, 64), dtFloat32)
    wanda.recordActivationScales("layer1", activations)
    check wanda.activationScales.len == 1

  test "wanda prune":
    let wanda = newWandaPruner()
    let linear = newLinear(10, 10)
    let mask = wanda.wandaPrune(linear.weight, "weight", 0.5)
    check mask.sparsity == 0.5

  test "wanda prune preserves shape":
    let wanda = newWandaPruner()
    let linear = newLinear(64, 128)
    let mask = wanda.wandaPrune(linear.weight, "weight", 0.3)
    check mask.shape == linear.weight.data.shape

# =============================================================================
# Global/Local Pruning Tests
# =============================================================================

suite "GlobalLocalPruning":
  test "global magnitude prune":
    let linear = newLinear(10, 10)
    let res = globalMagnitudePrune(linear, 0.5)
    check res.sparsity > 0.0
    check res.totalParams == 110  # 10*10 + 10 bias

  test "local magnitude prune":
    let linear = newLinear(10, 10)
    let res = localMagnitudePrune(linear, 0.5)
    check res.sparsity > 0.0

  test "global prune compression ratio":
    let linear = newLinear(100, 100)
    let res = globalMagnitudePrune(linear, 0.9)
    check res.compressionRatio > 1.0

  test "local prune layer sparsities":
    let linear = newLinear(10, 10)
    let res = localMagnitudePrune(linear, 0.5)
    check res.layerSparsities.len > 0

# =============================================================================
# PruningContext Tests
# =============================================================================

suite "PruningContext":
  test "create pruning context":
    let linear = newLinear(10, 10)
    let ctx = newPruningContext(linear, 0.5)
    check ctx.targetSparsity == 0.5
    check ctx.currentSparsity == 0.0
    check ctx.pruningMethod == pmMagnitude
    check ctx.pruningScope == psLocal
    check ctx.pruningSchedule == pschOneShot

  test "create pruning context custom":
    let linear = newLinear(10, 10)
    let ctx = newPruningContext(
      linear,
      targetSparsity = 0.8,
      pruningMethod = pmGradient,
      pruningScope = psGlobal,
      pruningSchedule = pschCubic,
      totalSteps = 100
    )
    check ctx.targetSparsity == 0.8
    check ctx.pruningMethod == pmGradient
    check ctx.pruningScope == psGlobal
    check ctx.pruningSchedule == pschCubic
    check ctx.totalSteps == 100

  test "pruning context step":
    let linear = newLinear(10, 10)
    let ctx = newPruningContext(linear, 0.5, totalSteps = 10)
    ctx.step()
    check ctx.currentStep == 1

  test "pruning context multiple steps":
    let linear = newLinear(10, 10)
    let ctx = newPruningContext(linear, 0.5, totalSteps = 10)
    for _ in 1..5:
      ctx.step()
    check ctx.currentStep == 5

  test "pruning context finalize":
    let linear = newLinear(10, 10)
    let ctx = newPruningContext(linear, 0.5, totalSteps = 1)
    ctx.step()
    let res = ctx.finalize()
    check res.totalParams > 0

  test "pruning context masks initialized":
    let linear = newLinear(10, 10)
    let ctx = newPruningContext(linear, 0.5)
    check ctx.masks.len > 0

# =============================================================================
# Pruning Schedule Tests
# =============================================================================

suite "PruningSchedule":
  test "one shot schedule":
    let linear = newLinear(10, 10)
    let ctx = newPruningContext(
      linear, 0.5,
      pruningSchedule = pschOneShot,
      totalSteps = 10
    )
    ctx.step()
    # One shot should go to 0 until final step

  test "iterative schedule":
    let linear = newLinear(10, 10)
    let ctx = newPruningContext(
      linear, 0.5,
      pruningSchedule = pschIterative,
      totalSteps = 10
    )
    for _ in 1..5:
      ctx.step()
    # Should be approximately at 25% sparsity

  test "cubic schedule":
    let linear = newLinear(10, 10)
    let ctx = newPruningContext(
      linear, 0.5,
      pruningSchedule = pschCubic,
      totalSteps = 10
    )
    for _ in 1..10:
      ctx.step()
    check ctx.currentSparsity > 0.0

  test "exponential schedule":
    let sparsity = computeTargetSparsity(pschExponential, 50, 100, 0.01, 0.9)
    check sparsity > 0.01
    check sparsity < 0.9

# =============================================================================
# PruningResult Tests
# =============================================================================

suite "PruningResult":
  test "pruning result fields":
    var res = PruningResult(
      paramsPruned: 500,
      totalParams: 1000,
      sparsity: 0.5,
      compressionRatio: 2.0,
      layerSparsities: initTable[string, float]()
    )
    check res.paramsPruned == 500
    check res.totalParams == 1000
    check res.sparsity == 0.5
    check res.compressionRatio == 2.0

  test "print pruning report":
    var layerSparsities = initTable[string, float]()
    layerSparsities["weight"] = 0.5
    layerSparsities["bias"] = 0.3
    var res = PruningResult(
      paramsPruned: 500,
      totalParams: 1000,
      sparsity: 0.5,
      compressionRatio: 2.0,
      layerSparsities: layerSparsities
    )
    # Should not raise
    printPruningReport(res)

# =============================================================================
# Utility Function Tests
# =============================================================================

suite "PruningUtilities":
  test "estimate speedup theoretical":
    check estimateSpeedupTheoretical(0.0) == 1.0
    check estimateSpeedupTheoretical(0.5) == 2.0
    check estimateSpeedupTheoretical(0.9) > 9.0

  test "estimate memory savings dense":
    let savings = estimateMemorySavings(0.5, false)
    check savings == 0.0  # No savings for dense storage

  test "estimate memory savings sparse":
    let savings = estimateMemorySavings(0.9, true)
    check savings > 0.0  # Some savings at high sparsity

  test "estimate memory savings low sparsity":
    let savings = estimateMemorySavings(0.1, true)
    # At low sparsity, sparse storage might be worse
    check savings >= 0.0

# =============================================================================
# Integration Tests
# =============================================================================

suite "PruningIntegration":
  test "full pruning workflow":
    # Create a simple model
    let linear = newLinear(64, 128)

    # Create pruning context
    let ctx = newPruningContext(
      linear,
      targetSparsity = 0.5,
      pruningMethod = pmMagnitude,
      pruningScope = psLocal,
      pruningSchedule = pschIterative,
      totalSteps = 10
    )

    # Run pruning steps
    for _ in 1..10:
      ctx.step()

    # Finalize and get report
    let res = ctx.finalize()
    check res.totalParams > 0

  test "prune then evaluate":
    let linear = newLinear(10, 10)

    # Prune
    let res = globalMagnitudePrune(linear, 0.5)

    # Model should still work
    let input = newTensorRef(newShape(32, 10), dtFloat32)
    let output = linear.forward(input)
    check output.shape.dims == @[32, 10]

  test "multiple pruning methods comparison":
    let linear1 = newLinear(64, 64)
    let linear2 = newLinear(64, 64)

    let res1 = globalMagnitudePrune(linear1, 0.5)
    let res2 = localMagnitudePrune(linear2, 0.5)

    # Both should prune similar amounts
    check res1.sparsity > 0.0
    check res2.sparsity > 0.0

# =============================================================================
# Edge Cases Tests
# =============================================================================

suite "PruningEdgeCases":
  test "prune 0% sparsity":
    let linear = newLinear(10, 10)
    let res = globalMagnitudePrune(linear, 0.0)
    check res.paramsPruned == 0

  test "prune nearly 100% sparsity":
    let linear = newLinear(10, 10)
    let mask = magnitudePrune(linear.weight, 0.99)
    check mask.sparsity == 0.99

  test "prune empty module":
    let linear = newLinear(10, 10, useBias = false)
    let res = globalMagnitudePrune(linear, 0.5)
    check res.totalParams == 100  # Just weights

  test "prune small layer":
    let linear = newLinear(2, 2)
    let mask = magnitudePrune(linear.weight, 0.5)
    check mask.shape.dims == @[2, 2]

  test "prune large layer":
    let linear = newLinear(1024, 1024)
    let mask = magnitudePrune(linear.weight, 0.5)
    check mask.sparsity == 0.5
