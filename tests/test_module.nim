## Tests for module base

import unittest
import std/[options, tables, strutils]
import nimml_core
import ../src/nimml_nn/module

# Simple test module
type
  TestModule = ref object of Module
    weight: Parameter
    outputShape: Shape

proc newTestModule(inputDim, outputDim: int): TestModule =
  result = TestModule()
  result.initModule("TestModule")
  result.outputShape = newShape(outputDim)
  result.weight = result.registerParameter("weight", newShape(inputDim, outputDim), dtFloat32)

method forward(m: TestModule, inputs: varargs[TensorRef]): TensorRef =
  # Stub: just return a tensor ref with output shape
  newTensorRef(m.outputShape, dtFloat32)

suite "Parameter Creation":
  test "create parameter":
    let param = newParameter("weight", newShape(10, 5), dtFloat32)
    check param.name == "weight"
    check param.data.shape == newShape(10, 5)
    check param.requiresGrad

  test "create parameter without grad":
    let param = newParameter("bias", newShape(5), dtFloat32, requiresGrad = false)
    check not param.requiresGrad

  test "create parameter from data":
    let data = newTensorData(newShape(3, 3), dtFloat32)
    let param = newParameterFromData("conv", data)
    check param.data == data

suite "Module Creation":
  test "create empty module":
    let m = newModule("test")
    check m.name == "test"
    check m.parameters.len == 0
    check m.submodules.len == 0
    check m.training
    check not m.frozen

  test "create test module":
    let m = newTestModule(10, 5)
    check m.name == "TestModule"
    check m.hasParameter("weight")

suite "Parameter Management":
  test "register parameter":
    let m = newModule("test")
    let param = m.registerParameter("weight", newShape(10, 5), dtFloat32)
    check m.hasParameter("weight")
    check param.name == "weight"

  test "get parameter":
    let m = newModule("test")
    discard m.registerParameter("weight", newShape(10), dtFloat32)

    let paramOpt = m.getParameter("weight")
    check paramOpt.isSome
    check paramOpt.get.name == "weight"

  test "get non-existent parameter":
    let m = newModule("test")
    check m.getParameter("missing").isNone

  test "parameters list":
    let m = newModule("test")
    discard m.registerParameter("weight", newShape(10, 5), dtFloat32)
    discard m.registerParameter("bias", newShape(5), dtFloat32)

    let params = m.localParameters()
    check params.len == 2

  test "set and clear gradient":
    let m = newModule("test")
    discard m.registerParameter("weight", newShape(10), dtFloat32)

    let gradRef = newTensorRef(newShape(10), dtFloat32)
    m.setGradient("weight", gradRef)

    check m.parameters["weight"].gradRef.isSome

    m.clearGradient("weight")
    check m.parameters["weight"].gradRef.isNone

suite "Submodule Management":
  test "register submodule":
    let parent = newModule("parent")
    let child = newModule("child")

    parent.registerModule("layer1", child)
    check parent.hasModule("layer1")
    check child.name == "layer1"

  test "get submodule":
    let parent = newModule("parent")
    let child = newModule("child")
    parent.registerModule("layer1", child)

    let modOpt = parent.getModule("layer1")
    check modOpt.isSome
    check modOpt.get == child

  test "modules list (recursive)":
    let parent = newModule("parent")
    let child1 = newModule()
    let child2 = newModule()
    parent.registerModule("c1", child1)
    parent.registerModule("c2", child2)

    let mods = parent.modules()
    check mods.len == 3  # parent + 2 children

  test "named modules":
    let parent = newModule("parent")
    let child = newModule()
    parent.registerModule("layer", child)

    let named = parent.namedModules()
    check named.len == 2

suite "Recursive Parameters":
  test "parameters from submodules":
    let parent = newTestModule(10, 5)
    let child = newTestModule(5, 3)
    parent.registerModule("child", child)

    let allParams = parent.parameters()
    check allParams.len == 2  # parent weight + child weight

  test "named parameters":
    let parent = newModule("parent")
    discard parent.registerParameter("weight", newShape(10), dtFloat32)

    let child = newModule()
    discard child.registerParameter("bias", newShape(5), dtFloat32)
    parent.registerModule("fc", child)

    let named = parent.namedParameters()
    check named.len == 2

    var hasParentWeight = false
    var hasChildBias = false
    for (name, param) in named:
      if name == "weight":
        hasParentWeight = true
      if name == "fc.bias":
        hasChildBias = true

    check hasParentWeight
    check hasChildBias

suite "Training Mode":
  test "train mode":
    let m = newModule("test")
    check m.isTraining()

    m.eval()
    check not m.isTraining()

    m.train()
    check m.isTraining()

  test "train mode propagates to submodules":
    let parent = newModule("parent")
    let child = newModule()
    parent.registerModule("child", child)

    parent.eval()
    check not parent.isTraining()
    check not child.isTraining()

    parent.train()
    check parent.isTraining()
    check child.isTraining()

suite "Freezing":
  test "freeze module":
    let m = newModule("test")
    check not m.isFrozen()

    m.freeze()
    check m.isFrozen()

    m.unfreeze()
    check not m.isFrozen()

  test "freeze propagates to submodules":
    let parent = newModule("parent")
    let child = newModule()
    parent.registerModule("child", child)

    parent.freeze()
    check parent.isFrozen()
    check child.isFrozen()

  test "trainable parameters excludes frozen":
    let m = newTestModule(10, 5)
    check m.trainableParameters().len == 1

    m.freeze()
    check m.trainableParameters().len == 0

suite "State Management":
  test "state dict":
    let m = newTestModule(10, 5)
    let state = m.stateDict()

    check state.parameters.len == 1
    check "weight" in state.parameters

  test "load state dict":
    let m1 = newTestModule(10, 5)
    let state = m1.stateDict()

    let m2 = newTestModule(10, 5)
    m2.loadStateDict(state)

    # Check that parameters were loaded
    check m2.parameters["weight"].data.shape == newShape(10, 5)

suite "Parameter Count":
  test "count local parameters":
    let m = newTestModule(10, 5)  # 10*5 = 50 params
    check m.numLocalParameters() == 50

  test "count all parameters":
    let parent = newTestModule(10, 5)  # 50 params
    let child = newTestModule(5, 3)    # 15 params
    parent.registerModule("child", child)

    check parent.numParameters() == 65

  test "count excludes frozen":
    let m = newTestModule(10, 5)
    m.freeze()
    check m.numParameters(countFrozen = false) == 0
    check m.numParameters(countFrozen = true) == 50

suite "Summary":
  test "module summary":
    let m = newTestModule(10, 5)
    let summary = m.summary()
    check "TestModule" in summary
    check "50 params" in summary

  test "summary with submodules":
    let parent = newTestModule(10, 5)
    let child = newTestModule(5, 3)
    parent.registerModule("fc", child)

    let summary = parent.summary()
    check "fc" in summary

suite "Sequential Container":
  test "create empty sequential":
    let seq1 = newSequential()
    check seq1.len == 0

  test "create sequential with layers":
    let layer1 = newTestModule(10, 5)
    let layer2 = newTestModule(5, 3)

    let seq1 = newSequential(layer1, layer2)
    check seq1.len == 2

  test "add layer to sequential":
    let seq1 = newSequential()
    let layer = newTestModule(10, 5)

    seq1.add(layer)
    check seq1.len == 1

  test "sequential forward":
    let layer1 = newTestModule(10, 5)
    let layer2 = newTestModule(5, 3)
    let seq1 = newSequential(layer1, layer2)

    let input = newTensorRef(newShape(10), dtFloat32)
    let output = seq1.forward(input)

    check output.shape == newShape(3)  # Output of last layer

  test "sequential parameters":
    let layer1 = newTestModule(10, 5)
    let layer2 = newTestModule(5, 3)
    let seq1 = newSequential(layer1, layer2)

    check seq1.numParameters() == 65  # 50 + 15

  test "index access":
    let layer1 = newTestModule(10, 5)
    let layer2 = newTestModule(5, 3)
    let seq1 = newSequential(layer1, layer2)

    check seq1[0] == layer1
    check seq1[1] == layer2

suite "ModuleList Container":
  test "create module list":
    let m1 = newTestModule(10, 5)
    let m2 = newTestModule(5, 3)

    let ml = newModuleList(m1, m2)
    check ml.len == 2

  test "add to module list":
    let ml = newModuleList()
    let m = newTestModule(10, 5)

    ml.add(m)
    check ml.len == 1

  test "index module list":
    let m1 = newTestModule(10, 5)
    let m2 = newTestModule(5, 3)
    let ml = newModuleList(m1, m2)

    check ml[0] == m1
    check ml[1] == m2

  test "iterate module list":
    let m1 = newTestModule(10, 5)
    let m2 = newTestModule(5, 3)
    let ml = newModuleList(m1, m2)

    var count = 0
    for m in ml:
      count.inc
    check count == 2

suite "ModuleDict Container":
  test "create module dict":
    let md = newModuleDict()
    check md.len == 0

  test "add to module dict":
    let md = newModuleDict()
    let m = newTestModule(10, 5)

    md["encoder"] = m
    check md.len == 1
    check "encoder" in md

  test "get from module dict":
    let md = newModuleDict()
    let m = newTestModule(10, 5)
    md["layer"] = m

    check md["layer"] == m

  test "iterate module dict":
    let md = newModuleDict()
    md["a"] = newTestModule(10, 5)
    md["b"] = newTestModule(5, 3)

    var count = 0
    for key, m in md:
      count.inc
    check count == 2

  test "module dict keys":
    let md = newModuleDict()
    md["encoder"] = newTestModule(10, 5)
    md["decoder"] = newTestModule(5, 3)

    let keys = md.keys
    check keys.len == 2
    check "encoder" in keys
    check "decoder" in keys

suite "Apply Function":
  test "apply to all modules":
    let parent = newModule("parent")
    let child = newModule()
    parent.registerModule("child", child)

    var visitCount = 0
    parent.apply(proc(m: Module) =
      visitCount.inc
    )

    check visitCount == 2

suite "Clear Gradients":
  test "clear all gradients recursive":
    let parent = newModule("parent")
    discard parent.registerParameter("w1", newShape(10), dtFloat32)

    let child = newModule()
    discard child.registerParameter("w2", newShape(5), dtFloat32)
    parent.registerModule("child", child)

    # Set gradients
    parent.setGradient("w1", newTensorRef(newShape(10), dtFloat32))
    child.setGradient("w2", newTensorRef(newShape(5), dtFloat32))

    # Clear all
    parent.clearAllGradients()

    check parent.parameters["w1"].gradRef.isNone
    check child.parameters["w2"].gradRef.isNone
