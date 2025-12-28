## Module Base
##
## Base class for all neural network components.
## Provides parameter management and hierarchical structure.

import std/[tables, options, strutils, hashes]
import nimml_core

type
  ModuleError* = object of CatchableError
    ## Error in module operations

  Parameter* = object
    ## A trainable parameter
    name*: string
    tensorRef*: TensorRef
    data*: TensorData
    requiresGrad*: bool
    gradRef*: Option[TensorRef]

  Module* = ref object of RootObj
    ## Base class for all neural network modules
    name*: string
    parameters*: OrderedTable[string, Parameter]
    submodules*: OrderedTable[string, Module]
    training*: bool
    frozen*: bool

  ModuleState* = object
    ## Serializable state of a module
    parameters*: Table[string, TensorData]
    buffers*: Table[string, TensorData]

# Parameter constructors

proc newParameter*(name: string, shape: Shape, dtype: DType,
                   requiresGrad: bool = true): Parameter =
  ## Create a new parameter
  let data = newTensorData(shape, dtype)
  let tensorRef = newTensorRef(data)
  Parameter(
    name: name,
    tensorRef: tensorRef,
    data: data,
    requiresGrad: requiresGrad,
    gradRef: none(TensorRef)
  )

proc newParameterFromData*(name: string, data: TensorData,
                           requiresGrad: bool = true): Parameter =
  ## Create parameter from existing data
  Parameter(
    name: name,
    tensorRef: newTensorRef(data),
    data: data,
    requiresGrad: requiresGrad,
    gradRef: none(TensorRef)
  )

# Module constructors

proc newModule*(name: string = ""): Module =
  ## Create a new module
  Module(
    name: name,
    parameters: initOrderedTable[string, Parameter](),
    submodules: initOrderedTable[string, Module](),
    training: true,
    frozen: false
  )

proc initModule*(m: Module, name: string = "") =
  ## Initialize module fields (for subclasses)
  m.name = name
  m.parameters = initOrderedTable[string, Parameter]()
  m.submodules = initOrderedTable[string, Module]()
  m.training = true
  m.frozen = false

# Module lifecycle (virtual methods)

method init*(m: Module) {.base.} =
  ## Initialize module (allocate parameters)
  discard

method forward*(m: Module, inputs: varargs[TensorRef]): TensorRef {.base.} =
  ## Forward pass - must be implemented by subclasses
  raise newException(ModuleError, "forward() not implemented for " & m.name)

method reset*(m: Module) {.base.} =
  ## Reset parameters to initial values
  discard

# Parameter management

proc registerParameter*(m: Module, name: string, shape: Shape,
                        dtype: DType, requiresGrad: bool = true): Parameter =
  ## Register a new parameter with the module
  let param = newParameter(name, shape, dtype, requiresGrad)
  m.parameters[name] = param
  param

proc registerParameterData*(m: Module, name: string, data: TensorData,
                            requiresGrad: bool = true): Parameter =
  ## Register parameter from existing data
  let param = newParameterFromData(name, data, requiresGrad)
  m.parameters[name] = param
  param

proc getParameter*(m: Module, name: string): Option[Parameter] =
  ## Get a parameter by name
  if name in m.parameters:
    some(m.parameters[name])
  else:
    none(Parameter)

proc hasParameter*(m: Module, name: string): bool =
  ## Check if parameter exists
  name in m.parameters

proc setGradient*(m: Module, name: string, gradRef: TensorRef) =
  ## Set gradient for a parameter
  if name in m.parameters:
    m.parameters[name].gradRef = some(gradRef)

proc clearGradient*(m: Module, name: string) =
  ## Clear gradient for a parameter
  if name in m.parameters:
    m.parameters[name].gradRef = none(TensorRef)

proc clearAllGradients*(m: Module) =
  ## Clear all gradients
  for name in m.parameters.keys:
    m.parameters[name].gradRef = none(TensorRef)
  for submodule in m.submodules.values:
    submodule.clearAllGradients()

proc localParameters*(m: Module): seq[Parameter] =
  ## Get parameters of this module (non-recursive)
  for param in m.parameters.values:
    result.add(param)

proc parameters*(m: Module): seq[Parameter] =
  ## Get all parameters (recursive)
  result = m.localParameters()
  for submodule in m.submodules.values:
    result.add(submodule.parameters())

proc namedParameters*(m: Module, prefix: string = ""): seq[(string, Parameter)] =
  ## Get all parameters with full paths
  let actualPrefix = if prefix.len > 0: prefix & "." else: ""

  for name, param in m.parameters.pairs:
    result.add((actualPrefix & name, param))

  for subName, submodule in m.submodules.pairs:
    result.add(submodule.namedParameters(actualPrefix & subName))

proc trainableParameters*(m: Module): seq[Parameter] =
  ## Get only trainable parameters
  for param in m.parameters():
    if param.requiresGrad and not m.frozen:
      result.add(param)

# Submodule management

proc registerModule*(m: Module, name: string, submodule: Module) =
  ## Register a submodule
  if submodule.isNil:
    return
  submodule.name = name
  m.submodules[name] = submodule

proc getModule*(m: Module, name: string): Option[Module] =
  ## Get a submodule by name
  if name in m.submodules:
    some(m.submodules[name])
  else:
    none(Module)

proc hasModule*(m: Module, name: string): bool =
  ## Check if submodule exists
  name in m.submodules

proc localModules*(m: Module): seq[Module] =
  ## Get direct submodules (non-recursive)
  for submodule in m.submodules.values:
    result.add(submodule)

proc modules*(m: Module): seq[Module] =
  ## Get all submodules (recursive)
  result.add(m)
  for submodule in m.submodules.values:
    result.add(submodule.modules())

proc namedModules*(m: Module, prefix: string = ""): seq[(string, Module)] =
  ## Get all submodules with full paths
  let actualPrefix = if prefix.len > 0: prefix & "." else: ""

  let selfName = if prefix.len > 0: prefix else: m.name
  result.add((selfName, m))

  for subName, submodule in m.submodules.pairs:
    result.add(submodule.namedModules(actualPrefix & subName))

# Training mode

proc train*(m: Module, mode: bool = true) =
  ## Set training mode (affects dropout, batchnorm, etc.)
  m.training = mode
  for submodule in m.submodules.values:
    submodule.train(mode)

proc eval*(m: Module) =
  ## Set evaluation mode
  m.train(false)

proc isTraining*(m: Module): bool =
  ## Check if in training mode
  m.training

# Freezing

proc freeze*(m: Module) =
  ## Freeze all parameters (no gradient update)
  m.frozen = true
  for submodule in m.submodules.values:
    submodule.freeze()

proc unfreeze*(m: Module) =
  ## Unfreeze parameters
  m.frozen = false
  for submodule in m.submodules.values:
    submodule.unfreeze()

proc isFrozen*(m: Module): bool =
  ## Check if frozen
  m.frozen

# State management

proc stateDict*(m: Module): ModuleState =
  ## Get module state for saving
  result = ModuleState(
    parameters: initTable[string, TensorData](),
    buffers: initTable[string, TensorData]()
  )

  for (name, param) in m.namedParameters():
    result.parameters[name] = param.data

proc loadStateDict*(m: Module, state: ModuleState) =
  ## Load module state
  for name, data in state.parameters.pairs:
    # Find the parameter and update its data
    let parts = name.split(".")
    var current = m

    for i, part in parts:
      if i == parts.len - 1:
        # Final part is parameter name
        if part in current.parameters:
          current.parameters[part].data = data
          current.parameters[part].tensorRef = newTensorRef(data)
      else:
        # Navigate to submodule
        if part in current.submodules:
          current = current.submodules[part]

# Device/dtype conversion

proc toDType*(m: Module, dtype: DType) =
  ## Convert all parameters to dtype
  # Note: This is a placeholder - actual implementation would
  # need tensor type conversion
  discard

proc apply*(m: Module, fn: proc(module: Module)) =
  ## Apply function to all modules
  fn(m)
  for submodule in m.submodules.values:
    submodule.apply(fn)

# Summary and statistics

proc numLocalParameters*(m: Module): int =
  ## Count parameters in this module only
  for param in m.parameters.values:
    var size = 1
    for dim in param.data.shape.dims:
      size = size * dim
    result += size

proc numParameters*(m: Module, countFrozen: bool = true): int =
  ## Total parameter count
  if not countFrozen and m.frozen:
    return 0

  result = m.numLocalParameters()
  for submodule in m.submodules.values:
    result += submodule.numParameters(countFrozen)

proc summary*(m: Module, indent: int = 0): string =
  ## Human-readable summary
  let prefix = "  ".repeat(indent)
  result = prefix & m.name
  if m.name.len == 0:
    result = prefix & "(unnamed)"

  let paramCount = m.numLocalParameters()
  if paramCount > 0:
    result &= " [" & $paramCount & " params]"

  if m.frozen:
    result &= " (frozen)"

  result &= "\n"

  for subName, submodule in m.submodules.pairs:
    result &= prefix & "  " & subName & ":\n"
    result &= submodule.summary(indent + 2)

# Container Modules

type
  Sequential* = ref object of Module
    ## Sequential container - applies layers in order
    layers*: seq[Module]

  ModuleList* = ref object of Module
    ## List of modules (not applied sequentially)
    moduleList*: seq[Module]

  ModuleDict* = ref object of Module
    ## Dictionary of modules
    moduleDict*: Table[string, Module]

# Sequential

proc newSequential*(layers: varargs[Module]): Sequential =
  ## Create a Sequential container
  result = Sequential()
  result.initModule("Sequential")
  result.layers = @[]

  for i, layer in layers:
    let name = $i
    result.layers.add(layer)
    result.registerModule(name, layer)

proc add*(s: Sequential, layer: Module) =
  ## Add a layer to the sequential
  let name = $s.layers.len
  s.layers.add(layer)
  s.registerModule(name, layer)

proc len*(s: Sequential): int =
  ## Number of layers
  s.layers.len

proc `[]`*(s: Sequential, i: int): Module =
  ## Get layer by index
  s.layers[i]

method forward*(s: Sequential, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass through all layers in sequence
  if s.layers.len == 0:
    if inputs.len > 0:
      return inputs[0]
    return nil

  var x = s.layers[0].forward(inputs)
  for i in 1..<s.layers.len:
    x = s.layers[i].forward(x)
  x

# ModuleList

proc newModuleList*(modules: varargs[Module]): ModuleList =
  ## Create a ModuleList
  result = ModuleList()
  result.initModule("ModuleList")
  result.moduleList = @[]

  for i, m in modules:
    result.moduleList.add(m)
    result.registerModule($i, m)

proc add*(ml: ModuleList, m: Module) =
  ## Add module to list
  let name = $ml.moduleList.len
  ml.moduleList.add(m)
  ml.registerModule(name, m)

proc len*(ml: ModuleList): int =
  ## Number of modules
  ml.moduleList.len

proc `[]`*(ml: ModuleList, i: int): Module =
  ## Get module by index
  ml.moduleList[i]

method forward*(ml: ModuleList, inputs: varargs[TensorRef]): TensorRef =
  ## Forward not defined for ModuleList (use modules individually)
  raise newException(ModuleError, "ModuleList does not support forward()")

iterator items*(ml: ModuleList): Module =
  ## Iterate over modules
  for m in ml.moduleList:
    yield m

# ModuleDict

proc newModuleDict*(): ModuleDict =
  ## Create a ModuleDict
  result = ModuleDict()
  result.initModule("ModuleDict")
  result.moduleDict = initTable[string, Module]()

proc `[]`*(md: ModuleDict, key: string): Module =
  ## Get module by key
  md.moduleDict[key]

proc `[]=`*(md: ModuleDict, key: string, m: Module) =
  ## Set module by key
  md.moduleDict[key] = m
  md.registerModule(key, m)

proc contains*(md: ModuleDict, key: string): bool =
  ## Check if key exists
  key in md.moduleDict

proc len*(md: ModuleDict): int =
  ## Number of modules
  md.moduleDict.len

proc keys*(md: ModuleDict): seq[string] =
  ## Get all keys
  for key in md.moduleDict.keys:
    result.add(key)

method forward*(md: ModuleDict, inputs: varargs[TensorRef]): TensorRef =
  ## Forward not defined for ModuleDict (use modules individually)
  raise newException(ModuleError, "ModuleDict does not support forward()")

iterator pairs*(md: ModuleDict): (string, Module) =
  ## Iterate over key-module pairs
  for key, m in md.moduleDict.pairs:
    yield (key, m)
