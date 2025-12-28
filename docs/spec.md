# nim-ml-nn Specification

## Overview

ニューラルネットワークの構成要素（レイヤー、損失関数、オプティマイザ）を提供する。
Moduleパターンで構成可能なネットワークを構築。

---

## Module Structure

```
nim-ml-nn/
├── src/
│   ├── nimml_nn.nim             # エントリポイント
│   └── nimml_nn/
│       ├── module.nim           # Moduleインターフェース
│       ├── layers/
│       │   ├── dense.nim
│       │   ├── conv.nim
│       │   ├── norm.nim
│       │   ├── activation.nim
│       │   ├── dropout.nim
│       │   ├── embedding.nim
│       │   └── lora.nim         # LoRA adapter
│       ├── loss.nim
│       └── optim.nim
└── nimml_nn.nimble
```

---

## 1. Module (`nimml_nn/module.nim`)

### Purpose

全NNコンポーネントの基底クラス。パラメータ管理と階層構造を提供。

### Types

```nim
type
  Parameter* = object
    name*: string
    tensorRef*: TensorRef
    data*: TensorData
    requiresGrad*: bool
    gradRef*: Option[TensorRef]

  Module* = ref object of RootObj
    name*: string
    parameters*: OrderedTable[string, Parameter]
    submodules*: OrderedTable[string, Module]
    training*: bool
    frozen*: bool

  ModuleState* = object
    parameters*: Table[string, TensorData]
    buffers*: Table[string, TensorData]
```

### API

```nim
# Module lifecycle
method init*(m: Module): void {.base.}
  ## Initialize module (allocate parameters).

method forward*(m: Module, inputs: varargs[TensorRef]): TensorRef {.base.}
  ## Forward pass. Must be implemented by subclasses.

method reset*(m: Module): void {.base.}
  ## Reset parameters to initial values.

# Parameter management
proc registerParameter*(m: Module, name: string, shape: Shape,
                        dtype: DType, requiresGrad: bool = true): Parameter
proc getParameter*(m: Module, name: string): Option[Parameter]
proc parameters*(m: Module): seq[Parameter]
  ## Get all parameters (recursive).
proc namedParameters*(m: Module): seq[(string, Parameter)]
  ## Get all parameters with full paths.

# Submodule management
proc registerModule*(m: Module, name: string, submodule: Module): void
proc getModule*(m: Module, name: string): Option[Module]
proc modules*(m: Module): seq[Module]
  ## Get all submodules (recursive).
proc namedModules*(m: Module): seq[(string, Module)]

# Training mode
proc train*(m: Module, mode: bool = true): void
  ## Set training mode (affects dropout, batchnorm, etc.).
proc eval*(m: Module): void
  ## Set evaluation mode.
proc isTraining*(m: Module): bool

# Freezing
proc freeze*(m: Module): void
  ## Freeze all parameters (no gradient update).
proc unfreeze*(m: Module): void
proc isFrozen*(m: Module): bool

# State management
proc stateDict*(m: Module): ModuleState
  ## Get module state (for saving).
proc loadStateDict*(m: Module, state: ModuleState): void
  ## Load module state.

# Device/dtype
proc to*(m: Module, dtype: DType): void
  ## Convert all parameters to dtype.

proc apply*(m: Module, fn: proc(m: Module)): void
  ## Apply function to all modules.

# Summary
proc summary*(m: Module): string
  ## Human-readable summary.
proc numParameters*(m: Module, countFrozen: bool = true): int
  ## Total parameter count.
```

### Container Modules

```nim
type
  Sequential* = ref object of Module
    layers*: seq[Module]

  ModuleList* = ref object of Module
    modules*: seq[Module]

  ModuleDict* = ref object of Module
    modules*: Table[string, Module]

proc newSequential*(layers: varargs[Module]): Sequential
proc add*(s: Sequential, layer: Module): void
method forward*(s: Sequential, inputs: varargs[TensorRef]): TensorRef

proc newModuleList*(modules: varargs[Module]): ModuleList
proc `[]`*(ml: ModuleList, i: int): Module
proc len*(ml: ModuleList): int

proc newModuleDict*(): ModuleDict
proc `[]`*(md: ModuleDict, key: string): Module
proc `[]=`*(md: ModuleDict, key: string, m: Module): void
```

---

## 2. Layers

### Dense Layer (`layers/dense.nim`)

```nim
type
  Linear* = ref object of Module
    inFeatures*: int
    outFeatures*: int
    useBias*: bool
    weight*: Parameter
    bias*: Option[Parameter]

proc newLinear*(inFeatures, outFeatures: int,
                useBias: bool = true,
                dtype: DType = dtFloat32): Linear
method forward*(l: Linear, input: TensorRef): TensorRef
  ## output = input @ weight^T + bias
```

### Convolution (`layers/conv.nim`)

```nim
type
  Conv2d* = ref object of Module
    inChannels*, outChannels*: int
    kernelSize*: (int, int)
    stride*: (int, int)
    padding*: (int, int)
    dilation*: (int, int)
    groups*: int
    useBias*: bool
    weight*: Parameter
    bias*: Option[Parameter]

proc newConv2d*(inChannels, outChannels: int,
                kernelSize: int or (int, int),
                stride: int or (int, int) = 1,
                padding: int or (int, int) = 0,
                dilation: int or (int, int) = 1,
                groups: int = 1,
                useBias: bool = true): Conv2d
method forward*(c: Conv2d, input: TensorRef): TensorRef
```

### Normalization (`layers/norm.nim`)

```nim
type
  BatchNorm2d* = ref object of Module
    numFeatures*: int
    eps*: float
    momentum*: float
    affine*: bool
    trackRunningStats*: bool
    weight*, bias*: Option[Parameter]
    runningMean*, runningVar*: TensorRef

  LayerNorm* = ref object of Module
    normalizedShape*: seq[int]
    eps*: float
    weight*, bias*: Option[Parameter]

proc newBatchNorm2d*(numFeatures: int, eps: float = 1e-5,
                     momentum: float = 0.1,
                     affine: bool = true): BatchNorm2d

proc newLayerNorm*(normalizedShape: seq[int],
                   eps: float = 1e-5): LayerNorm
```

### Activation (`layers/activation.nim`)

```nim
type
  ReLU* = ref object of Module
    inplace*: bool

  GELU* = ref object of Module

  Sigmoid* = ref object of Module

  Tanh* = ref object of Module

  Softmax* = ref object of Module
    dim*: int

proc newReLU*(inplace: bool = false): ReLU
proc newGELU*(): GELU
proc newSigmoid*(): Sigmoid
proc newTanh*(): Tanh
proc newSoftmax*(dim: int = -1): Softmax
```

### Dropout (`layers/dropout.nim`)

```nim
type
  Dropout* = ref object of Module
    rate*: float

proc newDropout*(rate: float = 0.5): Dropout
method forward*(d: Dropout, input: TensorRef): TensorRef
  ## During training: randomly zero elements.
  ## During eval: identity.
```

### Embedding (`layers/embedding.nim`)

```nim
type
  Embedding* = ref object of Module
    numEmbeddings*: int
    embeddingDim*: int
    paddingIdx*: Option[int]
    weight*: Parameter

proc newEmbedding*(numEmbeddings, embeddingDim: int,
                   paddingIdx: Option[int] = none(int)): Embedding
method forward*(e: Embedding, indices: TensorRef): TensorRef
```

### LoRA Adapter (`layers/lora.nim`)

```nim
type
  LoRAConfig* = object
    rank*: int                # LoRA rank (e.g., 8, 16, 32)
    alpha*: float             # Scaling factor
    dropoutRate*: float
    targetModules*: seq[string]  # e.g., ["q_proj", "v_proj"]

  LoRALinear* = ref object of Module
    baseLinear*: Linear       # Frozen base layer
    loraA*: Parameter         # Low-rank A matrix
    loraB*: Parameter         # Low-rank B matrix
    scaling*: float           # alpha / rank
    dropout*: Dropout
    merged*: bool

proc newLoRALinear*(baseLinear: Linear, config: LoRAConfig): LoRALinear
  ## Wrap existing Linear with LoRA adapter.

method forward*(l: LoRALinear, input: TensorRef): TensorRef
  ## output = base(input) + (dropout(input) @ A @ B) * scaling

proc merge*(l: LoRALinear): void
  ## Merge LoRA weights into base (for inference).

proc unmerge*(l: LoRALinear): void
  ## Unmerge for continued training.

# Apply LoRA to model
proc applyLoRA*(model: Module, config: LoRAConfig): void
  ## Replace target layers with LoRA versions.

proc getLoRAParameters*(model: Module): seq[Parameter]
  ## Get only LoRA parameters (for optimizer).
```

---

## 3. Loss Functions (`loss.nim`)

### Types

```nim
type
  Loss* = ref object of RootObj
    reduction*: Reduction

  Reduction* = enum
    rNone     # No reduction
    rMean     # Mean of all elements
    rSum      # Sum of all elements

  CrossEntropyLoss* = ref object of Loss
    ignoreIndex*: Option[int]
    labelSmoothing*: float

  MSELoss* = ref object of Loss

  L1Loss* = ref object of Loss

  BCELoss* = ref object of Loss

  BCEWithLogitsLoss* = ref object of Loss
```

### API

```nim
proc newCrossEntropyLoss*(reduction: Reduction = rMean,
                          ignoreIndex: Option[int] = none(int),
                          labelSmoothing: float = 0.0): CrossEntropyLoss
proc forward*(l: CrossEntropyLoss, input, target: TensorRef): TensorRef
  ## input: (N, C) logits, target: (N,) class indices

proc newMSELoss*(reduction: Reduction = rMean): MSELoss
proc forward*(l: MSELoss, input, target: TensorRef): TensorRef

proc newL1Loss*(reduction: Reduction = rMean): L1Loss

proc newBCELoss*(reduction: Reduction = rMean): BCELoss

proc newBCEWithLogitsLoss*(reduction: Reduction = rMean): BCEWithLogitsLoss
```

---

## 4. Optimizers (`optim.nim`)

### Types

```nim
type
  Optimizer* = ref object of RootObj
    parameters*: seq[Parameter]
    defaults*: Table[string, float]
    state*: Table[Hash256, OptimizerState]

  OptimizerState* = object
    step*: int
    tensors*: Table[string, TensorData]  # e.g., "momentum", "variance"

  SGD* = ref object of Optimizer
    lr*: float
    momentum*: float
    dampening*: float
    weightDecay*: float
    nesterov*: bool

  Adam* = ref object of Optimizer
    lr*: float
    betas*: (float, float)
    eps*: float
    weightDecay*: float
    amsgrad*: bool

  AdamW* = ref object of Optimizer
    lr*: float
    betas*: (float, float)
    eps*: float
    weightDecay*: float

  LRScheduler* = ref object of RootObj
    optimizer*: Optimizer
    lastEpoch*: int

  StepLR* = ref object of LRScheduler
    stepSize*: int
    gamma*: float

  CosineAnnealingLR* = ref object of LRScheduler
    tMax*: int
    etaMin*: float
```

### API

```nim
# Optimizer base
method step*(o: Optimizer, executor: Executor): void {.base.}
  ## Perform optimization step.

method zeroGrad*(o: Optimizer): void {.base.}
  ## Zero all parameter gradients.

proc addParamGroup*(o: Optimizer, params: seq[Parameter],
                    options: Table[string, float] = initTable()): void
  ## Add parameter group with custom options.

proc stateDict*(o: Optimizer): JsonNode
proc loadStateDict*(o: Optimizer, state: JsonNode): void

# SGD
proc newSGD*(params: seq[Parameter], lr: float,
             momentum: float = 0.0,
             dampening: float = 0.0,
             weightDecay: float = 0.0,
             nesterov: bool = false): SGD
method step*(o: SGD, executor: Executor): void
  ## v = momentum * v + grad
  ## param = param - lr * (v + weightDecay * param)

# Adam
proc newAdam*(params: seq[Parameter], lr: float = 0.001,
              betas: (float, float) = (0.9, 0.999),
              eps: float = 1e-8,
              weightDecay: float = 0.0,
              amsgrad: bool = false): Adam
method step*(o: Adam, executor: Executor): void

# AdamW (decoupled weight decay)
proc newAdamW*(params: seq[Parameter], lr: float = 0.001,
               betas: (float, float) = (0.9, 0.999),
               eps: float = 1e-8,
               weightDecay: float = 0.01): AdamW
method step*(o: AdamW, executor: Executor): void

# LR Schedulers
proc newStepLR*(optimizer: Optimizer, stepSize: int,
                gamma: float = 0.1): StepLR
proc step*(s: StepLR): void
  ## Decay LR by gamma every stepSize epochs.

proc newCosineAnnealingLR*(optimizer: Optimizer, tMax: int,
                           etaMin: float = 0.0): CosineAnnealingLR
proc step*(s: CosineAnnealingLR): void

proc getLR*(s: LRScheduler): float
  ## Get current learning rate.
```

---

## Dependencies

```nim
# nimml_nn.nimble
requires "nim >= 2.0.0"
requires "nimml_core >= 0.1.0"
requires "nimml_autograd >= 0.1.0"
```

---

## Usage Example

```nim
import nimml_nn
import nimml_autograd
import nimml_executor

# Define a simple MLP
type
  MLP = ref object of Module
    fc1: Linear
    fc2: Linear
    relu: ReLU

proc newMLP(inputDim, hiddenDim, outputDim: int): MLP =
  result = MLP()
  result.fc1 = newLinear(inputDim, hiddenDim)
  result.fc2 = newLinear(hiddenDim, outputDim)
  result.relu = newReLU()
  result.registerModule("fc1", result.fc1)
  result.registerModule("fc2", result.fc2)

method forward(m: MLP, input: TensorRef): TensorRef =
  var x = m.fc1.forward(input)
  x = m.relu.forward(x)
  result = m.fc2.forward(x)

# Create model and optimizer
let model = newMLP(784, 256, 10)
let optimizer = newAdam(model.parameters(), lr = 0.001)
let criterion = newCrossEntropyLoss()

# Training step
model.train()
let tape = newTape()

tape.withTape:
  let output = model.forward(input)
  let loss = criterion.forward(output, target)

let grads = backward(loss, tape, executor)
optimizer.step(executor)
optimizer.zeroGrad()

# Apply LoRA for fine-tuning
let loraConfig = LoRAConfig(rank: 8, alpha: 16.0, targetModules: @["fc1", "fc2"])
model.freeze()  # Freeze base model
applyLoRA(model, loraConfig)

# Now only LoRA parameters are trainable
let loraOptimizer = newAdamW(getLoRAParameters(model), lr = 1e-4)
```
