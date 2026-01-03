# nim-ml-nn

Neural network layers, loss functions, and optimizers for Nim.

## Overview

nim-ml-nn provides PyTorch-style neural network building blocks:

- **Module System** - Base class with parameter management
- **Dense Layers** - Linear, LazyLinear, Bilinear, Identity, Flatten
- **Activations** - ReLU, GELU, Sigmoid, Tanh, Softmax, and more
- **Dropout** - Dropout, Dropout2d, AlphaDropout
- **Loss Functions** - MSE, CrossEntropy, BCE, Huber, Triplet, CTC
- **Optimizers** - SGD, Adam, AdamW, Adagrad, RMSprop, Adadelta

## Installation

```bash
nimble install ml_nn
```

## Modules

| Module | Description | Tests |
|--------|-------------|-------|
| module | Base Module class, Sequential, ModuleList | 44 |
| activation | 18 activation functions | 53 |
| dense | Linear, Bilinear, Flatten layers | 40 |
| dropout | Dropout variants | 31 |
| loss | 16 loss functions | 47 |
| optim | 6 optimizers, 10 LR schedulers | 51 |

**Total: 266 tests passing**

## Usage

```nim
import ml_nn

# Create a simple network
let model = newSequential(@[
  newLinear(784, 256),
  newReLU(),
  newDropout(0.5),
  newLinear(256, 10),
  newSoftmax(dim = -1)
])

# Forward pass
let output = model.forward(input)

# Create optimizer
let optimizer = newAdam(model.parameters(), lr = 0.001)

# Training step
optimizer.zeroGrad()
let loss = crossEntropy(output, target)
# backward pass...
optimizer.step()
```

## Activation Functions

ReLU, LeakyReLU, PReLU, ELU, SELU, GELU, Sigmoid, Tanh, Hardtanh, Hardsigmoid, Softmax, LogSoftmax, Softplus, Softsign, Hardswish, SiLU, Mish, GLU

## Loss Functions

MSELoss, L1Loss, SmoothL1Loss, HuberLoss, CrossEntropyLoss, NLLLoss, BCELoss, BCEWithLogitsLoss, KLDivLoss, MarginRankingLoss, HingeEmbeddingLoss, CosineEmbeddingLoss, TripletMarginLoss, MultiMarginLoss, CTCLoss, PoissonNLLLoss

## License

Apache-2.0
