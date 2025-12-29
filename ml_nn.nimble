# Package

version       = "0.0.1"
author        = "jasagiri"
description   = "Neural network layers, loss functions, and optimizers"
license       = "Apache-2.0"
srcDir        = "src"

# Dependencies

requires "nim >= 2.0.0"
requires "ml_core >= 0.0.4"

# Tasks
task test, "Run tests":
  exec "nim c -r --path:../nim-ml-core/src tests/test_module.nim"
  exec "nim c -r --path:../nim-ml-core/src tests/test_activation.nim"
  exec "nim c -r --path:../nim-ml-core/src tests/test_dense.nim"
  exec "nim c -r --path:../nim-ml-core/src tests/test_dropout.nim"
  exec "nim c -r --path:../nim-ml-core/src tests/test_loss.nim"
  exec "nim c -r --path:../nim-ml-core/src tests/test_optim.nim"
