# Package

version       = "0.0.1"
author        = "jasagiri"
description   = "Neural network layers, loss functions, and optimizers"
license       = "MIT"
srcDir        = "src"

# Dependencies

requires "nim >= 2.0.0"
requires "nimml_core >= 0.0.4"

# Tasks
task test, "Run tests":
  exec "nim c -r --path:../nim-ml-core/src tests/test_module.nim"
