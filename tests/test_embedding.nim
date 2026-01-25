## Test embedding layers
## nim c -r tests/test_embedding.nim

import std/[unittest, math, options]
import ml_core
import ml_nn/module
import ml_nn/layers/embedding

suite "Embedding Layer":
  test "Create Embedding":
    let emb = newEmbedding(1000, 128)
    check emb.numEmbeddings == 1000
    check emb.embeddingDim == 128
    check emb.weight.data.shape.dims == @[1000, 128]

  test "Embedding with padding index":
    let emb = newEmbedding(100, 32, paddingIdx = some(0))
    check emb.paddingIdx.isSome
    check emb.paddingIdx.get == 0

  test "Embedding lookup":
    let emb = newEmbedding(10, 4)

    # Set known values
    let arr = emb.weight.data.asFloat32
    for i in 0 ..< 10:
      for j in 0 ..< 4:
        arr[i * 4 + j] = (i * 10 + j).float32

    let res = emb.lookup(@[2, 5])
    let resArr = res.asFloat32
    check res.shape.dims == @[2, 4]
    check resArr[0] == 20.0  # Index 2, dim 0
    check resArr[4] == 50.0  # Index 5, dim 0

  test "Embedding similarity":
    let emb = newEmbedding(3, 4)
    let arr = emb.weight.data.asFloat32

    # Set embedding 0 = [1, 0, 0, 0]
    arr[0] = 1.0
    arr[1] = 0.0
    arr[2] = 0.0
    arr[3] = 0.0

    # Set embedding 1 = [1, 0, 0, 0] (same)
    arr[4] = 1.0
    arr[5] = 0.0
    arr[6] = 0.0
    arr[7] = 0.0

    # Set embedding 2 = [0, 1, 0, 0] (orthogonal)
    arr[8] = 0.0
    arr[9] = 1.0
    arr[10] = 0.0
    arr[11] = 0.0

    let sim01 = embeddingSimilarity(emb, 0, 1)
    let sim02 = embeddingSimilarity(emb, 0, 2)

    check abs(sim01 - 1.0) < 0.01  # Same vectors
    check abs(sim02 - 0.0) < 0.01  # Orthogonal

suite "EmbeddingBag":
  test "Create EmbeddingBag":
    let bag = newEmbeddingBag(500, 64, mode = "mean")
    check bag.numEmbeddings == 500
    check bag.embeddingDim == 64
    check bag.mode == "mean"

suite "Positional Encoding":
  test "Create PositionalEncoding":
    let pe = newPositionalEncoding(100, 64)
    check pe.maxLen == 100
    check pe.embeddingDim == 64
    check pe.encoding.shape.dims == @[100, 64]

  test "Sinusoidal pattern":
    let pe = newPositionalEncoding(10, 4)
    let arr = pe.encoding.asFloat32

    # Position 0 should have sin(0)=0, cos(0)=1 pattern
    check abs(arr[0] - 0.0) < 0.01  # sin(0)
    check abs(arr[1] - 1.0) < 0.01  # cos(0)

  test "Get encoding subset":
    let pe = newPositionalEncoding(100, 64)
    let subset = pe.getEncoding(10)
    check subset.shape.dims == @[10, 64]

suite "Learned Positional Embedding":
  test "Create LearnedPositionalEmbedding":
    let lpe = newLearnedPositionalEmbedding(512, 128)
    check lpe.maxLen == 512
    check lpe.embeddingDim == 128
    check lpe.weight.data.shape.dims == @[512, 128]

  test "Lookup positions":
    let lpe = newLearnedPositionalEmbedding(10, 4)

    # Set known values
    let arr = lpe.weight.data.asFloat32
    for i in 0 ..< 10:
      for j in 0 ..< 4:
        arr[i * 4 + j] = (i * 10 + j).float32

    let res = lpe.lookup(@[0, 5, 9])
    let resArr = res.asFloat32
    check res.shape.dims == @[3, 4]
    check resArr[0] == 0.0   # Position 0, dim 0
    check resArr[4] == 50.0  # Position 5, dim 0
    check resArr[8] == 90.0  # Position 9, dim 0

suite "Rotary Embedding (RoPE)":
  test "Create RotaryEmbedding":
    let rope = newRotaryEmbedding(64, maxSeqLen = 2048)
    check rope.dim == 64
    check rope.maxSeqLen == 2048
    check rope.base == 10000.0
    check rope.cosCache.shape.dims == @[2048, 32]  # halfDim = 32
    check rope.sinCache.shape.dims == @[2048, 32]

  test "RoPE dimension must be even":
    expect ValueError:
      discard newRotaryEmbedding(63)  # Odd dimension

  test "RoPE cache values":
    let rope = newRotaryEmbedding(4, maxSeqLen = 10)
    let cosArr = rope.cosCache.asFloat32
    let sinArr = rope.sinCache.asFloat32

    # Position 0 should have cos(0)=1, sin(0)=0 for first frequency
    check abs(cosArr[0] - 1.0) < 0.01
    check abs(sinArr[0] - 0.0) < 0.01

  test "Apply rotary embedding":
    let rope = newRotaryEmbedding(4, maxSeqLen = 10)

    # Create input tensor [2, 4]
    var input = newTensorData(newShape(2, 4), dtFloat32)
    let inputArr = input.asFloat32
    for i in 0 ..< 8:
      inputArr[i] = 1.0

    let output = rope.applyRotary(input, startPos = 0)
    check output.shape.dims == @[2, 4]

echo "All embedding tests passed!"
