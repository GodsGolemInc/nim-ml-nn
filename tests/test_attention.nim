## Attention Layer Tests
##
## Tests for MultiHeadAttention, Transformer, PositionalEncoding, etc.

import std/options
import unittest
import ml_core
import ml_nn/module
import ml_nn/layers/attention
import ml_nn/layers/norm

# =============================================================================
# MultiHeadAttention Tests
# =============================================================================

suite "MultiHeadAttention":
  test "create MultiHeadAttention":
    let mha = newMultiHeadAttention(512, 8)
    check mha.embedDim == 512
    check mha.numHeads == 8
    check mha.headDim == 64  # 512 / 8
    check mha.kdim == 512
    check mha.vdim == 512
    check mha.useBias == true
    check mha.dropout == 0.0
    check mha.batchFirst == false

  test "MultiHeadAttention custom kdim/vdim":
    let mha = newMultiHeadAttention(512, 8, kdim = 256, vdim = 256)
    check mha.kdim == 256
    check mha.vdim == 256

  test "MultiHeadAttention with dropout":
    let mha = newMultiHeadAttention(512, 8, dropout = 0.1)
    check mha.dropout == 0.1

  test "MultiHeadAttention batch first":
    let mha = newMultiHeadAttention(512, 8, batchFirst = true)
    check mha.batchFirst == true

  test "MultiHeadAttention submodules":
    let mha = newMultiHeadAttention(512, 8)
    check mha.hasModule("q_proj")
    check mha.hasModule("k_proj")
    check mha.hasModule("v_proj")
    check mha.hasModule("out_proj")

  test "MultiHeadAttention forward":
    let mha = newMultiHeadAttention(512, 8)
    let query = newTensorRef(newShape(10, 32, 512), dtFloat32)  # (seq, batch, embed)
    let output = mha.forward(query)
    check output.shape.dims == @[10, 32, 512]

  test "MultiHeadAttention forward batch first":
    let mha = newMultiHeadAttention(512, 8, batchFirst = true)
    let query = newTensorRef(newShape(32, 10, 512), dtFloat32)  # (batch, seq, embed)
    let output = mha.forward(query)
    check output.shape.dims == @[32, 10, 512]

  test "MultiHeadAttention with bias_kv":
    let mha = newMultiHeadAttention(512, 8, addBiasKv = true)
    check mha.addBiasKv == true
    check mha.biasK.isSome
    check mha.biasV.isSome

  test "MultiHeadAttention wrong dims":
    let mha = newMultiHeadAttention(512, 8)
    let query = newTensorRef(newShape(32, 512), dtFloat32)  # 2D instead of 3D
    expect(ModuleError):
      discard mha.forward(query)

  test "MultiHeadAttention embedDim divisibility":
    # embedDim must be divisible by numHeads
    expect(AssertionDefect):
      discard newMultiHeadAttention(512, 7)

# =============================================================================
# TransformerEncoderLayer Tests
# =============================================================================

suite "TransformerEncoderLayer":
  test "create TransformerEncoderLayer":
    let layer = newTransformerEncoderLayer(512, 8)
    check layer.dModel == 512
    check layer.nhead == 8
    check layer.dimFeedforward == 2048
    check layer.dropout == 0.1
    check layer.activationFn == "relu"
    check layer.normFirst == false

  test "TransformerEncoderLayer custom params":
    let layer = newTransformerEncoderLayer(
      512, 8,
      dimFeedforward = 1024,
      dropout = 0.2,
      activationFn = "gelu",
      normFirst = true
    )
    check layer.dimFeedforward == 1024
    check layer.dropout == 0.2
    check layer.activationFn == "gelu"
    check layer.normFirst == true

  test "TransformerEncoderLayer submodules":
    let layer = newTransformerEncoderLayer(512, 8)
    check layer.hasModule("self_attn")
    check layer.hasModule("linear1")
    check layer.hasModule("linear2")
    check layer.hasModule("norm1")
    check layer.hasModule("norm2")
    check layer.hasModule("dropout1")
    check layer.hasModule("dropout2")

  test "TransformerEncoderLayer forward":
    let layer = newTransformerEncoderLayer(512, 8)
    let src = newTensorRef(newShape(10, 32, 512), dtFloat32)
    let output = layer.forward(src)
    check output.shape.dims == @[10, 32, 512]

  test "TransformerEncoderLayer forward batch first":
    let layer = newTransformerEncoderLayer(512, 8, batchFirst = true)
    let src = newTensorRef(newShape(32, 10, 512), dtFloat32)
    let output = layer.forward(src)
    check output.shape.dims == @[32, 10, 512]

  test "TransformerEncoderLayer no input":
    let layer = newTransformerEncoderLayer(512, 8)
    expect(ModuleError):
      discard layer.forward()

# =============================================================================
# TransformerDecoderLayer Tests
# =============================================================================

suite "TransformerDecoderLayer":
  test "create TransformerDecoderLayer":
    let layer = newTransformerDecoderLayer(512, 8)
    check layer.dModel == 512
    check layer.nhead == 8
    check layer.dimFeedforward == 2048

  test "TransformerDecoderLayer submodules":
    let layer = newTransformerDecoderLayer(512, 8)
    check layer.hasModule("self_attn")
    check layer.hasModule("multihead_attn")  # Cross-attention
    check layer.hasModule("linear1")
    check layer.hasModule("linear2")
    check layer.hasModule("norm1")
    check layer.hasModule("norm2")
    check layer.hasModule("norm3")

  test "TransformerDecoderLayer forward":
    let layer = newTransformerDecoderLayer(512, 8)
    let tgt = newTensorRef(newShape(10, 32, 512), dtFloat32)
    let memory = newTensorRef(newShape(20, 32, 512), dtFloat32)
    let output = layer.forward(tgt, memory)
    check output.shape.dims == @[10, 32, 512]

  test "TransformerDecoderLayer missing memory":
    let layer = newTransformerDecoderLayer(512, 8)
    let tgt = newTensorRef(newShape(10, 32, 512), dtFloat32)
    expect(ModuleError):
      discard layer.forward(tgt)

# =============================================================================
# TransformerEncoder Tests
# =============================================================================

suite "TransformerEncoder":
  test "create TransformerEncoder":
    let layer = newTransformerEncoderLayer(512, 8)
    let encoder = newTransformerEncoder(layer, 6)
    check encoder.numLayers == 6
    check encoder.layers.len == 6

  test "TransformerEncoder with norm":
    let layer = newTransformerEncoderLayer(512, 8)
    let norm = newLayerNorm(512)
    let encoder = newTransformerEncoder(layer, 6, norm = some(norm))
    check encoder.norm.isSome

  test "TransformerEncoder forward":
    let layer = newTransformerEncoderLayer(512, 8)
    let encoder = newTransformerEncoder(layer, 6)
    let src = newTensorRef(newShape(10, 32, 512), dtFloat32)
    let output = encoder.forward(src)
    check output.shape.dims == @[10, 32, 512]

  test "TransformerEncoder no input":
    let layer = newTransformerEncoderLayer(512, 8)
    let encoder = newTransformerEncoder(layer, 6)
    expect(ModuleError):
      discard encoder.forward()

# =============================================================================
# TransformerDecoder Tests
# =============================================================================

suite "TransformerDecoder":
  test "create TransformerDecoder":
    let layer = newTransformerDecoderLayer(512, 8)
    let decoder = newTransformerDecoder(layer, 6)
    check decoder.numLayers == 6
    check decoder.layers.len == 6

  test "TransformerDecoder with norm":
    let layer = newTransformerDecoderLayer(512, 8)
    let norm = newLayerNorm(512)
    let decoder = newTransformerDecoder(layer, 6, norm = some(norm))
    check decoder.norm.isSome

  test "TransformerDecoder forward":
    let layer = newTransformerDecoderLayer(512, 8)
    let decoder = newTransformerDecoder(layer, 6)
    let tgt = newTensorRef(newShape(10, 32, 512), dtFloat32)
    let memory = newTensorRef(newShape(20, 32, 512), dtFloat32)
    let output = decoder.forward(tgt, memory)
    check output.shape.dims == @[10, 32, 512]

  test "TransformerDecoder missing inputs":
    let layer = newTransformerDecoderLayer(512, 8)
    let decoder = newTransformerDecoder(layer, 6)
    let tgt = newTensorRef(newShape(10, 32, 512), dtFloat32)
    expect(ModuleError):
      discard decoder.forward(tgt)

# =============================================================================
# Full Transformer Tests
# =============================================================================

suite "Transformer":
  test "create Transformer default":
    let transformer = newTransformer()
    check transformer.dModel == 512
    check transformer.nhead == 8
    check transformer.numEncoderLayers == 6
    check transformer.numDecoderLayers == 6
    check transformer.dimFeedforward == 2048
    check transformer.dropout == 0.1
    check transformer.activationFn == "relu"

  test "create Transformer custom":
    let transformer = newTransformer(
      dModel = 256,
      nhead = 4,
      numEncoderLayers = 3,
      numDecoderLayers = 3,
      dimFeedforward = 1024,
      dropout = 0.2,
      activationFn = "gelu"
    )
    check transformer.dModel == 256
    check transformer.nhead == 4
    check transformer.numEncoderLayers == 3

  test "Transformer submodules":
    let transformer = newTransformer()
    check transformer.hasModule("encoder")
    check transformer.hasModule("decoder")

  test "Transformer forward":
    let transformer = newTransformer()
    let src = newTensorRef(newShape(10, 32, 512), dtFloat32)
    let tgt = newTensorRef(newShape(20, 32, 512), dtFloat32)
    let output = transformer.forward(src, tgt)
    check output.shape.dims == @[20, 32, 512]

  test "Transformer missing inputs":
    let transformer = newTransformer()
    let src = newTensorRef(newShape(10, 32, 512), dtFloat32)
    expect(ModuleError):
      discard transformer.forward(src)

  test "Transformer batch first":
    let transformer = newTransformer(batchFirst = true)
    check transformer.batchFirst == true

  test "Transformer norm first (pre-LN)":
    let transformer = newTransformer(normFirst = true)
    check transformer.normFirst == true

# =============================================================================
# PositionalEncoding Tests
# =============================================================================

suite "PositionalEncoding":
  test "create PositionalEncoding":
    let pe = newPositionalEncoding(512)
    check pe.dModel == 512
    check pe.dropout == 0.1
    check pe.maxLen == 5000

  test "PositionalEncoding custom params":
    let pe = newPositionalEncoding(256, dropout = 0.2, maxLen = 1000)
    check pe.dModel == 256
    check pe.dropout == 0.2
    check pe.maxLen == 1000

  test "PositionalEncoding forward":
    let pe = newPositionalEncoding(512)
    let input = newTensorRef(newShape(100, 32, 512), dtFloat32)
    let output = pe.forward(input)
    check output.shape.dims == @[100, 32, 512]

  test "PositionalEncoding no input":
    let pe = newPositionalEncoding(512)
    expect(ModuleError):
      discard pe.forward()

# =============================================================================
# LearnedPositionalEmbedding Tests
# =============================================================================

suite "LearnedPositionalEmbedding":
  test "create LearnedPositionalEmbedding":
    let lpe = newLearnedPositionalEmbedding(512, 768)
    check lpe.numPositions == 512
    check lpe.embeddingDim == 768

  test "LearnedPositionalEmbedding parameters":
    let lpe = newLearnedPositionalEmbedding(512, 768)
    check lpe.parameters().len == 1

  test "LearnedPositionalEmbedding forward":
    let lpe = newLearnedPositionalEmbedding(512, 768)
    let input = newTensorRef(newShape(32, 100, 768), dtFloat32)
    let output = lpe.forward(input)
    check output.shape.dims == @[32, 100, 768]

  test "LearnedPositionalEmbedding no input":
    let lpe = newLearnedPositionalEmbedding(512, 768)
    expect(ModuleError):
      discard lpe.forward()

# =============================================================================
# RotaryEmbedding Tests
# =============================================================================

suite "RotaryEmbedding":
  test "create RotaryEmbedding":
    let rope = newRotaryEmbedding(64)
    check rope.dim == 64
    check rope.maxSeqLen == 2048
    check rope.base == 10000.0

  test "RotaryEmbedding custom params":
    let rope = newRotaryEmbedding(128, maxSeqLen = 4096, base = 10000.0)
    check rope.dim == 128
    check rope.maxSeqLen == 4096

  test "RotaryEmbedding forward":
    let rope = newRotaryEmbedding(64)
    let input = newTensorRef(newShape(32, 100, 64), dtFloat32)
    let output = rope.forward(input)
    check output.shape.dims == @[32, 100, 64]

  test "RotaryEmbedding no input":
    let rope = newRotaryEmbedding(64)
    expect(ModuleError):
      discard rope.forward()

# =============================================================================
# FlashAttention Tests
# =============================================================================

suite "FlashAttention":
  test "create FlashAttention":
    let flash = newFlashAttention(64)
    check flash.headDim == 64
    check flash.dropout == 0.0
    check flash.causal == false

  test "FlashAttention causal":
    let flash = newFlashAttention(64, causal = true)
    check flash.causal == true

  test "FlashAttention with dropout":
    let flash = newFlashAttention(64, dropout = 0.1)
    check flash.dropout == 0.1

  test "FlashAttention forward":
    let flash = newFlashAttention(64)
    let q = newTensorRef(newShape(32, 8, 100, 64), dtFloat32)
    let k = newTensorRef(newShape(32, 8, 100, 64), dtFloat32)
    let v = newTensorRef(newShape(32, 8, 100, 64), dtFloat32)
    let output = flash.forward(q, k, v)
    check output.shape.dims == @[32, 8, 100, 64]

  test "FlashAttention missing inputs":
    let flash = newFlashAttention(64)
    let q = newTensorRef(newShape(32, 8, 100, 64), dtFloat32)
    let k = newTensorRef(newShape(32, 8, 100, 64), dtFloat32)
    expect(ModuleError):
      discard flash.forward(q, k)

# =============================================================================
# ScaledDotProductAttention Tests
# =============================================================================

suite "ScaledDotProductAttention":
  test "scaledDotProductAttention basic":
    let q = newTensorRef(newShape(32, 8, 100, 64), dtFloat32)
    let k = newTensorRef(newShape(32, 8, 100, 64), dtFloat32)
    let v = newTensorRef(newShape(32, 8, 100, 64), dtFloat32)
    let output = scaledDotProductAttention(q, k, v)
    check output.shape.dims == @[32, 8, 100, 64]

  test "scaledDotProductAttention causal":
    let q = newTensorRef(newShape(32, 8, 100, 64), dtFloat32)
    let k = newTensorRef(newShape(32, 8, 100, 64), dtFloat32)
    let v = newTensorRef(newShape(32, 8, 100, 64), dtFloat32)
    let output = scaledDotProductAttention(q, k, v, isCausal = true)
    check output.shape.dims == @[32, 8, 100, 64]

  test "scaledDotProductAttention with dropout":
    let q = newTensorRef(newShape(32, 8, 100, 64), dtFloat32)
    let k = newTensorRef(newShape(32, 8, 100, 64), dtFloat32)
    let v = newTensorRef(newShape(32, 8, 100, 64), dtFloat32)
    let output = scaledDotProductAttention(q, k, v, dropoutP = 0.1)
    check output.shape.dims == @[32, 8, 100, 64]

  test "scaledDotProductAttention wrong dims":
    let q = newTensorRef(newShape(64), dtFloat32)  # 1D instead of 2D+
    let k = newTensorRef(newShape(64), dtFloat32)
    let v = newTensorRef(newShape(64), dtFloat32)
    expect(ModuleError):
      discard scaledDotProductAttention(q, k, v)

# =============================================================================
# Training Mode Tests
# =============================================================================

suite "Attention Training Mode":
  test "MultiHeadAttention training mode":
    let mha = newMultiHeadAttention(512, 8)
    check mha.training == true

    mha.eval()
    check mha.training == false

    mha.train()
    check mha.training == true

  test "Transformer training mode":
    let transformer = newTransformer()
    check transformer.training == true

    transformer.eval()
    check transformer.training == false

    transformer.train()
    check transformer.training == true

# =============================================================================
# Parameter Count Tests
# =============================================================================

suite "Attention Parameter Counts":
  test "MultiHeadAttention parameters":
    let mha = newMultiHeadAttention(512, 8)
    let params = mha.parameters()
    # q_proj, k_proj, v_proj, out_proj - each with weight + bias
    check params.len >= 8

  test "TransformerEncoderLayer parameters":
    let layer = newTransformerEncoderLayer(512, 8)
    let params = layer.parameters()
    # Should have parameters from attention + FFN + layer norms
    check params.len > 10

  test "Transformer total parameters":
    let transformer = newTransformer(
      dModel = 256,
      nhead = 4,
      numEncoderLayers = 2,
      numDecoderLayers = 2
    )
    let numParams = transformer.numParameters()
    # Just ensure we have a reasonable number
    check numParams > 0
