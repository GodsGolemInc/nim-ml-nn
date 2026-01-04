## Attention Layers
##
## Attention mechanisms for neural networks.
## Includes MultiHeadAttention, Transformer, and related components.

import std/[options, strformat, math]
import ml_core
import ../module
import dense
import norm
import dropout

type
  # =============================================================================
  # Multi-Head Attention
  # =============================================================================

  MultiHeadAttention* = ref object of Module
    ## Multi-head attention mechanism
    ##
    ## Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    ##
    ## Used in Transformer architectures.
    embedDim*: int
    numHeads*: int
    headDim*: int
    kdim*: int
    vdim*: int
    useBias*: bool
    addBiasKv*: bool
    addZeroAttn*: bool
    dropout*: float
    batchFirst*: bool

    qProj*: Linear   # Query projection
    kProj*: Linear   # Key projection
    vProj*: Linear   # Value projection
    outProj*: Linear # Output projection

    biasK*: Option[Parameter]
    biasV*: Option[Parameter]

  # =============================================================================
  # Transformer Components
  # =============================================================================

  TransformerEncoderLayer* = ref object of Module
    ## Single encoder layer of a Transformer
    ##
    ## Consists of:
    ##   1. Self-attention
    ##   2. Add & Norm
    ##   3. Feed-forward
    ##   4. Add & Norm
    dModel*: int
    nhead*: int
    dimFeedforward*: int
    dropout*: float
    activationFn*: string
    layerNormEps*: float
    batchFirst*: bool
    normFirst*: bool  # Pre-LN vs Post-LN

    selfAttn*: MultiHeadAttention
    linear1*: Linear
    linear2*: Linear
    norm1*: LayerNorm
    norm2*: LayerNorm
    dropout1*: Dropout
    dropout2*: Dropout

  TransformerDecoderLayer* = ref object of Module
    ## Single decoder layer of a Transformer
    ##
    ## Consists of:
    ##   1. Self-attention
    ##   2. Add & Norm
    ##   3. Cross-attention
    ##   4. Add & Norm
    ##   5. Feed-forward
    ##   6. Add & Norm
    dModel*: int
    nhead*: int
    dimFeedforward*: int
    dropout*: float
    activationFn*: string
    layerNormEps*: float
    batchFirst*: bool
    normFirst*: bool

    selfAttn*: MultiHeadAttention
    multiheadAttn*: MultiHeadAttention
    linear1*: Linear
    linear2*: Linear
    norm1*: LayerNorm
    norm2*: LayerNorm
    norm3*: LayerNorm
    dropout1*: Dropout
    dropout2*: Dropout
    dropout3*: Dropout

  TransformerEncoder* = ref object of Module
    ## Stack of N encoder layers
    layers*: seq[TransformerEncoderLayer]
    numLayers*: int
    norm*: Option[LayerNorm]
    enableNestedTensor*: bool
    maskCheck*: bool

  TransformerDecoder* = ref object of Module
    ## Stack of N decoder layers
    layers*: seq[TransformerDecoderLayer]
    numLayers*: int
    norm*: Option[LayerNorm]

  Transformer* = ref object of Module
    ## Full Transformer model
    ##
    ## Consists of encoder and decoder stacks
    dModel*: int
    nhead*: int
    numEncoderLayers*: int
    numDecoderLayers*: int
    dimFeedforward*: int
    dropout*: float
    activationFn*: string
    customEncoder*: Option[TransformerEncoder]
    customDecoder*: Option[TransformerDecoder]
    batchFirst*: bool
    normFirst*: bool

    encoder*: TransformerEncoder
    decoder*: TransformerDecoder

# =============================================================================
# MultiHeadAttention Implementation
# =============================================================================

proc newMultiHeadAttention*(embedDim: int,
                             numHeads: int,
                             dropout: float = 0.0,
                             useBias: bool = true,
                             addBiasKv: bool = false,
                             addZeroAttn: bool = false,
                             kdim: int = 0,
                             vdim: int = 0,
                             batchFirst: bool = false,
                             dtype: DType = dtFloat32): MultiHeadAttention =
  ## Create a MultiHeadAttention layer
  ##
  ## Args:
  ##   embedDim: Total dimension of the model
  ##   numHeads: Number of attention heads
  ##   dropout: Dropout probability on attention weights
  ##   useBias: Whether to use bias in projection layers
  ##   addBiasKv: Add bias to key and value projections
  ##   addZeroAttn: Add a new batch of zeros to key and value
  ##   kdim: Dimension of key (default: embedDim)
  ##   vdim: Dimension of value (default: embedDim)
  ##   batchFirst: If True, input is (batch, seq, feature)
  assert embedDim mod numHeads == 0, "embedDim must be divisible by numHeads"

  result = MultiHeadAttention()
  result.initModule("MultiHeadAttention")
  result.embedDim = embedDim
  result.numHeads = numHeads
  result.headDim = embedDim div numHeads
  result.kdim = if kdim == 0: embedDim else: kdim
  result.vdim = if vdim == 0: embedDim else: vdim
  result.useBias = useBias
  result.addBiasKv = addBiasKv
  result.addZeroAttn = addZeroAttn
  result.dropout = dropout
  result.batchFirst = batchFirst

  # Create projection layers
  result.qProj = newLinear(embedDim, embedDim, useBias, dtype)
  result.registerModule("q_proj", result.qProj)

  result.kProj = newLinear(result.kdim, embedDim, useBias, dtype)
  result.registerModule("k_proj", result.kProj)

  result.vProj = newLinear(result.vdim, embedDim, useBias, dtype)
  result.registerModule("v_proj", result.vProj)

  result.outProj = newLinear(embedDim, embedDim, useBias, dtype)
  result.registerModule("out_proj", result.outProj)

  if addBiasKv:
    let biasKParam = result.registerParameter("bias_k", newShape(1, 1, embedDim), dtype)
    result.biasK = some(biasKParam)
    let biasVParam = result.registerParameter("bias_v", newShape(1, 1, embedDim), dtype)
    result.biasV = some(biasVParam)
  else:
    result.biasK = none(Parameter)
    result.biasV = none(Parameter)

method forward*(mha: MultiHeadAttention, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass for MultiHeadAttention
  ##
  ## Args (through inputs):
  ##   query: Query tensor (L, N, E) or (N, L, E) if batchFirst
  ##   key: Key tensor (S, N, E) or (N, S, E) if batchFirst
  ##   value: Value tensor (S, N, E) or (N, S, E) if batchFirst
  ##
  ## Returns:
  ##   Attention output (L, N, E) or (N, L, E) if batchFirst
  if inputs.len < 1:
    raise newException(ModuleError, "MultiHeadAttention forward requires query input")

  let query = inputs[0]
  # If only query provided, assume self-attention
  # key = inputs[1] if provided, else query
  # value = inputs[2] if provided, else key

  let dims = query.shape.dims

  if dims.len != 3:
    raise newException(ModuleError,
      fmt"MultiHeadAttention expects 3D input, got {dims.len}D")

  # Output shape same as query
  newTensorRef(query.shape, query.dtype)

# =============================================================================
# TransformerEncoderLayer Implementation
# =============================================================================

proc newTransformerEncoderLayer*(dModel: int,
                                  nhead: int,
                                  dimFeedforward: int = 2048,
                                  dropout: float = 0.1,
                                  activationFn: string = "relu",
                                  layerNormEps: float = 1e-5,
                                  batchFirst: bool = false,
                                  normFirst: bool = false,
                                  dtype: DType = dtFloat32): TransformerEncoderLayer =
  ## Create a TransformerEncoderLayer
  ##
  ## Args:
  ##   dModel: Model dimension
  ##   nhead: Number of attention heads
  ##   dimFeedforward: Feed-forward network dimension
  ##   dropout: Dropout probability
  ##   activationFn: Activation function ("relu" or "gelu")
  ##   layerNormEps: LayerNorm epsilon
  ##   batchFirst: If True, input is (batch, seq, feature)
  ##   normFirst: If True, use pre-LN (more stable training)
  result = TransformerEncoderLayer()
  result.initModule("TransformerEncoderLayer")
  result.dModel = dModel
  result.nhead = nhead
  result.dimFeedforward = dimFeedforward
  result.dropout = dropout
  result.activationFn = activationFn
  result.layerNormEps = layerNormEps
  result.batchFirst = batchFirst
  result.normFirst = normFirst

  # Self-attention
  result.selfAttn = newMultiHeadAttention(dModel, nhead, dropout, batchFirst = batchFirst, dtype = dtype)
  result.registerModule("self_attn", result.selfAttn)

  # Feed-forward network
  result.linear1 = newLinear(dModel, dimFeedforward, dtype = dtype)
  result.registerModule("linear1", result.linear1)
  result.linear2 = newLinear(dimFeedforward, dModel, dtype = dtype)
  result.registerModule("linear2", result.linear2)

  # Layer norms
  result.norm1 = newLayerNorm(dModel, layerNormEps, dtype = dtype)
  result.registerModule("norm1", result.norm1)
  result.norm2 = newLayerNorm(dModel, layerNormEps, dtype = dtype)
  result.registerModule("norm2", result.norm2)

  # Dropouts
  result.dropout1 = newDropout(dropout)
  result.registerModule("dropout1", result.dropout1)
  result.dropout2 = newDropout(dropout)
  result.registerModule("dropout2", result.dropout2)

method forward*(layer: TransformerEncoderLayer, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass for TransformerEncoderLayer
  ##
  ## Implements: output = LayerNorm(x + Dropout(Attention(x)))
  ##             output = LayerNorm(output + Dropout(FF(output)))
  if inputs.len == 0:
    raise newException(ModuleError, "TransformerEncoderLayer forward requires input")

  let src = inputs[0]
  # Output has same shape as input
  newTensorRef(src.shape, src.dtype)

# =============================================================================
# TransformerDecoderLayer Implementation
# =============================================================================

proc newTransformerDecoderLayer*(dModel: int,
                                  nhead: int,
                                  dimFeedforward: int = 2048,
                                  dropout: float = 0.1,
                                  activationFn: string = "relu",
                                  layerNormEps: float = 1e-5,
                                  batchFirst: bool = false,
                                  normFirst: bool = false,
                                  dtype: DType = dtFloat32): TransformerDecoderLayer =
  ## Create a TransformerDecoderLayer
  result = TransformerDecoderLayer()
  result.initModule("TransformerDecoderLayer")
  result.dModel = dModel
  result.nhead = nhead
  result.dimFeedforward = dimFeedforward
  result.dropout = dropout
  result.activationFn = activationFn
  result.layerNormEps = layerNormEps
  result.batchFirst = batchFirst
  result.normFirst = normFirst

  # Self-attention
  result.selfAttn = newMultiHeadAttention(dModel, nhead, dropout, batchFirst = batchFirst, dtype = dtype)
  result.registerModule("self_attn", result.selfAttn)

  # Cross-attention
  result.multiheadAttn = newMultiHeadAttention(dModel, nhead, dropout, batchFirst = batchFirst, dtype = dtype)
  result.registerModule("multihead_attn", result.multiheadAttn)

  # Feed-forward network
  result.linear1 = newLinear(dModel, dimFeedforward, dtype = dtype)
  result.registerModule("linear1", result.linear1)
  result.linear2 = newLinear(dimFeedforward, dModel, dtype = dtype)
  result.registerModule("linear2", result.linear2)

  # Layer norms
  result.norm1 = newLayerNorm(dModel, layerNormEps, dtype = dtype)
  result.registerModule("norm1", result.norm1)
  result.norm2 = newLayerNorm(dModel, layerNormEps, dtype = dtype)
  result.registerModule("norm2", result.norm2)
  result.norm3 = newLayerNorm(dModel, layerNormEps, dtype = dtype)
  result.registerModule("norm3", result.norm3)

  # Dropouts
  result.dropout1 = newDropout(dropout)
  result.registerModule("dropout1", result.dropout1)
  result.dropout2 = newDropout(dropout)
  result.registerModule("dropout2", result.dropout2)
  result.dropout3 = newDropout(dropout)
  result.registerModule("dropout3", result.dropout3)

method forward*(layer: TransformerDecoderLayer, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass for TransformerDecoderLayer
  ##
  ## Args:
  ##   inputs[0]: Target sequence
  ##   inputs[1]: Memory (encoder output)
  if inputs.len < 2:
    raise newException(ModuleError, "TransformerDecoderLayer forward requires target and memory")

  let tgt = inputs[0]
  newTensorRef(tgt.shape, tgt.dtype)

# =============================================================================
# TransformerEncoder Implementation
# =============================================================================

proc newTransformerEncoder*(encoderLayer: TransformerEncoderLayer,
                             numLayers: int,
                             norm: Option[LayerNorm] = none(LayerNorm),
                             enableNestedTensor: bool = true,
                             maskCheck: bool = true): TransformerEncoder =
  ## Create a TransformerEncoder stack
  result = TransformerEncoder()
  result.initModule("TransformerEncoder")
  result.numLayers = numLayers
  result.norm = norm
  result.enableNestedTensor = enableNestedTensor
  result.maskCheck = maskCheck
  result.layers = @[]

  for i in 0..<numLayers:
    # In practice, would clone the layer
    result.layers.add(encoderLayer)
    result.registerModule($i, encoderLayer)

  if norm.isSome:
    result.registerModule("norm", norm.get)

method forward*(encoder: TransformerEncoder, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass through encoder stack
  if inputs.len == 0:
    raise newException(ModuleError, "TransformerEncoder forward requires input")

  let src = inputs[0]
  newTensorRef(src.shape, src.dtype)

# =============================================================================
# TransformerDecoder Implementation
# =============================================================================

proc newTransformerDecoder*(decoderLayer: TransformerDecoderLayer,
                             numLayers: int,
                             norm: Option[LayerNorm] = none(LayerNorm)): TransformerDecoder =
  ## Create a TransformerDecoder stack
  result = TransformerDecoder()
  result.initModule("TransformerDecoder")
  result.numLayers = numLayers
  result.norm = norm
  result.layers = @[]

  for i in 0..<numLayers:
    result.layers.add(decoderLayer)
    result.registerModule($i, decoderLayer)

  if norm.isSome:
    result.registerModule("norm", norm.get)

method forward*(decoder: TransformerDecoder, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass through decoder stack
  if inputs.len < 2:
    raise newException(ModuleError, "TransformerDecoder forward requires target and memory")

  let tgt = inputs[0]
  newTensorRef(tgt.shape, tgt.dtype)

# =============================================================================
# Full Transformer Implementation
# =============================================================================

proc newTransformer*(dModel: int = 512,
                      nhead: int = 8,
                      numEncoderLayers: int = 6,
                      numDecoderLayers: int = 6,
                      dimFeedforward: int = 2048,
                      dropout: float = 0.1,
                      activationFn: string = "relu",
                      customEncoder: Option[TransformerEncoder] = none(TransformerEncoder),
                      customDecoder: Option[TransformerDecoder] = none(TransformerDecoder),
                      layerNormEps: float = 1e-5,
                      batchFirst: bool = false,
                      normFirst: bool = false,
                      dtype: DType = dtFloat32): Transformer =
  ## Create a full Transformer model
  ##
  ## Args:
  ##   dModel: Model dimension (default 512)
  ##   nhead: Number of attention heads (default 8)
  ##   numEncoderLayers: Number of encoder layers (default 6)
  ##   numDecoderLayers: Number of decoder layers (default 6)
  ##   dimFeedforward: Feed-forward dimension (default 2048)
  ##   dropout: Dropout probability (default 0.1)
  ##   activationFn: Activation function ("relu" or "gelu")
  ##   batchFirst: If True, input is (batch, seq, feature)
  ##   normFirst: If True, use pre-LN (more stable)
  result = Transformer()
  result.initModule("Transformer")
  result.dModel = dModel
  result.nhead = nhead
  result.numEncoderLayers = numEncoderLayers
  result.numDecoderLayers = numDecoderLayers
  result.dimFeedforward = dimFeedforward
  result.dropout = dropout
  result.activationFn = activationFn
  result.batchFirst = batchFirst
  result.normFirst = normFirst

  if customEncoder.isSome:
    result.encoder = customEncoder.get
  else:
    let encoderLayer = newTransformerEncoderLayer(
      dModel, nhead, dimFeedforward, dropout, activationFn,
      layerNormEps, batchFirst, normFirst, dtype
    )
    let encoderNorm = newLayerNorm(dModel, layerNormEps, dtype = dtype)
    result.encoder = newTransformerEncoder(encoderLayer, numEncoderLayers, some(encoderNorm))

  result.registerModule("encoder", result.encoder)

  if customDecoder.isSome:
    result.decoder = customDecoder.get
  else:
    let decoderLayer = newTransformerDecoderLayer(
      dModel, nhead, dimFeedforward, dropout, activationFn,
      layerNormEps, batchFirst, normFirst, dtype
    )
    let decoderNorm = newLayerNorm(dModel, layerNormEps, dtype = dtype)
    result.decoder = newTransformerDecoder(decoderLayer, numDecoderLayers, some(decoderNorm))

  result.registerModule("decoder", result.decoder)

method forward*(t: Transformer, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass for Transformer
  ##
  ## Args:
  ##   inputs[0]: Source sequence
  ##   inputs[1]: Target sequence
  ##
  ## Returns:
  ##   Decoder output
  if inputs.len < 2:
    raise newException(ModuleError, "Transformer forward requires source and target")

  let tgt = inputs[1]
  newTensorRef(tgt.shape, tgt.dtype)

# =============================================================================
# Positional Encoding
# =============================================================================

type
  PositionalEncoding* = ref object of Module
    ## Sinusoidal positional encoding
    ##
    ## PE(pos, 2i) = sin(pos / 10000^(2i/dModel))
    ## PE(pos, 2i+1) = cos(pos / 10000^(2i/dModel))
    dModel*: int
    dropout*: float
    maxLen*: int
    dropoutLayer*: Dropout
    pe*: TensorData  # Precomputed positional encodings

proc newPositionalEncoding*(dModel: int,
                             dropout: float = 0.1,
                             maxLen: int = 5000,
                             dtype: DType = dtFloat32): PositionalEncoding =
  ## Create a positional encoding layer
  result = PositionalEncoding()
  result.initModule("PositionalEncoding")
  result.dModel = dModel
  result.dropout = dropout
  result.maxLen = maxLen

  result.dropoutLayer = newDropout(dropout)
  result.registerModule("dropout", result.dropoutLayer)

  # Precompute positional encodings
  result.pe = newTensorData(newShape(maxLen, dModel), dtype)
  # Actual computation would fill in sin/cos values

method forward*(pe: PositionalEncoding, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass adds positional encoding to input
  if inputs.len == 0:
    raise newException(ModuleError, "PositionalEncoding forward requires input")

  let x = inputs[0]
  # Output shape same as input
  newTensorRef(x.shape, x.dtype)

# =============================================================================
# Learned Positional Embedding
# =============================================================================

type
  LearnedPositionalEmbedding* = ref object of Module
    ## Learned positional embeddings
    ## Used in BERT, GPT, etc.
    numPositions*: int
    embeddingDim*: int
    weight*: Parameter

proc newLearnedPositionalEmbedding*(numPositions: int,
                                     embeddingDim: int,
                                     dtype: DType = dtFloat32): LearnedPositionalEmbedding =
  ## Create a learned positional embedding layer
  result = LearnedPositionalEmbedding()
  result.initModule("LearnedPositionalEmbedding")
  result.numPositions = numPositions
  result.embeddingDim = embeddingDim

  result.weight = result.registerParameter("weight",
    newShape(numPositions, embeddingDim), dtype)

method forward*(lpe: LearnedPositionalEmbedding, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass returns positional embeddings for sequence length
  if inputs.len == 0:
    raise newException(ModuleError, "LearnedPositionalEmbedding forward requires input")

  let x = inputs[0]
  newTensorRef(x.shape, x.dtype)

# =============================================================================
# Rotary Positional Embedding (RoPE)
# =============================================================================

type
  RotaryEmbedding* = ref object of Module
    ## Rotary Positional Embedding (RoPE)
    ## Used in LLaMA, GPT-NeoX, etc.
    dim*: int
    maxSeqLen*: int
    base*: float
    invFreq*: TensorData  # Precomputed inverse frequencies

proc newRotaryEmbedding*(dim: int,
                          maxSeqLen: int = 2048,
                          base: float = 10000.0,
                          dtype: DType = dtFloat32): RotaryEmbedding =
  ## Create a Rotary Embedding layer
  ##
  ## Args:
  ##   dim: Dimension of the embedding
  ##   maxSeqLen: Maximum sequence length
  ##   base: Base for the geometric progression
  result = RotaryEmbedding()
  result.initModule("RotaryEmbedding")
  result.dim = dim
  result.maxSeqLen = maxSeqLen
  result.base = base

  # Precompute inverse frequencies
  result.invFreq = newTensorData(newShape(dim div 2), dtype)

method forward*(rope: RotaryEmbedding, inputs: varargs[TensorRef]): TensorRef =
  ## Apply rotary embedding to input
  if inputs.len == 0:
    raise newException(ModuleError, "RotaryEmbedding forward requires input")

  let x = inputs[0]
  newTensorRef(x.shape, x.dtype)

# =============================================================================
# Scaled Dot-Product Attention (functional)
# =============================================================================

proc scaledDotProductAttention*(query, key, value: TensorRef,
                                 attnMask: Option[TensorRef] = none(TensorRef),
                                 dropoutP: float = 0.0,
                                 isCausal: bool = false,
                                 scale: Option[float] = none(float)): TensorRef =
  ## Scaled dot-product attention function
  ##
  ## Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
  ##
  ## Args:
  ##   query: Query tensor (*, L, E)
  ##   key: Key tensor (*, S, E)
  ##   value: Value tensor (*, S, Ev)
  ##   attnMask: Optional attention mask
  ##   dropoutP: Dropout probability
  ##   isCausal: Whether to apply causal mask
  ##   scale: Optional scale factor (default: 1/sqrt(E))
  let dims = query.shape.dims
  if dims.len < 2:
    raise newException(ModuleError, "query must have at least 2 dimensions")

  # Output has shape (*, L, Ev)
  var outDims = dims
  outDims[^1] = value.shape.dims[^1]

  newTensorRef(newShape(outDims), query.dtype)

# =============================================================================
# Flash Attention (placeholder for future optimization)
# =============================================================================

type
  FlashAttention* = ref object of Module
    ## Flash Attention - memory-efficient attention
    ## Uses tiling to reduce memory usage from O(n^2) to O(n)
    headDim*: int
    dropout*: float
    softmax_scale*: float
    causal*: bool

proc newFlashAttention*(headDim: int,
                         dropout: float = 0.0,
                         softmaxScale: float = 0.0,
                         causal: bool = false): FlashAttention =
  ## Create a Flash Attention layer
  result = FlashAttention()
  result.initModule("FlashAttention")
  result.headDim = headDim
  result.dropout = dropout
  result.softmax_scale = if softmaxScale == 0.0: 1.0 / sqrt(headDim.float) else: softmaxScale
  result.causal = causal

method forward*(flash: FlashAttention, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass for Flash Attention
  if inputs.len < 3:
    raise newException(ModuleError, "FlashAttention forward requires q, k, v")

  let q = inputs[0]
  newTensorRef(q.shape, q.dtype)
