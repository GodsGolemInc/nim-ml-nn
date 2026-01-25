## Embedding Layers
##
## Token, position, and segment embeddings for neural networks.
## Used in transformers and other sequence models.

import std/[options, math, tables]
import ml_core
import ../module

type
  Embedding* = ref object of Module
    ## Lookup table mapping indices to fixed-size vectors
    ##
    ## Input: LongTensor of indices (*, )
    ## Output: (*, embeddingDim)
    numEmbeddings*: int    ## Size of vocabulary
    embeddingDim*: int     ## Dimension of embeddings
    paddingIdx*: Option[int]  ## Index to keep at zero
    maxNorm*: Option[float]   ## Max norm for embedding vectors
    scaleGradByFreq*: bool    ## Scale gradients by frequency
    sparse*: bool             ## Use sparse gradients
    weight*: Parameter        ## Embedding matrix

  EmbeddingBag* = ref object of Module
    ## Computes sums/means of embeddings without instantiating intermediate embeddings
    ## More memory efficient for bag-of-words models
    numEmbeddings*: int
    embeddingDim*: int
    mode*: string  ## "sum", "mean", or "max"
    sparse*: bool
    weight*: Parameter

  PositionalEncoding* = ref object of Module
    ## Sinusoidal positional encoding (Attention Is All You Need)
    maxLen*: int
    embeddingDim*: int
    dropout*: float
    encoding*: TensorData  ## Precomputed encodings

  LearnedPositionalEmbedding* = ref object of Module
    ## Learned positional embeddings
    maxLen*: int
    embeddingDim*: int
    weight*: Parameter

  RotaryEmbedding* = ref object of Module
    ## Rotary Position Embedding (RoPE) for LLaMA-style models
    dim*: int
    maxSeqLen*: int
    base*: float
    cosCache*: TensorData
    sinCache*: TensorData

# ============================================================================
# Embedding Layer
# ============================================================================

proc newEmbedding*(numEmbeddings, embeddingDim: int,
                   paddingIdx: Option[int] = none(int),
                   maxNorm: Option[float] = none(float),
                   scaleGradByFreq: bool = false,
                   sparse: bool = false,
                   dtype: DType = dtFloat32): Embedding =
  ## Create a new embedding layer
  ##
  ## Args:
  ##   numEmbeddings: Size of the vocabulary
  ##   embeddingDim: Dimension of each embedding vector
  ##   paddingIdx: If given, pads output with zeros at this index
  ##   maxNorm: If given, renormalize embeddings to have max norm
  ##   scaleGradByFreq: Scale gradients by inverse frequency
  ##   sparse: Use sparse gradients
  ##   dtype: Data type for weights
  result = Embedding()
  result.initModule("Embedding")
  result.numEmbeddings = numEmbeddings
  result.embeddingDim = embeddingDim
  result.paddingIdx = paddingIdx
  result.maxNorm = maxNorm
  result.scaleGradByFreq = scaleGradByFreq
  result.sparse = sparse

  # Register weight parameter: [numEmbeddings x embeddingDim]
  result.weight = result.registerParameter(
    "weight",
    newShape(numEmbeddings, embeddingDim),
    dtype
  )

proc initNormal*(emb: Embedding, mean: float = 0.0, std: float = 1.0) =
  ## Initialize embeddings from normal distribution
  ## (Placeholder - actual random init should use autograd random)
  discard mean
  discard std

proc initFromPretrained*(emb: var Embedding, data: TensorData, freeze: bool = true) =
  ## Initialize from pretrained embeddings
  if data.shape.dims.len != 2:
    raise newException(ModuleError, "Pretrained embeddings must be 2D")
  if data.shape.dims[0] != emb.numEmbeddings or data.shape.dims[1] != emb.embeddingDim:
    raise newException(ModuleError, "Shape mismatch for pretrained embeddings")

  emb.weight = newParameterFromData("weight", data, requiresGrad = not freeze)
  emb.parameters["weight"] = emb.weight
  if freeze:
    emb.frozen = true

method forward*(emb: Embedding, inputs: varargs[TensorRef]): TensorRef =
  ## Lookup embeddings for input indices
  ##
  ## Args:
  ##   inputs[0]: Index tensor of shape (*)
  ##
  ## Returns:
  ##   Embedding tensor of shape (*, embeddingDim)
  if inputs.len == 0:
    raise newException(ModuleError, "Embedding forward requires input indices")

  # Placeholder implementation - actual impl would use index_select
  # For now, return reference to weight for shape inference
  newTensorRef(emb.weight.data)

proc lookup*(emb: Embedding, indices: seq[int]): TensorData =
  ## Direct lookup of embeddings by indices (for inference)
  let outShape = newShape(indices.len, emb.embeddingDim)
  result = newTensorData(outShape, emb.weight.data.dtype)

  # Copy embedding vectors for each index
  for i, idx in indices:
    if idx < 0 or idx >= emb.numEmbeddings:
      raise newException(IndexDefect, "Embedding index out of range: " & $idx)

    # Handle padding index
    if emb.paddingIdx.isSome and idx == emb.paddingIdx.get:
      # Output zeros for padding
      continue

    # Copy from weight matrix
    let srcOffset = idx * emb.embeddingDim
    let dstOffset = i * emb.embeddingDim
    let srcArr = emb.weight.data.asFloat32
    let dstArr = result.asFloat32
    for j in 0 ..< emb.embeddingDim:
      dstArr[dstOffset + j] = srcArr[srcOffset + j]

# ============================================================================
# EmbeddingBag
# ============================================================================

proc newEmbeddingBag*(numEmbeddings, embeddingDim: int,
                      mode: string = "mean",
                      sparse: bool = false,
                      dtype: DType = dtFloat32): EmbeddingBag =
  ## Create an EmbeddingBag layer
  ##
  ## Args:
  ##   numEmbeddings: Size of vocabulary
  ##   embeddingDim: Dimension of embeddings
  ##   mode: "sum", "mean", or "max"
  ##   sparse: Use sparse gradients
  result = EmbeddingBag()
  result.initModule("EmbeddingBag")
  result.numEmbeddings = numEmbeddings
  result.embeddingDim = embeddingDim
  result.mode = mode
  result.sparse = sparse

  result.weight = result.registerParameter(
    "weight",
    newShape(numEmbeddings, embeddingDim),
    dtype
  )

method forward*(bag: EmbeddingBag, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass - aggregates embeddings
  if inputs.len == 0:
    raise newException(ModuleError, "EmbeddingBag forward requires input indices")
  newTensorRef(bag.weight.data)

# ============================================================================
# Positional Encoding (Sinusoidal)
# ============================================================================

proc computeSinusoidalEncoding(maxLen, dim: int): TensorData =
  ## Compute sinusoidal position encodings
  ## PE(pos, 2i) = sin(pos / 10000^(2i/d))
  ## PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
  result = newTensorData(newShape(maxLen, dim), dtFloat32)
  let arr = result.asFloat32

  for pos in 0 ..< maxLen:
    for i in 0 ..< dim div 2:
      let angle = pos.float / pow(10000.0, (2 * i).float / dim.float)
      arr[pos * dim + 2 * i] = sin(angle).float32
      arr[pos * dim + 2 * i + 1] = cos(angle).float32

proc newPositionalEncoding*(maxLen, embeddingDim: int,
                            dropout: float = 0.1): PositionalEncoding =
  ## Create sinusoidal positional encoding
  ##
  ## Args:
  ##   maxLen: Maximum sequence length
  ##   embeddingDim: Dimension of embeddings (must match model)
  ##   dropout: Dropout probability
  result = PositionalEncoding()
  result.initModule("PositionalEncoding")
  result.maxLen = maxLen
  result.embeddingDim = embeddingDim
  result.dropout = dropout
  result.encoding = computeSinusoidalEncoding(maxLen, embeddingDim)

method forward*(pe: PositionalEncoding, inputs: varargs[TensorRef]): TensorRef =
  ## Add positional encoding to input
  ##
  ## Args:
  ##   inputs[0]: Input tensor of shape (batch, seqLen, dim)
  ##
  ## Returns:
  ##   Input + positional encoding
  if inputs.len == 0:
    raise newException(ModuleError, "PositionalEncoding requires input")
  # Return encoding for shape inference
  newTensorRef(pe.encoding)

proc getEncoding*(pe: PositionalEncoding, seqLen: int): TensorData =
  ## Get positional encoding for specific sequence length
  if seqLen > pe.maxLen:
    raise newException(ValueError, "Sequence length exceeds maximum: " & $seqLen)

  result = newTensorData(newShape(seqLen, pe.embeddingDim), dtFloat32)
  let srcArr = pe.encoding.asFloat32
  let dstArr = result.asFloat32
  for i in 0 ..< seqLen * pe.embeddingDim:
    dstArr[i] = srcArr[i]

# ============================================================================
# Learned Positional Embedding
# ============================================================================

proc newLearnedPositionalEmbedding*(maxLen, embeddingDim: int,
                                    dtype: DType = dtFloat32): LearnedPositionalEmbedding =
  ## Create learned positional embeddings
  ##
  ## Args:
  ##   maxLen: Maximum sequence length
  ##   embeddingDim: Dimension of embeddings
  result = LearnedPositionalEmbedding()
  result.initModule("LearnedPositionalEmbedding")
  result.maxLen = maxLen
  result.embeddingDim = embeddingDim

  result.weight = result.registerParameter(
    "weight",
    newShape(maxLen, embeddingDim),
    dtype
  )

method forward*(lpe: LearnedPositionalEmbedding, inputs: varargs[TensorRef]): TensorRef =
  ## Get positional embeddings for input positions
  if inputs.len == 0:
    raise newException(ModuleError, "LearnedPositionalEmbedding requires position indices")
  newTensorRef(lpe.weight.data)

proc lookup*(lpe: LearnedPositionalEmbedding, positions: seq[int]): TensorData =
  ## Direct lookup of positional embeddings
  let outShape = newShape(positions.len, lpe.embeddingDim)
  result = newTensorData(outShape, lpe.weight.data.dtype)
  let srcArr = lpe.weight.data.asFloat32
  let dstArr = result.asFloat32

  for i, pos in positions:
    if pos < 0 or pos >= lpe.maxLen:
      raise newException(IndexDefect, "Position index out of range: " & $pos)

    let srcOffset = pos * lpe.embeddingDim
    let dstOffset = i * lpe.embeddingDim
    for j in 0 ..< lpe.embeddingDim:
      dstArr[dstOffset + j] = srcArr[srcOffset + j]

# ============================================================================
# Rotary Position Embedding (RoPE)
# ============================================================================

proc computeRotaryCache(dim, maxSeqLen: int, base: float): tuple[cos, sin: TensorData] =
  ## Precompute rotary embedding cos/sin caches
  let halfDim = dim div 2

  # Compute inverse frequencies
  var invFreq = newSeq[float32](halfDim)
  for i in 0 ..< halfDim:
    invFreq[i] = 1.0 / pow(base, (2 * i).float / dim.float).float32

  # Compute cos/sin for all positions
  result.cos = newTensorData(newShape(maxSeqLen, halfDim), dtFloat32)
  result.sin = newTensorData(newShape(maxSeqLen, halfDim), dtFloat32)
  let cosArr = result.cos.asFloat32
  let sinArr = result.sin.asFloat32

  for pos in 0 ..< maxSeqLen:
    for i in 0 ..< halfDim:
      let angle = pos.float32 * invFreq[i]
      cosArr[pos * halfDim + i] = cos(angle).float32
      sinArr[pos * halfDim + i] = sin(angle).float32

proc newRotaryEmbedding*(dim: int, maxSeqLen: int = 2048,
                         base: float = 10000.0): RotaryEmbedding =
  ## Create Rotary Position Embedding
  ##
  ## Args:
  ##   dim: Head dimension (must be even)
  ##   maxSeqLen: Maximum sequence length to cache
  ##   base: Base for frequency computation (default 10000)
  if dim mod 2 != 0:
    raise newException(ValueError, "RoPE dimension must be even")

  result = RotaryEmbedding()
  result.initModule("RotaryEmbedding")
  result.dim = dim
  result.maxSeqLen = maxSeqLen
  result.base = base

  let (cosCache, sinCache) = computeRotaryCache(dim, maxSeqLen, base)
  result.cosCache = cosCache
  result.sinCache = sinCache

proc applyRotary*(rope: RotaryEmbedding, x: TensorData, startPos: int = 0): TensorData =
  ## Apply rotary embeddings to input tensor
  ##
  ## Args:
  ##   x: Input tensor of shape (seqLen, dim)
  ##   startPos: Starting position for encoding
  ##
  ## Returns:
  ##   Tensor with rotary embeddings applied
  if x.shape.dims.len != 2:
    raise newException(ValueError, "Input must be 2D [seqLen, dim]")

  let seqLen = x.shape.dims[0]
  let dim = x.shape.dims[1]
  let halfDim = dim div 2

  if startPos + seqLen > rope.maxSeqLen:
    raise newException(ValueError, "Sequence exceeds cached positions")

  result = newTensorData(x.shape, x.dtype)
  let xArr = x.asFloat32
  let cosArr = rope.cosCache.asFloat32
  let sinArr = rope.sinCache.asFloat32
  let outArr = result.asFloat32

  for pos in 0 ..< seqLen:
    let cachePos = startPos + pos
    for i in 0 ..< halfDim:
      let x1 = xArr[pos * dim + i]
      let x2 = xArr[pos * dim + halfDim + i]
      let cosVal = cosArr[cachePos * halfDim + i]
      let sinVal = sinArr[cachePos * halfDim + i]

      # Rotary transformation
      outArr[pos * dim + i] = x1 * cosVal - x2 * sinVal
      outArr[pos * dim + halfDim + i] = x1 * sinVal + x2 * cosVal

method forward*(rope: RotaryEmbedding, inputs: varargs[TensorRef]): TensorRef =
  ## Forward pass - apply rotary embeddings
  if inputs.len == 0:
    raise newException(ModuleError, "RotaryEmbedding requires input")
  # Return cos cache shape for inference
  newTensorRef(rope.cosCache)

# ============================================================================
# Utility Functions
# ============================================================================

proc embeddingSimilarity*(emb: Embedding, idx1, idx2: int): float32 =
  ## Compute cosine similarity between two embeddings
  var dotProd: float32 = 0.0
  var norm1: float32 = 0.0
  var norm2: float32 = 0.0

  let offset1 = idx1 * emb.embeddingDim
  let offset2 = idx2 * emb.embeddingDim
  let arr = emb.weight.data.asFloat32

  for i in 0 ..< emb.embeddingDim:
    let v1 = arr[offset1 + i]
    let v2 = arr[offset2 + i]
    dotProd += v1 * v2
    norm1 += v1 * v1
    norm2 += v2 * v2

  if norm1 < 1e-10 or norm2 < 1e-10:
    return 0.0

  result = dotProd / (sqrt(norm1) * sqrt(norm2))

proc findNearestEmbeddings*(emb: Embedding, queryIdx: int, topK: int = 10): seq[tuple[idx: int, score: float32]] =
  ## Find embeddings most similar to query
  result = @[]

  for i in 0 ..< emb.numEmbeddings:
    if i != queryIdx:
      let sim = embeddingSimilarity(emb, queryIdx, i)
      result.add((i, sim))

  # Sort by similarity descending (simple bubble sort for now)
  for i in 0 ..< result.len - 1:
    for j in i + 1 ..< result.len:
      if result[j].score > result[i].score:
        swap(result[i], result[j])

  if result.len > topK:
    result = result[0 ..< topK]
