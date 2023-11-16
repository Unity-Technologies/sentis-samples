# Changelog
All notable changes to this package will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).


## [1.3.0-pre.1] - 2023-11-09

### Added
- Mad (multiply add) method added to Ops
- Multinomial method added to Ops
- TextureConverter ToTensor methods can accept allocated tensors as parameters
- AddInput method with TensorShape parameter added to Model
- UnknownOfRank method added to SymbolicTensorShape
- Save method with Stream added to ModelWriter
- SingleHeadAttention fused layer
- Importer can handle external weight files for large models
- Encrypted model sample project

### Changed
- Many Ops methods now accept generic Tensor types and infer output types
- Added kernel shape to Conv layer
- ONNXModelConverter takes model file path when instantiating
- Concat parameter ordering in Ops method

### Fixed
- GatherElements inference fixed when not gathering on innermost axis
- Optimizer pass for Einsum no longer breaks model
- Reduce operator no longer incorrectly optimizes to Identity in edge case
- ArgMax and ArgMin output data types correctly inferred as int
- Pad import with empty constant pad fixed
- Optimizer pass for removing duplicate layers and constants
- Internal dlls no longer visible from outside package and no longer cause conflicts

### Removed
- Dependencies on experimental code


## [1.2.0-exp.2] - 2023-09-20

### Added
- Fast inference path for 1D Resize
- OneHot operator now supports float tensors properly
- Import for LayerNormalization operator
- Import for Gelu operator
- Optimized ScalarMad layer with optimizer passes

### Removed
- Watermark on Unity scenes referencing Sentis
- License key requirement for Sentis

### Changed
- Moved CPU fallback pass to mandatory optimizer passes and remove MakeReadable calls in layer inference
- Binary search for allocator speedup
- Cached compute func instances for inference speedup
- Refactored AxisNormalization to be LayerNormalization

### Fixed
- Import settings for Upsample operator
- Condition for fast inference path for 2D Resize on CPU
- Inspector no longer crashes when model can't be imported
- Serialize to streaming assets now correctly uses optimal path and doesn't crash the editor with large models
- Serialize to streaming assets creates directory if it doesn't exist and refreshes asset database


## [1.1.1-exp.2] - 2023-09-04

### Fixed
- Fixed inference with multiple Conv layers with different attributes in same model
- Fixed inference for AveragePool and MaxPool with large channel counts on GPUPixel
- Fixed batched MatMul inference on GPUCompute and GPUCommandBuffer
- Fixed Conv inference with no bias on GPUCompute and GPUCommandBuffer
- Fixed inspector view when model can't be loaded
- Made Mode.Load(FileStream) added


## [1.1.1-exp.1] - 2023-08-31

### Fixed
- Fixed faulty Transpose optimization
- Fixed documentation titles
- Fixed problem in BatchNormalization optimization pass
- Fixed ReduceMean inference error
- Fixed Reduce issues on Unity 2021.3.30f on iOS devices


## [1.1.0-exp.2] - 2023-08-17

### Changed
- Documentation was updated to 1.1.0


## [1.1.0-exp.1] - 2023-08-14

### Added
- Added AsyncReadbackRequest and IsAsyncReadbackRequestDone for asynchronous readback from GPU
- Added Shrink, Abs (int), Neg (int), Not, Sign, PRelu, Hardmax, OneHot to GPUPixel backend
- Added And, Equal, Greater, GreaterOrEqual, IsInf, IsNaN, Less, LessOrEqual, Or, Xor, Where to GPUPixel backend
- Added Add (int), Sub (int), Div (int), Pow (int), FMod (int), Mod (int), FMod (int), Mul(int), Sum (int), Min (int), Max (int) to GPUPixel backend
- Added Expand (int), Concat (int), Slice (int), Transpose (int), Transpose (no permutations), Split (int), Tile, Resize (3D), Reshape (int) to GPUPixel backend
- Added ReduceMax (int), ReduceMin (int), ReduceProd (int), ReduceSum (int), ReduceL1 (int), ReduceSumSquare (int), ArgMax, ArgMin to GPUPixel backend
- Added Gather, GatherElements, GatherND, ScatterElements, ScatterND to GPUPixel backend
- Added InstanceNormalization, AxisNormalization, RoiAlign, RandomUniform, RandomNormal, Bernoulli, Range, Trilu, CumSum to GPUPixel backend
- Added MaxPool (1D), AveragePool (1D) to GPUPixel backend
- Added Conv (1D and 3D) to GPUPixel backend
- Added BatchNormalization layer and op
- Added CreateBackend method to BackendFactory
- Added SimplifyReshapeInputPass to optimizer to reduce shapes to constants when they are inputs to Reshape layer
- Added data type inference and added data types to outputs in inspector
- Added optimization passes to simplify and/or remove trivial Reshape, Expand, Tile, Reduce, Cast and CastLike layers
- Added CustomLayer class to inherit custom layers from
- Added 'Serialize To StreamingAssets' option for model asset
- Added method to ModelLoader to Load from path
- Added methods to ModelWriter to SaveModelDesc and SaveModelWeights to memory streams
- Added 'Add', 'Div', 'Sub' and 'Mul' utility methods to Ops for operations between tensors and floats
- Added 'Set' utility method to Ops to set a slice of a tensor from another tensor similar to NumPy

### Removed
- Removed ScheduleAsyncDownload for asynchronous readback from GPU
- Removed internal CPU cache for Tensors, Tensor on GPU must call MakeReadable to access tensor with indexes

### Changed
- Split IOps interface to Ops class with helper functions and IBackend interface with backend methods
- Improved Documentation for style, accuracy and clarity
- Renamed uploadCache to clearOnInit for the Pin method
- Changed Ops with int[] and float[] inputs to accept Span and ReadOnlySpan to reduce allocation
- ConvTranspose now imports with default pads and strides when they aren't provided
- Optimized Broadcast ops speed (Add, Div, Sub, Mul, Pow...) on GPUCompute and GPUCommandBuffer
- Optimized Reduce ops speed (ReduceMax, ReduseSum....) on GPUCompute and GPUCommandBuffer
- Optimized Transpose ops speed on GPUCompute and GPUCommandBuffer
- Optimized LSTM speed on GPUCompute and GPUCommandBuffer
- Made RoundDenormalWeightsPass part of optimization passes and removed option from inspector
- Unified shape inference and partial tensor inference to improve import time and better reduce number of layers in optimized model
- Duplicated constants when used on both GPU and CPU to avoid unnecessary uploads and readbacks
- Reduced memory and CPU overhead in model importer
- Changed Conv and ConvTranspose to accept dynamic weight tensor and use "kernel_shape" parameter for shape inference
- PeekOutput in IWorker no longer accepts 'prepareCacheForAccess' parameter (use MakeReadable on returned tensor instead)

### Fixed
- Fixed MatMul for 1D input tensors
- Fixed ConvTranspose to work with non 2D cases
- Fixed Conv to accept null bias tensor
- Fixed GPUPixel backend to work with TensorInts
- Fixed import crash for model constants with zero length


## [1.0.0-exp.6] - 2023-06-26

### Added
- First beta build.

#Contributors
- Alexandre Ribard
- Aurimas Petrovas
- Bob Donovan
- Giles Coope
- Jeffrey Rainy
- Mark Green
