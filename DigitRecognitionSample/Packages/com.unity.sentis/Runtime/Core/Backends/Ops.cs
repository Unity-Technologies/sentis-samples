using System;
using UnityEngine;
using UnityEngine.Assertions;

namespace Unity.Sentis
{
    /// <summary>
    /// Represents an `Ops` object that runs on the `CPU` backend.
    /// </summary>
    public class CPUOps : Ops
    {
        /// <summary>
        /// Instantiates and returns a `CPUOps` object.
        /// </summary>
        /// <param name="allocator">The optional allocator to use for new tensor allocations.</param>
        public CPUOps(ITensorAllocator allocator = null)
            : base(BackendType.CPU, allocator) { }
    }

    /// <summary>
    /// Represents an `Ops` object that runs on the `GPUCompute` backend.
    /// </summary>
    public class GPUComputeOps : Ops
    {
        /// <summary>
        /// Instantiates and returns a `GPUComputeOps` object.
        /// </summary>
        /// <param name="allocator">The optional allocator to use for new tensor allocations.</param>
        public GPUComputeOps(ITensorAllocator allocator = null)
            : base(BackendType.GPUCompute, allocator) { }
    }

    /// <summary>
    /// Represents an `Ops` object that runs on the `GPUCommandBuffer` backend.
    /// </summary>
    public class GPUCommandBufferOps : Ops
    {
        /// <summary>
        /// Instantiates and returns a `GPUCommandBufferOps` object.
        /// </summary>
        /// <param name="allocator">The optional allocator to use for new tensor allocations.</param>
        public GPUCommandBufferOps(ITensorAllocator allocator = null)
            : base(BackendType.GPUCommandBuffer, allocator) { }
    }

    /// <summary>
    /// Represents an `Ops` object that runs on the `GPUPixel` backend.
    /// </summary>
    public class GPUPixelOps : Ops
    {
        /// <summary>
        /// Instantiates and returns a `GPUPixelOps` object.
        /// </summary>
        /// <param name="allocator">The optional allocator to use for new tensor allocations.</param>
        public GPUPixelOps(ITensorAllocator allocator = null)
            : base(BackendType.GPUPixel, allocator) { }
    }

    /// <summary>
    /// Represents an object for carrying out tensor operations.
    /// </summary>
    public abstract class Ops : IDisposable
    {
        ITensorAllocator m_Allocator;
        IBackend m_Backend;
        BackendType m_BackendType;

        /// <summary>
        /// The backend type for the operation execution.
        /// </summary>
        public BackendType backendType => m_BackendType;

        /// <summary>
        /// Instantiates and returns an `Ops` object.
        /// </summary>
        /// <param name="backendType">The backend type to use for operation execution.</param>
        /// <param name="allocator">The optional allocator to use for new tensor allocations.</param>
        protected Ops(BackendType backendType, ITensorAllocator allocator)
        {
            m_BackendType = backendType;
            m_Backend = BackendFactory.CreateBackend(backendType, allocator, false);
            m_Allocator = allocator ?? new TensorCachingAllocator();
        }

        /// <summary>
        /// Disposes of the `Ops` and any associated memory.
        /// </summary>
        public void Dispose()
        {
            m_Allocator?.Dispose();
            m_Backend?.Dispose();
        }

        /// <summary>
        /// Performs an element-wise `Add` math operation between a tensor and a float: f(a, b) = a + b.
        /// </summary>
        /// <param name="A">The first argument as a tensor.</param>
        /// <param name="b">The second argument as a float.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Add(TensorFloat A, float b)
        {
            var O = m_Backend.NewOutputTensorFloat(A.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.ScalarMad(A, O, 1, b);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `Add` math operation between a float and a tensor: f(a, b) = a + b.
        /// </summary>
        /// <param name="a">The first argument as a float.</param>
        /// <param name="B">The second argument as a tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Add(float a, TensorFloat B)
        {
            var O = m_Backend.NewOutputTensorFloat(B.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.ScalarMad(B, O, 1, a);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `Sub` math operation between a tensor and a float: f(a, b) = a - b.
        /// </summary>
        /// <param name="A">The first argument as a tensor.</param>
        /// <param name="b">The second argument as a float.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Sub(TensorFloat A, float b)
        {
            var O = m_Backend.NewOutputTensorFloat(A.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.ScalarMad(A, O, 1, -b);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `Sub` math operation between a float and a tensor: f(a, b) = a - b.
        /// </summary>
        /// <param name="a">The first argument as a float.</param>
        /// <param name="B">The second argument as a tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Sub(float a, TensorFloat B)
        {
            var O = m_Backend.NewOutputTensorFloat(B.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.ScalarMad(B, O, -1, a);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `Mul` math operation between a tensor and a float: f(a, b) = a * b.
        /// </summary>
        /// <param name="A">The first argument as a tensor.</param>
        /// <param name="b">The second argument as a float.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Mul(TensorFloat A, float b)
        {
            var O = m_Backend.NewOutputTensorFloat(A.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.ScalarMad(A, O, b, 0);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `Mul` math operation between a float and a tensor: f(a, b) = a * b.
        /// </summary>
        /// <param name="a">The first argument as a float.</param>
        /// <param name="B">The second argument as a tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Mul(float a, TensorFloat B)
        {
            var O = m_Backend.NewOutputTensorFloat(B.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.ScalarMad(B, O, a, 0);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `Div` math operation between a tensor and a float: f(a, b) = a / b.
        /// </summary>
        /// <param name="A">The first argument as a tensor.</param>
        /// <param name="b">The second argument as a float.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Div(TensorFloat A, float b)
        {
            var O = m_Backend.NewOutputTensorFloat(A.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.ScalarMad(A, O, 1/b, 0);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `Mad` math operation: multiplies and adds bias to a tensor: f(A, s, b) = s * A + b.
        /// </summary>
        /// <param name="A">The argument as a tensor.</param>
        /// <param name="s">The value of the scale for multiplication.</param>
        /// <param name="b">The value of the bias for addition.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Mad(TensorFloat A, float s, float b)
        {
            var O = m_Backend.NewOutputTensorFloat(A.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.ScalarMad(A, O, s, b);
            return O;
        }

        /// <summary>
        /// Updates values of A with values from B similar to setting a slice in numpy. A[..., start:end, ....] = B
        ///
        /// This returns a new tensor rather than working on A in-place.
        ///
        /// This supports numpy-style one-directional broadcasting of B into A.
        /// </summary>
        /// <param name="A">The first argument as a tensor.</param>
        /// <param name="B">The second argument as a tensor.</param>
        /// <param name="axis">The axis along which to set the slice.</param>
        /// <param name="start">The inclusive start of the slice.</param>
        /// <param name="end">The exclusive end of the slice.</param>
        /// <typeparam name="T">The tensor type.</typeparam>
        /// <returns>The computed output tensor.</returns>
        public T Set<T>(T A, T B, int axis, int start, int end) where T : Tensor
        {
            var dim = A.shape[axis];
            start = start < 0 ? dim + start : start;
            start = Mathf.Clamp(start, 0, dim);
            end = end < 0 ? dim + end : end;
            end = Mathf.Clamp(end, 0, dim);
            var updatesShape = B.shape.Broadcast(TensorShape.Ones(A.shape.rank));
            Assert.IsTrue(end - start == updatesShape[axis] || updatesShape[axis] == 1, "ValueError: cannot broadcast B to A for Set between start and end.");
            updatesShape[axis] = 1;
            updatesShape = A.shape.Broadcast(updatesShape);
            updatesShape[axis] = end - start;
            using var updates = Expand(B, updatesShape);
            using var indicesRange = Range(start, end, 1);
            var indicesOnesShape = TensorShape.Ones(A.shape.rank);
            indicesOnesShape[axis] = updatesShape[axis];
            using var indicesOnes = Reshape(indicesRange, indicesOnesShape);
            using var indices = Expand(indicesOnes, updatesShape) as TensorInt;
            return ScatterElements(A, indices, updates, axis, Layers.ScatterReductionMode.None);
        }

        /// <summary>
        /// Performs a matrix multiplication operation with optional transposes: f(a, b) = a' x b'.
        /// </summary>
        /// <param name="X">The first input tensor.</param>
        /// <param name="Y">The second input tensor.</param>
        /// <param name="xTranspose">Whether to transpose the first input tensor before performing the matrix multiplication.</param>
        /// <param name="yTranspose">Whether to transpose the second input tensor before performing the matrix multiplication.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat MatMul2D(TensorFloat X, TensorFloat Y, bool xTranspose, bool yTranspose)
        {
            var O = m_Backend.NewOutputTensorFloat(ShapeInference.Gemm(X.shape, Y.shape, xTranspose, yTranspose));
            if (O.shape.HasZeroDims())
                return O;
            if (X.shape.HasZeroDims() || Y.shape.HasZeroDims())
                m_Backend.MemSet(O, 0.0f);
            else
                m_Backend.MatMul2D(X, Y, O, xTranspose, yTranspose);
            return O;
        }

        /// <summary>
        /// Performs a multi-dimensional matrix multiplication operation: f(a, b) = a x b.
        /// </summary>
        /// <param name="X">The first input tensor.</param>
        /// <param name="Y">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat MatMul(TensorFloat X, TensorFloat Y)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape.MatMul(Y.shape));
            if (O.shape.HasZeroDims())
                return O;
            if (X.shape.HasZeroDims() || Y.shape.HasZeroDims())
                m_Backend.MemSet(O, 0.0f);
            else
                m_Backend.MatMul(X, Y, O);
            return O;
        }

        /// <summary>
        /// Performs a matrix multiplication operation: f(X, w, b) = X x W + B.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="W">The weights tensor.</param>
        /// <param name="B">The bias tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Dense(TensorFloat X, TensorFloat W, TensorFloat B)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape.MatMul(W.shape));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Dense(X, W, B, O, Layers.FusableActivation.None);
            return O;
        }

        /// <summary>
        /// Computes the output tensor by retaining the lower triangular values from an input matrix or matrix batch and setting the other values to zero.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="k">The offset from the diagonal to keep.</param>
        /// <typeparam name="T">The tensor type.</typeparam>
        /// <returns>The computed output tensor.</returns>
        public T Tril<T>(T X, int k = 0) where T : Tensor
        {
            var O = m_Backend.NewOutputTensor(X.shape, X.dataType) as T;
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Tril(X, O, k);
            return O;
        }

        /// <summary>
        /// Computes the output tensor by retaining the upper triangular values from an input matrix or matrix batch and setting the other values to zero.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="k">The offset from the diagonal to exclude.</param>
        /// <typeparam name="T">The tensor type.</typeparam>
        /// <returns>The computed output tensor.</returns>
        public T Triu<T>(T X, int k = 0) where T : Tensor
        {
            var O = m_Backend.NewOutputTensor(X.shape, X.dataType) as T;
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Triu(X, O, k);
            return O;
        }

        /// <summary>
        /// Applies a convolution filter to an input tensor.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="K">The filter tensor.</param>
        /// <param name="B">The optional bias tensor.</param>
        /// <param name="groups">The number of groups that input channels and output channels are divided into.</param>
        /// <param name="stride">The optional stride value for each spatial dimension of the filter.</param>
        /// <param name="pad">The optional lower and upper padding values for each spatial dimension of the filter.</param>
        /// <param name="dilation">The optional dilation value of each spatial dimension of the filter.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Conv(TensorFloat X, TensorFloat K, TensorFloat B, int groups, int[] stride, int[] pad, int[] dilation)
        {
            var O = m_Backend.NewOutputTensorFloat(ShapeInference.Conv(X.shape, K.shape, groups, stride, pad, dilation));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Conv(X, K, B, O, groups, stride, pad, dilation, Layers.FusableActivation.None);
            return O;
        }

        /// <summary>
        /// Applies a transpose convolution filter to an input tensor.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="K">The filter tensor.</param>
        /// <param name="B">The optional bias tensor.</param>
        /// <param name="stride">The optional stride value for each spatial dimension of the filter.</param>
        /// <param name="pad">The optional lower and upper padding values for each spatial dimension of the filter.</param>
        /// <param name="outputAdjustment">The output padding value for each spatial dimension in the filter.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat ConvTranspose(TensorFloat X, TensorFloat K, TensorFloat B, int[] stride, int[] pad, int[] outputAdjustment)
        {
            var O = m_Backend.NewOutputTensorFloat(ShapeInference.ConvTranspose(X.shape, K.shape, stride, pad, outputAdjustment));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.ConvTranspose(X, K, B, O, stride, pad, outputAdjustment, Layers.FusableActivation.None);
            return O;
        }

        /// <summary>
        /// Calculates an output tensor by resampling the input tensor along the spatial dimensions with given scales.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="scale">The factor to scale each dimension by.</param>
        /// <param name="interpolationMode">The `InterpolationMode` to use for the operation.</param>
        /// <param name="nearestMode">The `NearestMode` to use for the operation when using `InterpolationMode.NearestMode`. The default is `NearestMode.RoundPreferFloor`.</param>
        /// <param name="coordTransformMode">The `CoordTransformMode` to use for the operation. The default is `CoordTransformMode.HalfPixel`.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Resize(TensorFloat X, ReadOnlySpan<float> scale, Layers.InterpolationMode interpolationMode, Layers.NearestMode nearestMode = Layers.NearestMode.RoundPreferFloor, Layers.CoordTransformMode coordTransformMode = Layers.CoordTransformMode.HalfPixel)
        {
            var O = m_Backend.NewOutputTensorFloat(ShapeInference.Resize(X.shape, scale));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Resize(X, O, scale, interpolationMode, nearestMode, coordTransformMode);
            return O;
        }

        /// <summary>
        /// Computes the output tensor by permuting data from depth into blocks of spatial data.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="blocksize">The size of the blocks to move the depth data into.</param>
        /// <param name="mode">The ordering of the data in the output tensor as a `DepthToSpaceMode`.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat DepthToSpace(TensorFloat X, int blocksize, Layers.DepthToSpaceMode mode)
        {
            var O = m_Backend.NewOutputTensorFloat(ShapeInference.DepthToSpace(X.shape, blocksize));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.DepthToSpace(X, O, blocksize, mode);
            return O;
        }

        /// <summary>
        /// Computes the output tensor by permuting data from blocks of spatial data into depth.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="blocksize">The size of the blocks to move the depth data into.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat SpaceToDepth(TensorFloat X, int blocksize)
        {
            var O = m_Backend.NewOutputTensorFloat(ShapeInference.SpaceToDepth(X.shape, blocksize));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.SpaceToDepth(X, O, blocksize);
            return O;
        }

        /// <summary>
        /// Calculates an output tensor by pooling the maximum values of the input tensor across its spatial dimensions according to the given pool and stride values.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="pool">The size of the kernel along each spatial axis.</param>
        /// <param name="stride">The stride along each spatial axis.</param>
        /// <param name="pad">The lower and upper padding values for each spatial dimension. For example, [pad_left, pad_right] for 1D, or [pad_top, pad_bottom, pad_left, pad_right] for 2D.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat MaxPool(TensorFloat X, int[] pool, int[] stride, int[] pad)
        {
            var O = m_Backend.NewOutputTensorFloat(ShapeInference.ApplyPool(X.shape, pool, stride, pad));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.MaxPool(X, O, pool, stride, pad);
            return O;
        }

        /// <summary>
        /// Calculates an output tensor by pooling the mean values of the input tensor across its spatial dimensions according to the given pool and stride values.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="pool">The size of the kernel along each spatial axis.</param>
        /// <param name="stride">The stride along each spatial axis.</param>
        /// <param name="pad">The lower and upper padding values for each spatial dimension. For example, [pad_left, pad_right] for 1D, or [pad_top, pad_bottom, pad_left, pad_right] for 2D.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat AveragePool(TensorFloat X, int[] pool, int[] stride, int[] pad)
        {
            var O = m_Backend.NewOutputTensorFloat(ShapeInference.ApplyPool(X.shape, pool, stride, pad));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.AveragePool(X, O, pool, stride, pad);
            return O;
        }

        /// <summary>
        /// Calculates an output tensor by pooling the maximum values of the input tensor across all of its spatial dimensions. The spatial dimensions of the output are size 1.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat GlobalMaxPool(TensorFloat X)
        {
            var O = m_Backend.NewOutputTensorFloat(ShapeInference.GlobalPool(X.shape));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.GlobalMaxPool(X, O);
            return O;
        }

        /// <summary>
        /// Calculates an output tensor by pooling the mean values of the input tensor across all of its spatial dimensions. The spatial dimensions of the output are size 1.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat GlobalAveragePool(TensorFloat X)
        {
            var O = m_Backend.NewOutputTensorFloat(ShapeInference.GlobalPool(X.shape));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.GlobalAveragePool(X, O);
            return O;
        }

        /// <summary>
        /// Calculates the output tensor by adding padding to the input tensor according to the given padding values and mode.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="pad">The lower and upper padding values for each dimension.</param>
        /// <param name="padMode">The `PadMode` to use when padding. The default value is `PadMode.Constant`.</param>
        /// <param name="constant">The constant value to fill with when using `PadMode.Constant`. The default value is 0.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Pad(TensorFloat X, ReadOnlySpan<int> pad, Layers.PadMode padMode = Layers.PadMode.Constant, float constant = 0.0f)
        {
            if (padMode != Layers.PadMode.Constant)
                Assert.IsFalse(X.shape.HasZeroDims(), "ValueError: zero dimensions input for Pad operator is not supported");
            var O = m_Backend.NewOutputTensorFloat(X.shape.Pad(pad));
            if (O.shape.HasZeroDims())
                return O;
            if (X.shape.HasZeroDims())
                m_Backend.MemSet(O, constant);
            else
                m_Backend.Pad(X, O, pad, padMode, constant);
            return O;
        }

        /// <summary>
        /// Computes the output tensor with an element-wise `ScaleBias` function: f(X, s, b) = x * s + b.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="S">The scale tensor.</param>
        /// <param name="B">The bias tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat ScaleBias(TensorFloat X, TensorFloat S, TensorFloat B)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.ScaleBias(X, S, B, O);
            return O;
        }

        /// <summary>
        /// Computes the mean variance on the spatial dimensions of the input tensor and normalizes them according to `scale` and `bias` tensors.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="S">The scale tensor.</param>
        /// <param name="B">The bias tensor.</param>
        /// <param name="epsilon">The epsilon value the layer uses to avoid division by zero.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat InstanceNormalization(TensorFloat X, TensorFloat S, TensorFloat B, float epsilon)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.InstanceNormalization(X, S, B, O, epsilon);
            return O;
        }

        /// <summary>
        /// Computes the mean variance on the last dimension of the input tensor and normalizes it according to `scale` and `bias` tensors.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="S">The scale tensor.</param>
        /// <param name="B">The bias tensor.</param>
        /// <param name="epsilon">The epsilon value the layer uses to avoid division by zero.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat LayerNormalization(TensorFloat X, TensorFloat S, TensorFloat B, float epsilon)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.LayerNormalization(X, S, B, O, epsilon);
            return O;
        }

        /// <summary>
        /// Normalizes the input tensor over local input regions.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="alpha">The scaling parameter to use for the normalization.</param>
        /// <param name="beta">The exponent to use for the normalization.</param>
        /// <param name="bias">The bias value to use for the normalization.</param>
        /// <param name="size">The number of channels to sum over.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat LRN(TensorFloat X, float alpha, float beta, float bias, int size)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.LRN(X, O, alpha, beta, bias, size);
            return O;
        }

        /// <summary>
        /// Generates an output tensor of a given shape with random values in a normal distribution with given `mean` and `scale`, and an optional `seed` value.
        /// </summary>
        /// <param name="S">The shape to use for the output tensor.</param>
        /// <param name="mean">The mean of the normal distribution to use to generate the output.</param>
        /// <param name="scale">The standard deviation of the normal distribution to use to generate the output.</param>
        /// <param name="seed">The optional seed to use for the random number generation. If this is `null` the operation generates a seed using `System.Random()`.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat RandomNormal(TensorShape S, float mean, float scale, float? seed)
        {
            var O = m_Backend.NewOutputTensorFloat(S);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.RandomNormal(O, mean, scale, seed);
            return O;
        }

        /// <summary>
        /// Generates an output tensor of a given shape with random values in a uniform distribution between a given `low` and `high`, and an optional `seed` value.
        /// </summary>
        /// <param name="S">The shape to use for the output tensor.</param>
        /// <param name="low">The lower end of the interval of the uniform distribution to use to generate the output.</param>
        /// <param name="high">The upper end of the interval of the uniform distribution to use to generate the output.</param>
        /// <param name="seed">The optional seed to use for the random number generation. If this is `null` the operation generates a seed using `System.Random()`.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat RandomUniform(TensorShape S, float low, float high, float? seed)
        {
            var O = m_Backend.NewOutputTensorFloat(S);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.RandomUniform(O, low, high, seed);
            return O;
        }

        /// <summary>
        /// Generates a one-hot tensor with a given `depth`, `indices` and on and off values.
        /// </summary>
        /// <param name="indices">The indices input tensor.</param>
        /// <param name="axis">The axis along which the operation adds the one-hot representation.</param>
        /// <param name="depth">The depth of the one-hot tensor.</param>
        /// <param name="offValue">The value to use for an off element.</param>
        /// <param name="onValue">The value to use for an on element.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt OneHot(TensorInt indices, int axis, int depth, int offValue, int onValue)
        {
            var O = m_Backend.NewOutputTensorInt(ShapeInference.OneHot(indices.shape, axis, depth));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.OneHot(indices, O, axis, depth, offValue, onValue);
            return O;
        }

        /// <summary>
        /// Calculates an output tensor by pooling the input tensor across each region of interest given by the `rois` tensor.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="Rois">The region of interest input tensor.</param>
        /// <param name="Indices">The indices input tensor.</param>
        /// <param name="mode">The pooling mode of the operation as an `RoiPoolingMode`.</param>
        /// <param name="outputHeight">The height of the output tensor.</param>
        /// <param name="outputWidth">The width of the output tensor.</param>
        /// <param name="samplingRatio">The number of sampling points in the interpolation grid used to compute the output value of each pooled output bin.</param>
        /// <param name="spatialScale">The multiplicative spatial scale factor used to translate coordinates from their input spatial scale to the scale used when pooling.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat RoiAlign(TensorFloat X, TensorFloat Rois, TensorInt Indices, Layers.RoiPoolingMode mode, int outputHeight, int outputWidth, int samplingRatio, float spatialScale)
        {
            var O = m_Backend.NewOutputTensorFloat(ShapeInference.RoiAlign(X.shape, Rois.shape, Indices.shape, outputHeight, outputWidth));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.RoiAlign(X, Rois, Indices, O, mode, outputHeight, outputWidth, samplingRatio, spatialScale);
            return O;
        }

        /// <summary>
        /// Generates a 1D output tensor where the values form an arithmetic progression defined by the `start`, `limit`, and `delta` values.
        /// </summary>
        /// <param name="start">The first value in the range.</param>
        /// <param name="limit">The limit of the range.</param>
        /// <param name="delta">The delta between subsequent values in the range.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Range(float start, float limit, float delta)
        {
            var O = m_Backend.NewOutputTensorFloat(ShapeInference.Range(start, limit, delta));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Range(O, start, delta);
            return O;
        }

        /// <summary>
        /// Generates a 1D output tensor where the values form an arithmetic progression defined by the `start`, `limit`, and `delta` values.
        /// </summary>
        /// <param name="start">The first value in the range.</param>
        /// <param name="limit">The limit of the range.</param>
        /// <param name="delta">The delta between subsequent values in the range.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Range(int start, int limit, int delta)
        {
            var O = m_Backend.NewOutputTensorInt(ShapeInference.Range(start, limit, delta));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Range(O, start, delta);
            return O;
        }

        /// <summary>
        /// Generates an output tensor with values 0 or 1 from a Bernoulli distribution. The input tensor contains the probabilities to use for generating the output values.
        /// </summary>
        /// <param name="X">The probabilities input tensor.</param>
        /// <param name="dataType">The data type of the output tensor.</param>
        /// <param name="seed">The optional seed to use for the random number generation. If this is `null` the operation generates a seed using `System.Random()`.</param>
        /// <returns>The computed output tensor.</returns>
        public Tensor Bernoulli(TensorFloat X, DataType dataType, float? seed)
        {
            var O = m_Backend.NewOutputTensor(X.shape, dataType);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Bernoulli(X, O, seed);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Relu` activation function: f(x) = max(0, x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Relu(TensorFloat X)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Relu(X, O);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the `Softmax` activation function along an axis: f(X, axis) = exp(X) / ReduceSum(exp(X), axis).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axis">The axis along which to apply the `Softmax` activation function. The default value is -1.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Softmax(TensorFloat X, int axis = -1)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Softmax(X, O, axis);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the `LogSoftmax` activation function along an axis: f(X, axis) = log(Softmax(X, axis)).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axis">The axis along which to apply the `LogSoftmax` activation function. The default value is -1.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat LogSoftmax(TensorFloat X, int axis = -1)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.LogSoftmax(X, O, axis);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the `Hardmax` activation function along an axis: f(X, axis) = 1 if x is the first maximum value along the specified axis, otherwise f(x) = 0.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axis">The axis along which to apply the `Hardmax` activation function. The default value is -1.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Hardmax(TensorFloat X, int axis = -1)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Hardmax(X, O, axis);
            return O;
        }

        /// <summary>
        /// Performs the cumulative sum along a given axis.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axis">The axis along which to apply the cumulative sum.</param>
        /// <param name="reverse">Whether to perform the cumulative sum from the end of the axis.</param>
        /// <param name="exclusive">Whether to include the respective input element in the cumulative sum.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat CumSum(TensorFloat X, int axis, bool reverse = false, bool exclusive = false)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.CumSum(X, O, axis, reverse: reverse, exclusive: exclusive);
            return O;
        }

        /// <summary>
        /// Performs the cumulative sum along a given axis.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axis">The axis along which to apply the cumulative sum.</param>
        /// <param name="reverse">Whether to perform the cumulative sum from the end of the axis.</param>
        /// <param name="exclusive">Whether to include the respective input element in the cumulative sum.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt CumSum(TensorInt X, int axis, bool reverse = false, bool exclusive = false)
        {
            var O = m_Backend.NewOutputTensorInt(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.CumSum(X, O, axis, reverse: reverse, exclusive: exclusive);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Tanh` activation function: f(x) = tanh(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Tanh(TensorFloat X)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Tanh(X, O);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Softplus` activation function: f(x) = ln(e^x + 1).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Softplus(TensorFloat X)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Softplus(X, O);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Sigmoid` activation function: f(x) = 1/(1 + e^(-x)).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Sigmoid(TensorFloat X)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Sigmoid(X, O);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `HardSigmoid` activation function: f(x) = clamp(alpha * x + beta, 0, 1).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="alpha">The alpha value to use for the `HardSigmoid` activation function.</param>
        /// <param name="beta">The beta value to use for the `HardSigmoid` activation function.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat HardSigmoid(TensorFloat X, float alpha, float beta)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.HardSigmoid(X, O, alpha, beta);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Elu` activation function: f(x) = x if x >= 0, otherwise f(x) = alpha * (e^x - 1).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="alpha">The alpha value to use for the `Elu` activation function.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Elu(TensorFloat X, float alpha)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Elu(X, O, alpha);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Gelu` activation function: f(x) = x / 2 * (1 + erf(x / sqrt(2))).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Gelu(TensorFloat X)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Gelu(X, O);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Relu6` activation function: f(x) = clamp(X, 0, 6).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Relu6(TensorFloat X)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Relu6(X, O);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `LeakyRelu` activation function: f(x) = x if x >= 0, otherwise f(x) = alpha * x.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="alpha">The alpha value to use for the `LeakyRelu` activation function.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat LeakyRelu(TensorFloat X, float alpha)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.LeakyRelu(X, O, alpha);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Selu` activation function: f(x) = gamma * x if x >= 0, otherwise f(x) = (alpha * e^x - alpha).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="alpha">The alpha value to use for the `Selu` activation function.</param>
        /// <param name="gamma">The alpha value to use for the `Selu` activation function.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Selu(TensorFloat X, float alpha, float gamma)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Selu(X, O, alpha, gamma);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `PRelu` activation function: f(x) = x if x >= 0, otherwise f(x) = slope * x.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="slope">The slope tensor, must be unidirectional broadcastable to x.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat PRelu(TensorFloat X, TensorFloat slope)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.PRelu(X, slope, O);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Swish` activation function: f(x) = sigmoid(x) * x = x / (1 + e^{-x}).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Swish(TensorFloat X)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Swish(X, O);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Abs` math function: f(x) = f(x) = |x|.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Abs(TensorFloat X)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Abs(X, O);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Abs` math function: f(x) = f(x) = |x|.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Abs(TensorInt X)
        {
            var O = m_Backend.NewOutputTensorInt(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Abs(X, O);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Neg` math function: f(x) = -x.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Neg(TensorFloat X)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Neg(X, O);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Neg` math function: f(x) = -x.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Neg(TensorInt X)
        {
            var O = m_Backend.NewOutputTensorInt(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Neg(X, O);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Ceil` math function: f(x) = ceil(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Ceil(TensorFloat X)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Ceil(X, O);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Clip` math function: f(x) = clamp(X, min, max).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="min">The lower clip value.</param>
        /// <param name="max">The upper clip value.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Clip(TensorFloat X, float min, float max)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Clip(X, O, min, max);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Floor` math function: f(x) = floor(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Floor(TensorFloat X)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Floor(X, O);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Round` math function: f(x) = round(x).
        ///
        /// If the fractional part is equal to 0.5, rounds to the nearest even integer.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Round(TensorFloat X)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Round(X, O);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Reciprocal` math function: f(x) = 1 / x.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Reciprocal(TensorFloat X)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Reciprocal(X, O);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Square` math function: f(x) = x * x.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Square(TensorFloat X)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Square(X, O);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Exp` math function: f(x) = exp(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Exp(TensorFloat X)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Exp(X, O);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Log` math function: f(x) = log(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Log(TensorFloat X)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Log(X, O);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Sqrt` math function: f(x) = sqrt(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Sqrt(TensorFloat X)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Sqrt(X, O);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Acos` trigonometric function: f(x) = acos(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Acos(TensorFloat X)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Acos(X, O);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Acosh` trigonometric function: f(x) = acosh(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Acosh(TensorFloat X)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Acosh(X, O);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Asin` trigonometric function: f(x) = asin(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Asin(TensorFloat X)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Asin(X, O);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Asinh` trigonometric function: f(x) = asinh(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Asinh(TensorFloat X)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Asinh(X, O);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Atan` trigonometric function: f(x) = atan(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Atan(TensorFloat X)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Atan(X, O);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Atanh` trigonometric function: f(x) = atanh(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Atanh(TensorFloat X)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Atanh(X, O);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Cos` trigonometric function: f(x) = cos(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Cos(TensorFloat X)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Cos(X, O);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Cosh` trigonometric function: f(x) = cosh(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Cosh(TensorFloat X)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Cosh(X, O);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Sin` trigonometric function: f(x) = sin(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Sin(TensorFloat X)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Sin(X, O);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Sinh` trigonometric function: f(x) = sinh(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Sinh(TensorFloat X)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Sinh(X, O);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Tan` trigonometric function: f(x) = tan(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Tan(TensorFloat X)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Tan(X, O);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Erf` activation function: f(x) = erf(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Erf(TensorFloat X)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Erf(X, O);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Celu` activation function: f(x) = max(0, x) + min(0, alpha * (exp(x / alpha) - 1)).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="alpha">The alpha value to use for the `Celu` activation function.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Celu(TensorFloat X, float alpha)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Celu(X, O, alpha);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `HardSwish` activation function: f(x) = x * max(0, min(1, 1/6 * x + 0.5)).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat HardSwish(TensorFloat X)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.HardSwish(X, O);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Shrink` activation function: f(x) = x + bias if x &lt; lambd. f(x) = x - bias if x &gt; lambd. Otherwise f(x) = 0.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="bias">The bias value to use for the `Shrink` activation function.</param>
        /// <param name="lambd">The lambda value to use for the `Shrink` activation function.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Shrink(TensorFloat X, float bias, float lambd)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Shrink(X, O, bias, lambd);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Softsign` activation function: f(x) = x/(|x| + 1).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Softsign(TensorFloat X)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Softsign(X, O);
            return O;
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `ThresholdedRelu` activation function: f(x) = x if x > alpha, otherwise f(x) = 0.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="alpha">The alpha value to use for the `ThresholdedRelu` activation function.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat ThresholdedRelu(TensorFloat X, float alpha)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.ThresholdedRelu(X, O, alpha);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `Sum` math operation: f(x1, x2 ... xn) = x1 + x2 ... xn.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="tensors">The input tensors.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Sum(params TensorFloat[] tensors)
        {
            var O = m_Backend.NewOutputTensorFloat(TensorShapeHelper.BroadcastShape(tensors));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Sum(tensors, O);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `Add` math operation: f(a, b) = a + b.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Add(TensorFloat A, TensorFloat B)
        {
            var O = m_Backend.NewOutputTensorFloat(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Add(A, B, O);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `Add` math operation: f(a, b) = a + b.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Add(TensorInt A, TensorInt B)
        {
            var O = m_Backend.NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Add(A, B, O);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `Sub` math operation: f(a, b) = a - b.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Sub(TensorFloat A, TensorFloat B)
        {
            var O = m_Backend.NewOutputTensorFloat(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Sub(A, B, O);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `Sub` math operation: f(a, b) = a - b.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Sub(TensorInt A, TensorInt B)
        {
            var O = m_Backend.NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Sub(A, B, O);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `Mul` math operation: f(a, b) = a * b.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Mul(TensorFloat A, TensorFloat B)
        {
            var O = m_Backend.NewOutputTensorFloat(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Mul(A, B, O);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `Mul` math operation: f(a, b) = a * b.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Mul(TensorInt A, TensorInt B)
        {
            var O = m_Backend.NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Mul(A, B, O);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `Div` math operation: f(a, b) = a / b.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Div(TensorFloat A, TensorFloat B)
        {
            var O = m_Backend.NewOutputTensorFloat(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Div(A, B, O);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `Div` math operation: f(a, b) = a / b.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Div(TensorInt A, TensorInt B)
        {
            var O = m_Backend.NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Div(A, B, O);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `Mod` math operation: f(a, b) = a % b.
        ///
        /// The sign of the remainder is the same as the sign of the divisor, as in Python.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Mod(TensorInt A, TensorInt B)
        {
            var O = m_Backend.NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Mod(A, B, O);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `Mod` math operation: f(a, b) = a % b.
        ///
        /// The sign of the remainder is the same as the sign of the dividend, as in C#.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt FMod(TensorInt A, TensorInt B)
        {
            var O = m_Backend.NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.FMod(A, B, O);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `Mod` math operation: f(a, b) = a % b.
        ///
        /// The sign of the remainder is the same as the sign of the dividend, as in C#.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat FMod(TensorFloat A, TensorFloat B)
        {
            var O = m_Backend.NewOutputTensorFloat(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.FMod(A, B, O);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `Pow` math operation: f(a, b) = pow(a, b).
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Pow(TensorFloat A, TensorFloat B)
        {
            var O = m_Backend.NewOutputTensorFloat(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Pow(A, B, O);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `Pow` math operation: f(a, b) = pow(a, b).
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Pow(TensorFloat A, TensorInt B)
        {
            var O = m_Backend.NewOutputTensorFloat(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Pow(A, B, O);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `Min` math operation: f(x1, x2 ... xn) = min(x1, x2 ... xn).
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="tensors">The input tensors.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Min(params TensorFloat[] tensors)
        {
            var O = m_Backend.NewOutputTensorFloat(TensorShapeHelper.BroadcastShape(tensors));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Min(tensors, O);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `Min` math operation: f(x1, x2 ... xn) = min(x1, x2 ... xn).
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="tensors">The input tensors.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Min(params TensorInt[] tensors)
        {
            var O = m_Backend.NewOutputTensorInt(TensorShapeHelper.BroadcastShape(tensors));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Min(tensors, O);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `Max` math operation: f(x1, x2 ... xn) = max(x1, x2 ... xn).
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="tensors">The input tensors.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Max(params TensorFloat[] tensors)
        {
            var O = m_Backend.NewOutputTensorFloat(TensorShapeHelper.BroadcastShape(tensors));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Max(tensors, O);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `Max` math operation: f(x1, x2 ... xn) = max(x1, x2 ... xn).
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="tensors">The input tensors.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Max(params TensorInt[] tensors)
        {
            var O = m_Backend.NewOutputTensorInt(TensorShapeHelper.BroadcastShape(tensors));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Max(tensors, O);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `Mean` math operation: f(x1, x2 ... xn) = (x1 + x2 ... xn) / n.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="tensors">The input tensors.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Mean(params TensorFloat[] tensors)
        {
            var O = m_Backend.NewOutputTensorFloat(TensorShapeHelper.BroadcastShape(tensors));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Mean(tensors, O);
            return O;
        }

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceMax` operation: f(x1, x2 ... xn) = max(x1, x2, ... , xn).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat ReduceMax(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape.Reduce(axes, keepdim));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.ReduceMax(X, O, axes, keepdim);
            return O;
        }

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceMean` operation: f(x1, x2 ... xn) = max(x1, x2, ... , xn).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt ReduceMax(TensorInt X, ReadOnlySpan<int> axes, bool keepdim)
        {
            var O = m_Backend.NewOutputTensorInt(X.shape.Reduce(axes, keepdim));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.ReduceMax(X, O, axes, keepdim);
            return O;
        }

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceMean` operation: f(x1, x2 ... xn) = (x1 + x2 + ... + xn) / n.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat ReduceMean(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape.Reduce(axes, keepdim));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.ReduceMean(X, O, axes, keepdim);
            return O;
        }

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceMin` operation: f(x1, x2 ... xn) = min(x1, x2, ... , xn).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat ReduceMin(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape.Reduce(axes, keepdim));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.ReduceMin(X, O, axes, keepdim);
            return O;
        }

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceMin` operation: f(x1, x2 ... xn) = min(x1, x2, ... , xn).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt ReduceMin(TensorInt X, ReadOnlySpan<int> axes, bool keepdim)
        {
            var O = m_Backend.NewOutputTensorInt(X.shape.Reduce(axes, keepdim));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.ReduceMin(X, O, axes, keepdim);
            return O;
        }

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceProd` operation: f(x1, x2 ... xn) = x1 * x2 * ... * xn.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat ReduceProd(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape.Reduce(axes, keepdim));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.ReduceProd(X, O, axes, keepdim);
            return O;
        }

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceProd` operation: f(x1, x2 ... xn) = x1 * x2 * ... * xn.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt ReduceProd(TensorInt X, ReadOnlySpan<int> axes, bool keepdim)
        {
            var O = m_Backend.NewOutputTensorInt(X.shape.Reduce(axes, keepdim));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.ReduceProd(X, O, axes, keepdim);
            return O;
        }

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceSum` operation: f(x1, x2 ... xn) = x1 + x2 + ... + xn.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat ReduceSum(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape.Reduce(axes, keepdim));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.ReduceSum(X, O, axes, keepdim);
            return O;
        }

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceSum` operation: f(x1, x2 ... xn) = x1 + x2 + ... + xn.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt ReduceSum(TensorInt X, ReadOnlySpan<int> axes, bool keepdim)
        {
            var O = m_Backend.NewOutputTensorInt(X.shape.Reduce(axes, keepdim));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.ReduceSum(X, O, axes, keepdim);
            return O;
        }

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceSumSquare` operation: f(x1, x2 ... xn) = x1² + x2² + ... + xn².
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat ReduceSumSquare(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape.Reduce(axes, keepdim));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.ReduceSumSquare(X, O, axes, keepdim);
            return O;
        }

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceSumSquare` operation: f(x1, x2 ... xn) = x1² + x2² + ... + xn².
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt ReduceSumSquare(TensorInt X, ReadOnlySpan<int> axes, bool keepdim)
        {
            var O = m_Backend.NewOutputTensorInt(X.shape.Reduce(axes, keepdim));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.ReduceSumSquare(X, O, axes, keepdim);
            return O;
        }

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceL1` operation: f(x1, x2 ... xn) = |x1| + |x2| + ... + |xn|.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat ReduceL1(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape.Reduce(axes, keepdim));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.ReduceL1(X, O, axes, keepdim);
            return O;
        }

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceL1` operation: f(x1, x2 ... xn) = |x1| + |x2| + ... + |xn|.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt ReduceL1(TensorInt X, ReadOnlySpan<int> axes, bool keepdim)
        {
            var O = m_Backend.NewOutputTensorInt(X.shape.Reduce(axes, keepdim));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.ReduceL1(X, O, axes, keepdim);
            return O;
        }

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceL2` operation: f(x1, x2 ... xn) = sqrt(x1² + x2² + ... + xn²).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat ReduceL2(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape.Reduce(axes, keepdim));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.ReduceL2(X, O, axes, keepdim);
            return O;
        }

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceLogSum` operation: f(x1, x2 ... xn) = log(x1 + x2 + ... + xn).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat ReduceLogSum(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape.Reduce(axes, keepdim));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.ReduceLogSum(X, O, axes, keepdim);
            return O;
        }

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceLogSumExp` operation: f(x1, x2 ... xn) = log(e^x1 + e^x2 + ... + e^xn).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat ReduceLogSumExp(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape.Reduce(axes, keepdim));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.ReduceLogSumExp(X, O, axes, keepdim);
            return O;
        }

        /// <summary>
        /// Computes the indices of the maximum elements of the input tensor along a given axis.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axis">The axis along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <param name="selectLastIndex">Whether to perform the operation from the back of the axis.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt ArgMax(TensorFloat X, int axis, bool keepdim, bool selectLastIndex = false)
        {
            var O = m_Backend.NewOutputTensorInt(X.shape.Reduce(axis, keepdim));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.ArgMax(X, O, axis, keepdim, selectLastIndex);
            return O;
        }

        /// <summary>
        /// Computes the indices of the maximum elements of the input tensor along a given axis.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axis">The axis along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <param name="selectLastIndex">Whether to perform the operation from the back of the axis.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt ArgMax(TensorInt X, int axis, bool keepdim, bool selectLastIndex = false)
        {
            var O = m_Backend.NewOutputTensorInt(X.shape.Reduce(axis, keepdim));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.ArgMax(X, O, axis, keepdim, selectLastIndex);
            return O;
        }

        /// <summary>
        /// Computes the indices of the minimum elements of the input tensor along a given axis.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axis">The axis along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <param name="selectLastIndex">Whether to perform the operation from the back of the axis.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt ArgMin(TensorFloat X, int axis, bool keepdim, bool selectLastIndex)
        {
            var O = m_Backend.NewOutputTensorInt(X.shape.Reduce(axis, keepdim));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.ArgMin(X, O, axis, keepdim, selectLastIndex);
            return O;
        }

        /// <summary>
        /// Computes the indices of the minimum elements of the input tensor along a given axis.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axis">The axis along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <param name="selectLastIndex">Whether to perform the operation from the back of the axis.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt ArgMin(TensorInt X, int axis, bool keepdim, bool selectLastIndex)
        {
            var O = m_Backend.NewOutputTensorInt(X.shape.Reduce(axis, keepdim));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.ArgMin(X, O, axis, keepdim, selectLastIndex);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `Greater` logical comparison operation: f(a, b) = 1 if a > b, otherwise f(x) = 0.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Greater(TensorFloat A, TensorFloat B)
        {
            var O = m_Backend.NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Greater(A, B, O);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `Greater` logical comparison operation: f(a, b) = 1 if a > b, otherwise f(x) = 0.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Greater(TensorInt A, TensorInt B)
        {
            var O = m_Backend.NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Greater(A, B, O);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `GreaterOrEqual` logical comparison operation: f(a, b) = 1 if a >= b, otherwise f(x) = 0.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt GreaterOrEqual(TensorFloat A, TensorFloat B)
        {
            var O = m_Backend.NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.GreaterOrEqual(A, B, O);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `GreaterOrEqual` logical comparison operation: f(a, b) = 1 if a >= b, otherwise f(x) = 0.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt GreaterOrEqual(TensorInt A, TensorInt B)
        {
            var O = m_Backend.NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.GreaterOrEqual(A, B, O);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `Less` logical comparison operation: f(a, b) = 1 if a &lt; b, otherwise f(x) = 0.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Less(TensorFloat A, TensorFloat B)
        {
            var O = m_Backend.NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Less(A, B, O);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `Less` logical comparison operation: f(a, b) = 1 if a &lt; b, otherwise f(x) = 0.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Less(TensorInt A, TensorInt B)
        {
            var O = m_Backend.NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Less(A, B, O);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `LessOrEqual` logical comparison operation: f(a, b) = 1 if a &lt;= b, otherwise f(x) = 0.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt LessOrEqual(TensorFloat A, TensorFloat B)
        {
            var O = m_Backend.NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.LessOrEqual(A, B, O);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `LessOrEqual` logical comparison operation: f(a, b) = 1 if a &lt;= b, otherwise f(x) = 0.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt LessOrEqual(TensorInt A, TensorInt B)
        {
            var O = m_Backend.NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.LessOrEqual(A, B, O);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `Equal` logical comparison operation: f(a, b) = 1 if a == b, otherwise f(x) = 0.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Equal(TensorFloat A, TensorFloat B)
        {
            var O = m_Backend.NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Equal(A, B, O);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `Equal` logical comparison operation: f(a, b) = 1 if a == b, otherwise f(x) = 0.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Equal(TensorInt A, TensorInt B)
        {
            var O = m_Backend.NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Equal(A, B, O);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `Or` logical operation: f(a, b) = a | b.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Or(TensorInt A, TensorInt B)
        {
            var O = m_Backend.NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Or(A, B, O);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `And` logical operation: f(a, b) = a &amp; b.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt And(TensorInt A, TensorInt B)
        {
            var O = m_Backend.NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.And(A, B, O);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `Xor` logical operation: f(a) = a ^ b.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Xor(TensorInt A, TensorInt B)
        {
            var O = m_Backend.NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Xor(A, B, O);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `Not` logical operation: f(x) = ~x.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Not(TensorInt X)
        {
            var O = m_Backend.NewOutputTensorInt(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Not(X, O);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `Sign` math operation: f(x) = 1 if x > 0. f(x) = -1 if x &lt; 0. Otherwise f(x) = 0.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Sign(TensorFloat X)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Sign(X, O);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `Sign` math operation: f(x) = 1 if x > 0. f(x) = -1 if x &lt; 0. Otherwise f(x) = 0.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Sign(TensorInt X)
        {
            var O = m_Backend.NewOutputTensorInt(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Sign(X, O);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `IsNaN` logical operation: f(x) = 1 if x is NaN, otherwise f(x) = 0.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt IsNaN(TensorFloat X)
        {
            var O = m_Backend.NewOutputTensorInt(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.IsNaN(X, O);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `IsInf` logical operation: f(x) = 1 elementwise if x is +Inf and `detectPositive` is `true`, or x is -Inf and `detectNegative` is `true`. Otherwise f(x) = 0.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="detectNegative">Whether to detect negative infinities in the `IsInf` function.</param>
        /// <param name="detectPositive">Whether to detect positive infinities in the `IsInf` function.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt IsInf(TensorFloat X, bool detectNegative, bool detectPositive)
        {
            var O = m_Backend.NewOutputTensorInt(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.IsInf(X, O, detectNegative, detectPositive);
            return O;
        }

        /// <summary>
        /// Performs an element-wise `Where` logical operation: f(condition, a, b) = a if `condition` is `true`, otherwise f(condition, a, b) = b.
        /// </summary>
        /// <param name="C">The condition tensor.</param>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <typeparam name="T">The tensor type of the input and output tensors.</typeparam>
        /// <returns>The computed output tensor.</returns>
        public T Where<T>(TensorInt C, T A, T B) where T : Tensor
        {
            var O = m_Backend.NewOutputTensor(A.shape.Broadcast(B.shape.Broadcast(C.shape)), A.dataType) as T;
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Where(C, A, B, O);
            return O;
        }

        /// <summary>
        /// Calculates an output tensor by copying the data from the input tensor and using a given shape. The data from the input tensor is unchanged.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="shape">The shape of the output tensor.</param>
        /// <typeparam name="T">The tensor type of the input and output tensors.</typeparam>
        /// <returns>The computed output tensor.</returns>
        public T Reshape<T>(T X, TensorShape shape) where T : Tensor
        {
            var O = m_Backend.NewOutputTensor(shape, X.dataType) as T;
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Reshape(X, O);
            return O;
        }

        /// <summary>
        /// Calculates an output tensor by broadcasting the input tensor into a given shape.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="shape">The shape to broadcast the input shape together with to calculate the output tensor.</param>
        /// <typeparam name="T">The tensor type of the input and output tensors.</typeparam>
        /// <returns>The computed output tensor.</returns>
        public T Expand<T>(T X, TensorShape shape) where T : Tensor
        {
            var O = m_Backend.NewOutputTensor(X.shape.Broadcast(shape), X.dataType) as T;
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Expand(X, O);
            return O;
        }

        /// <summary>
        /// Calculates an output tensor by reversing the dimensions of the input tensor.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <typeparam name="T">The tensor type of the input and output tensors.</typeparam>
        /// <returns>The computed output tensor.</returns>
        public T Transpose<T>(T X) where T : Tensor
        {
            var O = m_Backend.NewOutputTensor(X.shape.Transpose(), X.dataType) as T;
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Transpose(X, O);
            return O;
        }

        /// <summary>
        /// Calculates an output tensor by permuting the axes and data of the input tensor according to the given permutations.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="permutations">The axes to sample the output tensor from in the input tensor.</param>
        /// <typeparam name="T">The tensor type of the input and output tensors.</typeparam>
        /// <returns>The computed output tensor.</returns>
        public T Transpose<T>(T X, int[] permutations) where T : Tensor
        {
            var O = m_Backend.NewOutputTensor(X.shape.Transpose(permutations), X.dataType) as T;
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Transpose(X, O, permutations);
            return O;
        }

        /// <summary>
        /// Calculates an output tensor by concatenating the input tensors along a given axis.
        /// </summary>
        /// <param name="tensors">The input tensors.</param>
        /// <param name="axis">The axis along which to concatenate the input tensors.</param>
        /// <returns>The computed output tensor.</returns>
        public Tensor Concat(Tensor[] tensors, int axis)
        {
            var O = m_Backend.NewOutputTensor(TensorShapeHelper.ConcatShape(tensors, axis), tensors[0].dataType);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Concat(tensors, O, axis);
            return O;
        }

        /// <summary>
        /// Calculates an output tensor by splitting the input tensor along a given axis between start and end.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axis">The axis along which to split the input tensor.</param>
        /// <param name="start">The inclusive start value for the split.</param>
        /// <param name="end">The exclusive end value for the split.</param>
        /// <typeparam name="T">The tensor type of the input and output tensors.</typeparam>
        /// <returns>The computed output tensor.</returns>
        public T Split<T>(T X, int axis, int start = 0, int end = int.MaxValue) where T : Tensor
        {
            var O = m_Backend.NewOutputTensor(X.shape.Split(axis, start, end), X.dataType) as T;
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Split(X, O, axis, start);
            return O;
        }

        /// <summary>
        /// Calculates an output tensor by slicing the input tensor along given axes with given starts, ends, and steps.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="starts">The start index along each axis.</param>
        /// <param name="ends">The end index along each axis.</param>
        /// <param name="axes">The axes along which to slice. If this is `null`, the layer slices all axes.</param>
        /// <param name="steps">The step values for slicing. If this is `null`, the layer uses step size 1 throughout.</param>
        /// <typeparam name="T">The tensor type of the input and output tensors.</typeparam>
        /// <returns>The computed output tensor.</returns>
        public T Slice<T>(T X, ReadOnlySpan<int> starts, ReadOnlySpan<int> ends, ReadOnlySpan<int> axes, ReadOnlySpan<int> steps) where T : Tensor
        {
            var O = m_Backend.NewOutputTensor(X.shape.Slice(starts, ends, axes, steps), X.dataType) as T;
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Slice(X, O, starts, axes, steps);
            return O;
        }

        /// <summary>
        /// Calculates an output tensor by repeating the input layer a given number of times along each axis.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="repeats">The number of times to tile the input tensor along each axis.</param>
        /// <typeparam name="T">The tensor type of the input and output tensors.</typeparam>
        /// <returns>The computed output tensor.</returns>
        public T Tile<T>(T X, ReadOnlySpan<int> repeats) where T : Tensor
        {
            var O = m_Backend.NewOutputTensor(X.shape.Tile(repeats), X.dataType) as T;
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Tile(X, O, repeats);
            return O;
        }

        /// <summary>
        /// Takes values from the input tensor indexed by the indices tensor along a given axis and concatenates them.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="indices">The indices tensor.</param>
        /// <param name="axis">The axis along which to gather.</param>
        /// <typeparam name="T">The tensor type of the input and output tensors.</typeparam>
        /// <returns>The computed output tensor.</returns>
        public T Gather<T>(T X, TensorInt indices, int axis) where T : Tensor
        {
            var O = m_Backend.NewOutputTensor(ShapeInference.Gather(X.shape, indices.shape, axis), X.dataType) as T;
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Gather(X, indices, O, axis);
            return O;
        }

        /// <summary>
        /// Takes values from the input tensor indexed by the indices tensor along a given axis and concatenates them.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="indices">The indices tensor.</param>
        /// <param name="axis">The axis along which to gather.</param>
        /// <typeparam name="T">The tensor type of the input and output tensors.</typeparam>
        /// <returns>The computed output tensor.</returns>
        public T GatherElements<T>(T X, TensorInt indices, int axis) where T : Tensor
        {
            var O = m_Backend.NewOutputTensor(indices.shape, X.dataType) as T;
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.GatherElements(X, indices, O, axis);
            return O;
        }

        /// <summary>
        /// Takes slices of values from the batched input tensor indexed by the `indices` tensor.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="indices">The indices tensor.</param>
        /// <param name="batchDims">The number of batch dimensions of the input tensor, the gather begins at the next dimension.</param>
        /// <typeparam name="T">The tensor type of the input and output tensors.</typeparam>
        /// <returns>The computed output tensor.</returns>
        public T GatherND<T>(T X, TensorInt indices, int batchDims) where T : Tensor
        {
            var O = m_Backend.NewOutputTensor(ShapeInference.GatherND(X.shape, indices.shape, batchDims), X.dataType) as T;
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.GatherND(X, indices, O, batchDims);
            return O;
        }

        /// <summary>
        /// Copies the input tensor and updates values at indexes specified by the `indices` tensor with values specified by the `updates` tensor along a given axis.
        ///
        /// `ScatterElements` updates the values depending on the reduction mode used.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="indices">The indices tensor.</param>
        /// <param name="updates">The updates tensor.</param>
        /// <param name="axis">The axis on which to perform the scatter.</param>
        /// <param name="reduction">The reduction mode used to update the values as a `ScatterReductionMode`.</param>
        /// <typeparam name="T">The tensor type of the input and output tensors.</typeparam>
        /// <returns>The computed output tensor.</returns>
        public T ScatterElements<T>(T X, TensorInt indices, T updates, int axis, Layers.ScatterReductionMode reduction) where T : Tensor
        {
            var O = m_Backend.NewOutputTensor(X.shape, X.dataType) as T;
            if (O.shape.HasZeroDims())
                return O;
            if (indices.shape.HasZeroDims())
                m_Backend.MemCopy(X, O);
            else
                m_Backend.ScatterElements(X, indices, updates, O, axis, reduction);
            return O;
        }

        /// <summary>
        /// Copies the input tensor and updates values at indexes specified by the `indices` tensor with values specified by the `updates` tensor.
        ///
        /// `ScatterND` updates the values depending on the reduction mode used.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="indices">The indices tensor.</param>
        /// <param name="updates">The updates tensor.</param>
        /// <param name="reduction">The reduction mode used to update the values as a `ScatterReductionMode`.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat ScatterND(TensorFloat X, TensorInt indices, TensorFloat updates, Layers.ScatterReductionMode reduction)
        {
            var O = m_Backend.NewOutputTensorFloat(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.ScatterND(X, indices, updates, O, reduction);
            return O;
        }

        /// <summary>
        /// Copies the input tensor and updates values at indexes specified by the `indices` tensor with values specified by the `updates` tensor.
        ///
        /// `ScatterND` updates the values depending on the reduction mode used.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="indices">The indices tensor.</param>
        /// <param name="updates">The updates tensor.</param>
        /// <param name="reduction">The reduction mode used to update the values as a `ScatterReductionMode`.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt ScatterND(TensorInt X, TensorInt indices, TensorInt updates, Layers.ScatterReductionMode reduction)
        {
            var O = m_Backend.NewOutputTensorInt(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.ScatterND(X, indices, updates, O, reduction);
            return O;
        }

        /// <summary>
        /// Calculates the top-K largest or smallest elements of an input tensor along a given axis.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="k">The number of elements to calculate.</param>
        /// <param name="axis">The axis along which to perform the top-K operation.</param>
        /// <param name="largest">Whether to calculate the top-K largest elements. If this is `false`, the layer calculates the top-K smallest elements.</param>
        /// <param name="sorted">Whether to return the elements in sorted order.</param>
        /// <returns>The computed output tensor.</returns>
        public Tensor[] TopK(TensorFloat X, int k, int axis, bool largest, bool sorted)
        {
            var outputShape = new TensorShape(X.shape);
            outputShape[axis] = k;

            var values = m_Backend.NewOutputTensorFloat(outputShape);
            var indices = m_Backend.NewOutputTensorInt(outputShape);
            if (!outputShape.HasZeroDims())
                m_Backend.TopK(X, values, indices, k, axis, largest);
            return new Tensor[] { values, indices };
        }

        /// <summary>
        /// Generates a tensor with a given shape filled with a given value.
        /// </summary>
        /// <param name="X">The input tensor shape.</param>
        /// <param name="value">The fill value.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat ConstantOfShape(TensorShape X, float value)
        {
            var O = m_Backend.NewOutputTensorFloat(X);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.MemSet(O, value);
            return O;
        }

        /// <summary>
        /// Generates a tensor with a given shape filled with a given value.
        /// </summary>
        /// <param name="X">The input tensor shape.</param>
        /// <param name="value">The fill value.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt ConstantOfShape(TensorShape X, int value)
        {
            var O = m_Backend.NewOutputTensorInt(X);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.MemSet(O, value);
            return O;
        }

        /// <summary>
        /// Creates a copy of a given input tensor with the same shape and values.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <typeparam name="T">The tensor type of the input and output tensors.</typeparam>
        /// <returns>The computed output tensor.</returns>
        public T Copy<T>(T X) where T : Tensor
        {
            var O = m_Backend.NewOutputTensor(X.shape, X.dataType) as T;
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.MemCopy(X, O);
            return O;
        }

        /// <summary>
        /// Computes the output tensor using an element-wise `Cast` function: f(x) = (float)x or f(x) = (int)x depending on the value of `toType`.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="toType">The data type to cast to as a `DataType`.</param>
        /// <returns>The computed output tensor.</returns>
        public Tensor Cast(Tensor X, DataType toType)
        {
            var O = m_Backend.NewOutputTensor(X.shape, toType);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Cast(X, O);
            return O;
        }

        /// <summary>
        /// Represents a `Multinomial` random layer. This generates an output tensor with values from a multinomial distribution according to the probabilities given by the input tensor.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="count">The number of times to sample the input.</param>
        /// <param name="seed">The optional seed to use for the random number generation. If this is `null` the layer generates a seed using `System.Random()`.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Multinomial(TensorFloat X, int count, float? seed)
        {
            var O = m_Backend.NewOutputTensorInt(ShapeInference.Multinomial(X.shape, count));

            ArrayTensorData.Pin(X);
            ArrayTensorData.Pin(O, clearOnInit: false);

            uint finalSeed = Random.GetOpSeed(seed);
            finalSeed = finalSeed == 0 ? 1 : finalSeed;
            var random = new Mathematics.Random(finalSeed);

            // Tensorflow Multinomial for reference
            // See: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/multinomial_op.cc
            for (int n = 0; n < X.shape[0]; ++n)
            {
                var maxLogP = Mathf.NegativeInfinity;
                for (int i = 0; i < X.shape[1]; ++i)
                    maxLogP = Mathf.Max(X[n, i], maxLogP);

                float sumOfProbabilities = 0f;
                for (int i = 0; i < X.shape[1]; ++i)
                    sumOfProbabilities += Mathf.Exp(X[n, i] - maxLogP); // NOTE: X contains log-probabilities

                for (int sample = 0; sample < count; ++sample)
                {
                    float p = random.NextFloat() * sumOfProbabilities;

                    int i = 0;
                    float cumulativeP = 0f;
                    while (i < X.shape[1] && p > cumulativeP)
                    {
                        cumulativeP += Mathf.Exp(X[n, i] - maxLogP);
                        i++;
                    }
                    Logger.AssertIsTrue(i > 0, "Multinomial.ValueError: need at least one cumulative sample {0}", i);
                    O[n, sample] = i - 1;
                }
            }

            return O;
        }
    }
}
