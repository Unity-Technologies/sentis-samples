using System;
using Unity.Sentis.Layers;
using UnityEngine;

namespace Unity.Sentis
{
    static class ShapeInference
    {
        /// <summary>
        /// Calculates the output shape of a generalized 2D matrix multiplication operation with optional transposes.
        /// This does not support numpy-style broadcasting of shapes, however this method accepts tensor shapes with leading 1s before the final two dimensions.
        /// </summary>
        public static TensorShape Gemm(TensorShape X, TensorShape Y, bool transposeX, bool transposeY)
        {
            Logger.AssertAreEqual(1, X.Length(0, -2), "Gemm.ShapeError: incorrect shape for X, must be rank 2 or have leading dimensions of 1");
            Logger.AssertAreEqual(1, Y.Length(0, -2), "Gemm.ShapeError: incorrect shape for Y, must be rank 2 or have leading dimensions of 1");

            var shape = TensorShape.Ones(Mathf.Max(X.rank, Y.rank));
            shape[-2] = transposeX ? X[-1] : X[-2];
            shape[-1] = transposeY ? Y[-2] : Y[-1];
            return shape;
        }

        /// <summary>
        /// updates pad values so that output_shape[i] = ceil(input_shape[i] / strides[i])
        /// N.B: pad int[] needs to be previously allocated with 2*#of spatial dimensions
        /// </summary>
        public static void UpdatePadForPoolAutoPadding(TensorShape shape, int[] kernelShape, int[] strides, int[] pads, bool ceilMode, AutoPad autopad)
        {
            if (autopad == Layers.AutoPad.NotSet)
                return;

            int featureCount = strides.Length;

            Logger.AssertAreEqual(2 * featureCount, pads.Length, "ComputePoolAutoPadding.LengthError: incorrect length for pad, expecting {0}, got {1}", 2 * featureCount, pads.Length);

            for (var i = 0; i < featureCount; ++i)
            {
                if (autopad == Layers.AutoPad.Valid)
                {
                    pads[i] = 0;
                    pads[i + featureCount] = 0;
                    continue;
                }

                // Based on ONNX (AveragePool & MaxPool)
                //        https://github.com/onnx/onnx/blob/master/docs/Operators.md
                // and TensorFlow docs:
                //         https://www.tensorflow.org/api_guides/python/nn#Notes_on_SAME_Convolution_Padding
                int outputDim = (shape[2 + i] - 1) / strides[i] + 1;

                // C# automatically rounds down
                // https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/operators/arithmetic-operators
                int padAlongFeature = 0;
                if (ceilMode)
                    padAlongFeature = (outputDim - 1) * strides[i] + kernelShape[i] - strides[i] + 1 - shape[2 + i];
                else
                    padAlongFeature = (outputDim - 1) * strides[i] + kernelShape[i] - shape[2 + i];

                var featureSmall = padAlongFeature / 2;
                var featureLarge = padAlongFeature - featureSmall;

                if (autopad == Layers.AutoPad.SameUpper)
                {
                    pads[i] = featureSmall;
                    pads[i + featureCount] = featureLarge;
                }
                else
                {
                    pads[i] = featureLarge;
                    pads[i + featureCount] = featureSmall;
                }
            }
        }
        public static TensorShape ApplyPool(TensorShape shape, int[] pool, int[] stride, int[] pad, bool ceilMode = false)
        {
            int featureCount = stride.Length;
            Logger.AssertIsTrue(stride.Length != 0, "ApplyPool.LengthError: stride should not be an empty array");
            Logger.AssertIsTrue(stride.Length*2 == pad.Length, "ApplyPool.LengthError incorrect length match between strides: {0} and pad: {1}", stride.Length, pad.Length);
            Logger.AssertAreEqual(featureCount, (shape.rank - 2), "ApplyPool.RankError: stride length does not match input shape {0} feature count", shape);

            // Based on ONNX (AveragePool & MaxPool)
            //        https://github.com/onnx/onnx/blob/master/docs/Operators.md
            // Theano "Convolution arithmetic tutorial"
            //        http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html#quick-reference
            // and TensorFlow docs:
            //         https://www.tensorflow.org/api_guides/python/nn#Convolution
            //         https://www.tensorflow.org/api_guides/python/nn#Notes_on_SAME_Convolution_Padding
            //
            //   output_size = (input_size + pad_left + pad_right - kernel_size) / stride + 1
            var newShape = new TensorShape(shape);
            for (var i = 0; i < featureCount; ++i)
            {
                // C# automatically rounds down
                // https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/operators/arithmetic-operators
                if (ceilMode)
                    newShape[2 + i] = (shape[2 + i] + (pad[i] + pad[i + featureCount]) - pool[i] + stride[i] - 1) / stride[i] + 1;
                else
                    newShape[2 + i] = (shape[2 + i] + (pad[i] + pad[i + featureCount]) - pool[i]) / stride[i] + 1;
            }
            return newShape;
        }

        /// <summary>
        /// updates pad values so that output_shape[i] = ceil(input_shape[i] / strides[i])
        /// N.B: pad int[] needs to be previously allocated with 2*#of spatial dimensions
        /// </summary>
        public static void UpdatePadForConvAutoPadding(TensorShape shape, TensorShape kernel, Span<int> strides, Span<int> dilations, AutoPad autopad, Span<int> pads)
        {
            if (autopad == Layers.AutoPad.NotSet)
                return;

            var numSpatialDims = shape.rank - 2;
            Logger.AssertAreEqual(2 * numSpatialDims, pads.Length, "ComputeConvAutoPadding.LengthError: incorrect length for pad, expecting {0}, got {1}", 2 * numSpatialDims, pads.Length);
            Logger.AssertAreEqual(numSpatialDims, dilations.Length, "ComputeConvAutoPadding.LengthError: incorrect length for dilation, expecting {0}, got {1}", numSpatialDims, dilations.Length);

            for (var i = 0; i < numSpatialDims; ++i)
            {
                if (autopad == Layers.AutoPad.Valid)
                {
                    pads[i] = 0;
                    pads[i + numSpatialDims] = 0;
                    continue;
                }

                var padTotal = dilations[i] * (kernel[2 + i] - 1) - (shape[2 + i] - 1) % strides[i];
                var padSmall = padTotal / 2;
                var padLarge = padTotal - padSmall;

                pads[i] = autopad == Layers.AutoPad.SameUpper ? padSmall : padLarge;
                pads[i + numSpatialDims] = autopad == Layers.AutoPad.SameUpper ? padLarge : padSmall;
            }
        }

        static TensorShape ApplyConvKernel(TensorShape shape, TensorShape kernel, Span<int> strides, Span<int> pads, Span<int> dilations)
        {
            var featureCount = shape.rank - 2;
            var newShape = new TensorShape(shape);
            for (var i = 0; i < featureCount; ++i)
            {
                newShape[2 + i] = (shape[2 + i] + (pads[i] + pads[i + featureCount]) - dilations[i] * (kernel[2 + i] - 1) - 1) / strides[i] + 1;
            }
            newShape[1] = kernel[0];

            return newShape;
        }

        public static TensorShape Conv(TensorShape shape, TensorShape kernel, int groups, Span<int> stride, Span<int> pad, Span<int> dilation)
        {
            Logger.AssertIsTrue(shape.rank >= 3, "Conv.RankError: incorrect rank for input, expecting >= 3, got {0}", shape.rank);
            Logger.AssertIsTrue(kernel.rank >= 3, "Conv.RankError: incorrect rank for kernel, expecting >= 3, got {0}", kernel.rank);

            if (stride != null)
                Logger.AssertAreEqual(shape.rank - 2, stride.Length, "Conv.LengthError: incorrect length for stride, expecting rank-2, got {0}", stride.Length);
            if (pad != null)
                Logger.AssertAreEqual((shape.rank - 2) * 2, pad.Length, "Conv.LengthError: incorrect length for pad, expecting (rank-2)*2, got {0}", pad.Length);
            if (dilation != null)
                Logger.AssertAreEqual(shape.rank - 2, dilation.Length, "Conv.LengthError: incorrect length for dilation, expecting rank-2, got {0}", dilation.Length);

            Logger.AssertIsTrue(shape[1] % groups == 0, "Conv.ValueError: input channels {0} must be divisible by group count {1}", shape[1], groups);
            Logger.AssertIsTrue(kernel[0] % groups == 0, "Conv.ValueError: output features {0} must be divisible by group count {1}", kernel[0], groups);

            return ApplyConvKernel(shape, kernel, stride, pad, dilation);
        }

        public static void NonMaxSuppression(TensorShape boxes, TensorShape scores, float iouThreshold)
        {
            Logger.AssertIsTrue(boxes.rank == 3, "NonMaxSuppression.InputError: box data needs to be rank 3, got {0}", boxes.rank);
            Logger.AssertIsTrue(scores.rank == 3, "NonMaxSuppression.InputError: score data needs to be rank 3, got {0}", scores.rank);
            Logger.AssertIsTrue(boxes[2] == 4, "NonMaxSuppression.InputError: box data needs to have 4 values per box, got {0}", boxes[2]);
            Logger.AssertIsTrue(iouThreshold <= 1f, "NonMaxSuppression.InputError: iou threshold must be lower that 1, got {0}", iouThreshold);
            Logger.AssertIsTrue(iouThreshold >= 0f, "NonMaxSuppression.InputError: iou threshold must be higher that 0, got {0}", iouThreshold);
        }

        /// <summary>
        /// updates pad values so that output_shape[i] = input_shape[i] * strides[i]
        /// N.B: pad int[] needs to be previously allocated with 2*#of spatial dimensions
        /// </summary>
        public static void UpdatePadForConvTransAutoPadding(TensorShape shape, TensorShape kernel, Span<int> stride, AutoPad autopad, Span<int> outputPadding, Span<int> pads)
        {
            if (autopad == Layers.AutoPad.NotSet)
                return;

            var numSpatialDims = stride.Length;
            Logger.AssertAreEqual(2 * numSpatialDims, pads.Length, "ComputeConvTransAutoPadding.LengthError: incorrect length for pad, expecting {0}, got {1}", 2 * numSpatialDims, pads.Length);

            for (var i = 0; i < numSpatialDims; ++i)
            {
                if (autopad == Layers.AutoPad.Valid)
                {
                    pads[i] = 0;
                    pads[i + numSpatialDims] = 0;
                    continue;
                }

                var outputShape = shape[2 + i] * stride[i];
                var dilation = 1;
                var padAlongFeature = stride[i] * (shape[2 + i] - 1) + outputPadding[i] + (kernel[2 + i] - 1) * dilation + 1 - outputShape;

                var padSmall = padAlongFeature / 2;
                var padLarge = padAlongFeature - padSmall;

                pads[i] = autopad == Layers.AutoPad.SameUpper ? padSmall : padLarge;
                pads[i + numSpatialDims] = autopad == Layers.AutoPad.SameUpper ? padLarge : padSmall;
            }
        }

        public static TensorShape ApplyKernelInverse(TensorShape shape, TensorShape kernel, Span<int> stride, Span<int> pad, Span<int> outputAdjustment)
        {
            Logger.AssertIsTrue(stride.Length != 0, "ApplyKernelInverse.LengthError: stride should not be an empty array");
            Logger.AssertAreEqual(pad.Length, stride.Length * 2, "ApplyKernelInverse.LengthError incorrect length match between strides: {0} and pad: {1}", stride.Length, pad.Length);

            int featureCount = stride.Length;
            Logger.AssertAreEqual((shape.rank - 2), featureCount, "ApplyKernelInverse.LengthError incorrect length match between strides: {0} and shape: {1}", stride.Length, shape.rank);

            // Based on ONNX (ConvTranspose)
            //        https://github.com/onnx/onnx/blob/master/docs/Operators.md
            // and Theano "Convolution arithmetic tutorial"
            //        http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html#transposed-convolution-arithmetic
            //
            // Inverse of:
            //   output_size = (input_size + pad_left + pad_right - kernel_size) / stride + 1
            // Resulting in:
            //   output_size = (input_size - 1 ) * stride - (pad_left + pad_right) + kernel_size + output_adj
            //   output_adj = (input_size + (pad_left + pad_right) - kernel_size) % stride

            var newShape = new TensorShape(shape);
            for (var i = 0; i < stride.Length; ++i)
            {
                newShape[2 + i] = (shape[2 + i] - 1) * stride[i] - (pad[i] + pad[stride.Length + i]) + kernel[2 + i] + outputAdjustment[i];
            }

            newShape[1] = kernel[1];
            return newShape;
        }

        public static TensorShape ConvTranspose(TensorShape shapeX, TensorShape shapeW, Span<int> strides, Span<int> pads, Span<int> outputPadding)
        {
            Logger.AssertIsTrue(shapeX.rank >= 3, "ConvTranspose.RankError: incorrect rank for input, expecting >= 3, got {0}", shapeX.rank);
            Logger.AssertIsTrue(shapeW.rank >= 3, "ConvTranspose.RankError: incorrect rank for kernel, expecting >= 3, got {0}", shapeW.rank);

            Logger.AssertAreEqual(shapeX.rank - 2, strides.Length, "ConvTranspose.LengthError: incorrect length for stride, expecting rank-2, got {0}", strides.Length);
            Logger.AssertAreEqual((shapeX.rank - 2) * 2, pads.Length, "ConvTranspose.LengthError: incorrect length for pad, expecting (rank-2)*2, got {0}", pads.Length);

            return ApplyKernelInverse(shapeX, shapeW, strides, pads, outputPadding);
        }

        public static TensorShape Dense(TensorShape X, TensorShape W, TensorShape B)
        {
            Logger.AssertIsTrue(X.rank >= 2, "Dense.RankError: incorrect rank for input, expecting >= 2, got {0}", X.rank);
            Logger.AssertAreEqual(2, W.rank, "Dense.RankError: incorrect rank for input, expecting 2, got {0}", W.rank);
            Logger.AssertAreEqual(1, B.rank, "Dense.RankError: incorrect rank for input, expecting 1, got {0}", B.rank);

            Logger.AssertAreEqual(W[0], X[-1], "Dense.ValueError: incorrect shape match between X: {0}, and W {1}", X, W);
            Logger.AssertAreEqual(B[0], W[1], "Dense.ValueError: incorrect shape match between W: {0}, and B {1}", W, W);

            return X.MatMul(W);
        }

        public static TensorShape Squeeze(TensorShape shape, int[] axes)
        {
            return shape.Squeeze(axes);
        }

        public static TensorShape Unsqueeze(TensorShape shape, int[] axes)
        {
            return shape.Unsqueeze(axes);
        }

        public static TensorShape Concat(TensorShape[] shapes, int axis)
        {
            TensorShape output = shapes[0];
            axis = output.Axis(axis);

            for (int i = 1; i < shapes.Length; ++i)
            {
                if (shapes[i].HasZeroDims())
                    continue;

                #if (UNITY_ASSERTIONS)
                for (int r = 0; r < Math.Max(shapes[i].rank, shapes[0].rank); r++)
                {
                    if (r == axis)
                        continue;
                    Logger.AssertAreEqual(shapes[i][r], shapes[0][r], "ValueError: all input shapes for Concat must be equal {0}, {1} except on axis ({2}) dim", shapes[0], shapes[i], axis);
                }
                #endif
                output[axis] += shapes[i][axis];
            }

            return output;
        }

        public static TensorShape Resize(TensorShape shape, ReadOnlySpan<float> scale)
        {
            Logger.AssertIsTrue(!shape.HasZeroDims(), "Resize.InputError: zero dimensions are not supported");

            var scaled = new TensorShape(shape);
            for (int i = 0; i < scale.Length; i++)
            {
                scaled[i] = Mathf.FloorToInt(scaled[i] * scale[i]);
            }

            Logger.AssertIsTrue(!scaled.HasZeroDims(), "Resize.InputError: zero dimensions are not supported");
            return scaled;
        }

        public static TensorShape Resize(TensorShape shape, int axis, float scale)
        {
            Logger.AssertIsTrue(!shape.HasZeroDims(), "Resize.InputError: zero dimensions are not supported");

            var scaled = new TensorShape(shape);
            scaled[axis] = Mathf.FloorToInt(scaled[axis] * scale);

            Logger.AssertIsTrue(!scaled.HasZeroDims(), "Resize.InputError: zero dimensions are not supported");
            return scaled;
        }

        public static TensorShape DepthToSpace(TensorShape shape, int blocksize)
        {
            Logger.AssertAreEqual(4, shape.rank, "DepthToSpace.RankError: incorrect rank for input, expecting 4, got {0}", shape.rank);
            Logger.AssertAreEqual(0, shape[1] % (blocksize * blocksize), "DepthToSpace.ValueError: input channels {0}, must be modulus of blocksize {1} squared", shape[1], blocksize);

            return new TensorShape(shape[0], shape[1] / (blocksize * blocksize), shape[2] * blocksize, shape[3] * blocksize);
        }

        public static TensorShape SpaceToDepth(TensorShape shape, int blocksize)
        {
            Logger.AssertAreEqual(4, shape.rank, "SpaceToDepth.RankError: incorrect rank for input, expecting 4, got {0}", shape.rank);
            Logger.AssertAreEqual(0, shape[2] % blocksize, "SpaceToDepth.ValueError: input height {0}, must be modulus of blocksize {1}", shape[2], blocksize);
            Logger.AssertAreEqual(0, shape[3] % blocksize, "SpaceToDepth.ValueError: input width {0}, must be modulus of blocksize {1}", shape[3], blocksize);

            return new TensorShape(shape[0], shape[1] * (blocksize * blocksize), shape[2] / blocksize, shape[3] / blocksize);
        }

        public static TensorShape GlobalPool(TensorShape shape)
        {
            Logger.AssertIsTrue(shape.rank > 2, "GlobalPool2D.RankError: incorrect rank for input, expecting > 2, got {0}", shape.rank);

            TensorShape output = TensorShape.Ones(shape.rank);
            output[0] = shape[0];
            output[1] = shape[1];
            return output;
        }

        public static TensorShape GlobalAverageVariancePool(TensorShape shape)
        {
            Logger.AssertIsTrue(shape.rank > 2, "GlobalPool2D.RankError: incorrect rank for input, expecting > 2, got {0}", shape.rank);

            return new TensorShape(shape[0], shape[1], 2);
        }

        public static TensorShape OneHot(TensorShape shape, int axis, int depth)
        {
            TensorShape onehot = shape.Unsqueeze(axis);
            onehot[axis] = depth;

            return onehot;
        }

        public static TensorShape RoiAlign(TensorShape shape, TensorShape rois, TensorShape indices, int h, int w)
        {
            Logger.AssertAreEqual(4, shape.rank, "RoiAlign.RankError: incorrect rank for input, expecting 4, got {0}", shape.rank);
            Logger.AssertAreEqual(2, rois.rank, "RoiAlign.RankError: incorrect rank for rois, expecting 2, got {0}", rois.rank);
            Logger.AssertAreEqual(4, rois[1], "RoiAlign.ValueError: incorrect number of num_rois, expecting 4 got, {0}", rois[1]);
            Logger.AssertAreEqual(1, indices.rank, "RoiAlign.RankError: incorrect rank for indices, expecting 1, got {0}", indices.rank);
            Logger.AssertAreEqual(indices[0], rois[0], "RoiAlign.ValueError: mismatch number of num_rois between indices and rois", indices, rois);

            return new TensorShape(rois[0], shape[1], h, w);
        }

        public static TensorShape Compress(TensorShape shape, int numIndices, int axis)
        {
            var compressed = new TensorShape(shape);
            compressed[axis] = numIndices;
            return compressed;
        }

        public static TensorShape Gather(TensorShape shape, TensorShape indices, int axis)
        {
            TensorShape gathered = new TensorShape();
            gathered = gathered.BroadcastToRank(shape.rank - 1 + indices.rank);

            if (gathered.rank == 0)
                return gathered;

            axis = shape.Axis(axis);

            for (int i = 0; i < axis; i++)
                gathered[i] = shape[i];
            for (int i = 0; i < indices.rank; i++)
                gathered[axis + i] = indices[i];
            for (int i = axis + 1; i < shape.rank; i++)
                gathered[i + indices.rank - 1] = shape[i];

            return gathered;
        }

        public static TensorShape GatherND(TensorShape shape, TensorShape indices, int batchDims)
        {
            Logger.AssertIsTrue(shape.rank >= 1, "Input rank must at least be 1");
            Logger.AssertIsTrue(indices.rank >= 1, "Indices rank must at least be 1");

            for (int b = 0; b < batchDims; b++)
                Logger.AssertAreEqual(shape[b], indices[b], "The first batchDims dimensions of the shape of indices tensor and data tensor must be equal.");
            Logger.AssertIsTrue(batchDims < Math.Min(shape.rank, indices.rank), "batchDims must be smaller than the smaller rank of input tensors");

            var indexSize = indices[-1];
            Logger.AssertIsTrue(indexSize <= shape.rank - batchDims, "The indices[-1] should have a value between 1 (inclusive) and rank (input)-batchDims (inclusive)");

            var outputRank = shape.rank + indices.rank - indexSize - batchDims - 1;

            TensorShape gathered = new TensorShape();
            gathered = gathered.BroadcastToRank(outputRank);

            int j = 0;
            for (int i = 0; i < indices.rank - 1; i++)
                gathered[j++] = indices[i];
            for (int i = batchDims + indexSize; i < shape.rank; i++)
                gathered[j++] = shape[i];

            return gathered;
        }

        public static TensorShape Multinomial(TensorShape shape, int count)
        {
            Logger.AssertAreEqual(2, shape.rank, "Multinomial.RankError: incorrect rank for input, expecting 2, got {0}", shape.rank);

            return new TensorShape(shape[0], count);
        }

        public static TensorShape LSTM(TensorShape X, TensorShape W, TensorShape R, Layers.RnnLayout layout, out TensorShape Y, out TensorShape Y_h, out TensorShape Y_c)
        {
            Logger.AssertAreEqual(3, X.rank, "LSTM.RankError: incorrect rank for X, expecting 3, got {0}", X.rank);
            Logger.AssertAreEqual(3, W.rank, "LSTM.RankError: incorrect rank for W, expecting 3, got {0}", W.rank);
            Logger.AssertAreEqual(3, R.rank, "LSTM.RankError: incorrect rank for R, expecting 3, got {0}", R.rank);

            Logger.AssertAreEqual(W[0], R[0], "LSTM.ValueError: incorrect R shape {0}", R);
            Logger.AssertAreEqual(X[2], W[2], "LSTM.ValueError: incorrect W shape {0} vs X {1}", W, X);
            Logger.AssertAreEqual(W[1], R[1], "LSTM.ValueError: W[1] must match R[1], got: {0}, expected: {1}", W[1], R[1]);
            Logger.AssertAreEqual(R[1], 4 * R[2], "LSTM.ValueError: R[1] must match 4*R[2], got: {0}, expected {1}", R[1], R[2]);

            switch (layout)
            {
                case Layers.RnnLayout.SequenceFirst:
                    Y = new TensorShape(X[0], W[0], X[1], R[2]);
                    Y_h = new TensorShape(W[0], X[1], R[2]);
                    Y_c = new TensorShape(W[0], X[1], R[2]);
                    break;
                case Layers.RnnLayout.BatchFirst:
                    Y = new TensorShape(X[0], X[1], W[0], R[2]);
                    Y_h = new TensorShape(X[0], W[0], R[2]);
                    Y_c = new TensorShape(X[0], W[0], R[2]);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(layout), layout, null);
            }
            return Y;
        }

        public static TensorShape Range(float start, float limit, float delta)
        {
            return new TensorShape(Mathf.Max(Mathf.CeilToInt((limit - start) / delta), 0));
        }

        public static int SliceDim(int dim, int start, int end, int step)
        {
            int clampAdjustDirection = step < 0 ? -1 : 0;

            start = start < 0 ? dim + start : start;
            start = Mathf.Clamp(start, 0, dim + clampAdjustDirection);

            end = end < 0 ? dim + end : end;
            end = Mathf.Clamp(end, clampAdjustDirection, dim);

            int outputDim = (int)Math.Ceiling((double)(end - start) / (double)step);
            return Mathf.Max(outputDim, 0);
        }

        public static bool AreBroadcastable(TensorShape a, TensorShape b)
        {
            if (!((a.UnsafeGet(0) == b.UnsafeGet(0)) || (a.UnsafeGet(0) == 1) || (b.UnsafeGet(0) == 1) || (a.UnsafeGet(0) == 0 && b.UnsafeGet(0) != 0) || (b.UnsafeGet(0) == 0 && a.UnsafeGet(0) != 0)))
                return false;
            if (!((a.UnsafeGet(1) == b.UnsafeGet(1)) || (a.UnsafeGet(1) == 1) || (b.UnsafeGet(1) == 1) || (a.UnsafeGet(1) == 0 && b.UnsafeGet(1) != 0) || (b.UnsafeGet(1) == 0 && a.UnsafeGet(1) != 0)))
                return false;
            if (!((a.UnsafeGet(2) == b.UnsafeGet(2)) || (a.UnsafeGet(2) == 1) || (b.UnsafeGet(2) == 1) || (a.UnsafeGet(2) == 0 && b.UnsafeGet(2) != 0) || (b.UnsafeGet(2) == 0 && a.UnsafeGet(2) != 0)))
                return false;
            if (!((a.UnsafeGet(3) == b.UnsafeGet(3)) || (a.UnsafeGet(3) == 1) || (b.UnsafeGet(3) == 1) || (a.UnsafeGet(3) == 0 && b.UnsafeGet(3) != 0) || (b.UnsafeGet(3) == 0 && a.UnsafeGet(3) != 0)))
                return false;
            if (!((a.UnsafeGet(4) == b.UnsafeGet(4)) || (a.UnsafeGet(4) == 1) || (b.UnsafeGet(4) == 1) || (a.UnsafeGet(4) == 0 && b.UnsafeGet(4) != 0) || (b.UnsafeGet(4) == 0 && a.UnsafeGet(4) != 0)))
                return false;
            if (!((a.UnsafeGet(5) == b.UnsafeGet(5)) || (a.UnsafeGet(5) == 1) || (b.UnsafeGet(5) == 1) || (a.UnsafeGet(5) == 0 && b.UnsafeGet(5) != 0) || (b.UnsafeGet(5) == 0 && a.UnsafeGet(5) != 0)))
                return false;
            if (!((a.UnsafeGet(6) == b.UnsafeGet(6)) || (a.UnsafeGet(6) == 1) || (b.UnsafeGet(6) == 1) || (a.UnsafeGet(6) == 0 && b.UnsafeGet(6) != 0) || (b.UnsafeGet(6) == 0 && a.UnsafeGet(6) != 0)))
                return false;
            if (!((a.UnsafeGet(7) == b.UnsafeGet(7)) || (a.UnsafeGet(7) == 1) || (b.UnsafeGet(7) == 1) || (a.UnsafeGet(7) == 0 && b.UnsafeGet(7) != 0) || (b.UnsafeGet(7) == 0 && a.UnsafeGet(7) != 0)))
                return false;

            return true;
        }
    }
}
