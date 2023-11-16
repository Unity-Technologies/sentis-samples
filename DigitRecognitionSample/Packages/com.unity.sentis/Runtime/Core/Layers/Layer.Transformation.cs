using System;
using UnityEngine;
using UnityEngine.Assertions;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Options for the padding values for `Pad`.
    /// </summary>
    public enum PadMode
    {
        /// <summary>
        /// Use a constant value for the padded data.
        /// </summary>
        Constant,
        /// <summary>
        /// Use the reflection of the values of the input tensor mirrored on the first and last values along the axis. The edge values appear once in the output tensor.
        /// </summary>
        Reflect,
        /// <summary>
        /// Use the edge values of the input tensor.
        /// </summary>
        Edge,
        /// <summary>
        /// Use the reflection of the values of the input tensor mirrored half a step outside the first and last values along the axis. The edge values appear twice in the output tensor.
        /// </summary>
        Symmetric,
    }

    /// <summary>
    /// Options for the scaling mode to use for `Resize`.
    /// </summary>
    public enum ScaleMode
    {
        /// <summary>
        /// Use the size tensor directly for the shape of the output tensor.
        /// </summary>
        Sizes,
        /// <summary>
        /// Use the scales tensor to multiply the shape of the input tensor to calculate the shape of the output tensor.
        /// </summary>
        Scales
    }

    /// <summary>
    /// Options for the interpolation mode to use for `Resize`.
    /// </summary>
    public enum InterpolationMode
    {
        /// <summary>
        /// Use the nearest element to the calculated coordinate. The exact behaviour depends on `nearestMode`.
        /// </summary>
        Nearest,
        /// <summary>
        /// Use a linear sampling of the surrounding elements to the calculated coordinate.
        /// </summary>
        Linear,
        /// <summary>
        /// Use a cubic sampling of the surrounding elements to the calculated coordinate.
        /// </summary>
        Cubic
    }

    /// <summary>
    /// Options for how to sample the nearest element in `Resize` when using `InterpolationMode.NearestMode`.
    /// </summary>
    public enum NearestMode
    {
        /// <summary>
        /// Use rounding to the nearest integer coordinate. If the fractional part equals 0.5 then round down.
        /// </summary>
        RoundPreferFloor,
        /// <summary>
        /// Use rounding to the nearest integer coordinate. If the fractional part equals 0.5 then round up.
        /// </summary>
        RoundPreferCeil,
        /// <summary>
        /// Use rounding down to the next integer coordinate less than or equal to the input coordinate.
        /// </summary>
        Floor,
        /// <summary>
        /// Use rounding up to the next integer coordinate greater than or equal to the input coordinate.
        /// </summary>
        Ceil
    }

    /// <summary>
    /// Options for how to transform between the coordinate in the output tensor and the coordinate in the input tensor in `Resize`.
    /// </summary>
    public enum CoordTransformMode
    {
        /// <summary>
        /// Use shifting by half a pixel before and after scaling.
        /// </summary>
        HalfPixel,
        /// <summary>
        /// Use shifting by half a pixel before and after scaling if the output length is greater than 1, otherwise use 0.
        /// </summary>
        PytorchHalfPixel,
        /// <summary>
        /// Use scaling by `length - 1` so that corner pixels align.
        /// </summary>
        AlignCorners,
        /// <summary>
        /// Use direct scaling of coordinates by the scaling factor.
        /// </summary>
        Asymmetric,
    }

    /// <summary>
    /// Options for which part of the input matrix to retain in `Trilu`.
    /// </summary>
    public enum TriluMode
    {
        /// <summary>
        /// Use retaining of the lower part of the input matrix.
        /// </summary>
        Lower = 0,
        /// <summary>
        /// Use retaining of the upper part of the input matrix.
        /// </summary>
        Upper = 1,
    }

    /// <summary>
    /// Options for the ordering of the elements in `DepthToSpace`.
    /// </summary>
    public enum DepthToSpaceMode
    {
        /// <summary>
        /// Use depth, column, row ordering where the data is arranged (by * blocksize * channels) + (bx * channels) + c.
        /// </summary>
        DepthColumnRow,
        /// <summary>
        /// Use column, row, depth ordering where the data is arranged (c * blocksize * blocksize) + (by * blocksize) + bx.
        /// </summary>
        ColumnRowDepth,
    }

    /// <summary>
    /// Represents an element-wise `Cast` layer: f(x) = (float)x or f(x) = (int)x depending on the value of `toType`.
    /// </summary>
    [Serializable]
    public class Cast : Layer
    {
        /// <summary>
        /// The data type to cast to as a `DataType`.
        /// </summary>
        public DataType toType;

        /// <summary>
        /// Initializes and returns an instance of `Cast` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="toType">The data type to cast to as a `DataType`.</param>
        public Cast(string name, string input, DataType toType)
        {
            this.name = name;
            inputs = new[] { input };
            this.toType = toType;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            return new PartialTensor(toType, inputTensors[0].shape);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensor(inputTensors[0].shape, toType);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.Cast(inputTensors[0], O);
            return O;
        }

        internal override string profilerTag => "Cast";
    }

    /// <summary>
    /// Represents an element-wise `CastLike` layer: f(x) = (float)x or f(x) = (int)x depending on the data type of the targetType tensor.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.NoDataDependencyInputs(1)]
    public class CastLike : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `CastLike` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="targetType">The name to use for the targetType tensor of the layer. The layer does not use the values of this tensor in the computation.</param>
        public CastLike(string name, string input, string targetType)
        {
            this.name = name;
            inputs = new[] { input, targetType };
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var toType = inputTensors[1].dataType;
            return new PartialTensor(toType, inputTensors[0].shape);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensor(inputTensors[0].shape, inputTensors[1].dataType);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.Cast(inputTensors[0], O);
            return O;
        }

        internal override string profilerTag => "CastLike";
    }

    /// <summary>
    /// Represents a `Concat` concatenation layer. The layer computes the output tensor by concatenating the input tensors along a given axis.
    /// </summary>
    [Serializable]
    public class Concat : Layer
    {
        /// <summary>
        /// The axis along which to concatenate the input tensors.
        /// </summary>
        public int axis;

        /// <summary>
        /// Initializes and returns an instance of `Concat` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="inputs">The names to use for the input tensors of the layer.</param>
        /// <param name="axis">The axis along which to concatenate the input tensors.</param>
        public Concat(string name, string[] inputs, int axis)
        {
            this.name = name;
            this.inputs = inputs;
            this.axis = axis;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            Logger.AssertIsTrue(inputTensors.Length > 0, "Concat.InputError: can't broadcast shapes array of size 0");
            var dataType = inputTensors[0].dataType;

            var rank = SymbolicTensorDim.Unknown;
            foreach (var tensorInput in inputTensors)
            {
                if (tensorInput.shape.hasRank)
                    rank = SymbolicTensorDim.MaxDefinedDim(rank, new SymbolicTensorDim(tensorInput.shape.rank));
            }

            if (rank.isUnknown)
                return new PartialTensor(dataType, SymbolicTensorShape.UnknownShape);

            foreach (var tensorInput in inputTensors)
            {
                tensorInput.shape.DeclareRank(rank.value);
            }

            var shapeOut = SymbolicTensorShape.UnknownOfRank(rank.value);
            var axisOut = shapeOut.Axis(axis);

            for (var i = 0; i < shapeOut.rank; i++)
            {
                if (i == axisOut)
                {
                    shapeOut[i] = SymbolicTensorDim.Zero;
                    foreach (var tensorInput in inputTensors)
                    {
                        shapeOut[i] += tensorInput.shape[i];
                    }
                }
                else
                {
                    shapeOut[i] = SymbolicTensorDim.Unknown;
                    foreach (var tensorInput in inputTensors)
                    {
                        shapeOut[i] = SymbolicTensorDim.MaxDefinedDim(shapeOut[i], tensorInput.shape[i]);
                    }
                }
            }

            var tensorOut = new PartialTensor(dataType, shapeOut);

            if (shapeOut.rank != 1 || !tensorOut.isPartiallyKnown)
                return tensorOut;

            var index = 0;
            foreach (var X in inputTensors)
            {
                for (var i = 0; i < X.length; i++)
                {
                    tensorOut[index++] = X[i];
                }
            }

            return tensorOut;
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensor(TensorShapeHelper.ConcatShape(inputTensors, axis), inputTensors[0].dataType);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.Concat(inputTensors, O, axis);
            return O;
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, axis: {axis}";
        }

        internal override string profilerTag => "Concat";
    }

    /// <summary>
    /// Represents a `DepthToSpace` layer. The layer computes the output tensor by permuting data from depth into blocks of spatial data.
    /// </summary>
    [Serializable]
    public class DepthToSpace : Layer
    {
        /// <summary>
        /// The size of the blocks to move the depth data into.
        /// </summary>
        public int blocksize;
        /// <summary>
        /// The ordering of the data in the output tensor as a `DepthToSpaceMode`.
        /// </summary>
        public DepthToSpaceMode mode;

        /// <summary>
        /// Initializes and returns an instance of `DepthToSpace` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="blocksize">The size of the blocks to move the depth data into.</param>
        /// <param name="mode">The ordering of the data in the output tensor as a `DepthToSpaceMode`.</param>
        public DepthToSpace(string name, string input, int blocksize, DepthToSpaceMode mode)
        {
            this.name = name;
            inputs = new[] { input };
            this.blocksize = blocksize;
            this.mode = mode;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var shapeX = inputTensors[0].shape;
            shapeX.DeclareRank(4);
            return new PartialTensor(inputTensors[0].dataType, new SymbolicTensorShape(shapeX[0], shapeX[1] / (blocksize * blocksize), shapeX[2] * blocksize, shapeX[3] * blocksize));
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorFloat(ShapeInference.DepthToSpace(inputTensors[0].shape, blocksize));
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.DepthToSpace(inputTensors[0] as TensorFloat, O, blocksize, mode);
            return O;
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, blocksize: {string.Join(", ", blocksize)}, mode: {mode}";
        }

        internal override string profilerTag => "DepthToSpace";
    }

    /// <summary>
    /// Represents an `Expand` layer. The layer computes the output tensor by broadcasting the input tensor into a given shape.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(1)]
    public class Expand : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `Expand` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="shape">The name to use for the 1D shape tensor of the layer.</param>
        public Expand(string name, string input, string shape)
        {
            this.name = name;
            this.inputs = new[] { input, shape };
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            return new PartialTensor(inputTensors[0].dataType, inputTensors[1].ToSymbolicTensorShape().Broadcast(inputTensors[0].shape));
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var shape = new TensorShape(inputTensors[1].ToReadOnlySpan<int>());
            var O = ctx.backend.NewOutputTensor(inputTensors[0].shape.Broadcast(shape), inputTensors[0].dataType);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.Expand(inputTensors[0], O);
            return O;
        }

        internal override string profilerTag => "Expand";
    }

    /// <summary>
    /// Represents a `Flatten` layer. The layer computes the output tensor by reshaping the input tensor into a 2D matrix according to the given axis.
    /// </summary>
    [Serializable]
    public class Flatten : Layer
    {
        /// <summary>
        /// The axis up to which to flatten the input dimensions into the first dimension of the output tensor.
        /// </summary>
        public int axis;

        /// <summary>
        /// Initializes and returns an instance of `Flatten` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="axis">The axis up to which to flatten the input dimensions into the first dimension of the output tensor. The default value is 1.</param>
        public Flatten(string name, string input, int axis = 1)
        {
            this.name = name;
            inputs = new[] { input };
            this.axis = axis;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var shapeX = inputTensors[0].shape;
            if (!shapeX.hasRank)
            {
                if (axis == 0)
                    return inputTensors[0].Reshape(new SymbolicTensorShape(SymbolicTensorDim.One, inputTensors[0].shape.Length()));
                return inputTensors[0].Reshape(SymbolicTensorShape.UnknownOfRank(2));
            }

            var axisX = axis >= 0 ? axis : shapeX.rank + axis;

            var shapeOut = SymbolicTensorShape.Ones(2);
            for (var i = 0; i < axisX; i++)
            {
                shapeOut[0] *= shapeX[i];
            }
            for (var i = axisX; i < shapeX.rank; i++)
            {
                shapeOut[1] *= shapeX[i];
            }

            return inputTensors[0].Reshape(shapeOut);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var shape = inputTensors[0].shape.Flatten(axis);
            return ctx.backend.ShallowReshape(inputTensors[0], shape, AllocScope.LayerOutput);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, axis: {axis}";
        }

        internal override string profilerTag => "Flatten";
    }

    /// <summary>
    /// Represents an `Identity` layer. The output tensor is a copy of the input tensor.
    /// </summary>
    [Serializable]
    public class Identity : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `Identity` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Identity(string name, string input)
        {
            this.name = name;
            inputs = new[] { input };
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            return inputTensors[0];
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            return ctx.backend.ShallowCopy(inputTensors[0], AllocScope.LayerOutput);
        }

        internal override string profilerTag => "Identity";
    }

    /// <summary>
    /// Represents a `Pad` layer. The layer calculates the output tensor by adding padding to the input tensor according to the given padding values and mode.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(1, 2)]
    public class Pad : Layer
    {
        /// <summary>
        /// The `PadMode` to use when padding.
        /// </summary>
        public PadMode padMode;

        /// <summary>
        /// Initializes and returns an instance of `Pad` layer without a constant value tensor.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="data">The name to use for the input tensor of the layer.</param>
        /// <param name="pads">The name to use for the 1D pad tensor of the layer.</param>
        /// <param name="mode">The `PadMode` to use when padding.</param>
        public Pad(string name, string data, string pads, PadMode mode)
        {
            this.name = name;
            inputs = new[] { data, pads };
            padMode = mode;
        }

        /// <summary>
        /// Initializes and returns an instance of `Pad` layer with a constant value tensor.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="data">The name to use for the input tensor of the layer.</param>
        /// <param name="pads">The name to use for the 1D pad tensor of the layer.</param>
        /// <param name="constantValue">The name to use for the scalar constant value tensor of the layer.</param>
        /// <param name="mode">The `PadMode` to use when padding.</param>
        public Pad(string name, string data, string pads, string constantValue, PadMode mode)
        {
            this.name = name;
            inputs = new[] { data, pads, constantValue };
            padMode = mode;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var pads = inputTensors[1];
            var shapeX = inputTensors[0].shape;
            var shapePads = pads.shape;
            if (shapePads.hasRank)
            {
                Logger.AssertIsTrue(shapePads.rank == 1, "Pad.ValueError: pads must be rank 1");
                if (shapePads[0].isValue)
                {
                    Logger.AssertIsTrue(shapePads[0].value % 2 == 0, "Pad.ValueError: length of pads must divide by 2");
                    shapeX.DeclareRank(shapePads[0].value / 2);
                }
            }

            if (!shapeX.hasRank)
                return new PartialTensor(inputTensors[0].dataType);

            var shapeOut = SymbolicTensorShape.UnknownOfRank(shapeX.rank);

            for (var i = 0; i < shapeOut.rank; i++)
            {
                var dimPad = pads[i] + pads[i + shapeOut.rank];
                shapeOut[i] = shapeX[i] + (SymbolicTensorDim)dimPad;
            }

            return new PartialTensor(inputTensors[0].dataType, shapeOut);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var pads = inputTensors[1].ToReadOnlySpan<int>();
            var constantValue = inputTensors.Length > 2 && inputTensors[2] != null ? inputTensors[2].ToReadOnlySpan<float>()[0] : 0f;
            if (padMode != Layers.PadMode.Constant)
                Assert.IsFalse(inputTensors[0].shape.HasZeroDims(), "ValueError: zero dimensions input for Pad operator is not supported");
            var O = ctx.backend.NewOutputTensorFloat(inputTensors[0].shape.Pad(pads));
            if (O.shape.HasZeroDims())
                return O;
            if (inputTensors[0].shape.HasZeroDims())
                ctx.backend.MemSet(O, constantValue);
            else
                ctx.backend.Pad(inputTensors[0] as TensorFloat, O, pads, padMode, constantValue);
            return O;
        }

        internal override string profilerTag => "Pad";
    }

    /// <summary>
    /// Represents a `Reshape` layer. The layer calculates the output tensor by copying the data from the input tensor and using a given shape. The data from the input tensor is unchanged.
    ///
    /// Only one of the elements of the shape can be -1. The layer infers the size of this dimension from the remaining dimensions and the length of the input tensor.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(1)]
    public class Reshape : Layer
    {
        /// <summary>
        /// Whether to handle zeros in the shape like numpy.
        ///
        /// If the shape has a dimension of size 0 and `allowZero` is `true`, the output tensor has a dimension of size zero in the same place.
        ///
        /// If the shape has a dimension of size 0 and if `allowZero` is `false`, the output tensor has the same dimension as the input tensor at this axis.
        /// </summary>
        public bool allowZero;

        /// <summary>
        /// Initializes and returns an instance of `Reshape` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="shape">The name to use for the 1D shape tensor of the layer.</param>
        /// <param name="allowZero">Whether to handle zeros in the shape like numpy.
        ///
        /// The default value is `false` and zero-sized dimensions in the shape take their value from the input tensor shape.</param>
        public Reshape(string name, string input, string shape, bool allowZero = false)
        {
            this.name = name;
            inputs = new[] { input, shape };
            this.allowZero = allowZero;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var X = inputTensors[0];
            var shape = inputTensors[1];
            var shapeX = X.shape;
            shape.shape.DeclareRank(1);

            if (!shape.isPartiallyKnown)
            {
                if (shape.shape[0].isValue)
                    return X.Reshape(SymbolicTensorShape.UnknownOfRank(shape.shape[0].value));

                return X.Reshape(SymbolicTensorShape.UnknownShape);
            }

            var shapeOut = SymbolicTensorShape.UnknownOfRank(shape.length);

            var containsMinusOne = false;

            for (var i = 0; i < shapeOut.rank; i++)
            {
                if (shape[i] == -1)
                    containsMinusOne = true;
            }

            for (var i = 0; i < shapeOut.rank; i++)
            {
                if (shape[i].isUnknown)
                    continue;

                var dim = (SymbolicTensorDim)shape[i];
                if (shape[i].isParam)
                {
                    if (allowZero || (shapeX.hasRank && i >= shapeX.rank) || shapeX[i] == dim)
                        shapeOut[i] = dim;
                    else if (containsMinusOne)
                    {
                        for (var j = 0; j < shapeX.rank; j++)
                        {
                            if (shapeX[j] == dim)
                            {
                                shapeOut[i] = dim;
                                break;
                            }
                        }
                    }
                    continue;
                }

                if (shape[i].intValue > 0)
                    shapeOut[i] = dim;
                else if (shape[i].intValue == 0)
                    shapeOut[i] = allowZero ? SymbolicTensorDim.Zero : shapeX[i];
            }

            return X.Reshape(shapeOut, !containsMinusOne);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var shape = inputTensors[0].shape.Reshape(inputTensors[1].ToReadOnlySpan<int>(), allowZero);
            return ctx.backend.ShallowReshape(inputTensors[0], shape, AllocScope.LayerOutput);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, allowZero: {allowZero}";
        }

        internal override string profilerTag => "Reshape";
    }

    /// <summary>
    /// Represents a `Resize` layer. The layer calculates the output tensor by resampling the input tensor along the spatial dimensions to a given shape.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(1)]
    public class Resize : Layer
    {
        /// <summary>
        /// The `ScaleMode` to use for the layer.
        /// </summary>
        public ScaleMode scaleMode;
        /// <summary>
        /// The `CoordTransformMode` to use for the layer.
        /// </summary>
        public CoordTransformMode coordTransformMode;
        /// <summary>
        /// The `InterpolationMode` to use for the layer.
        /// </summary>
        public InterpolationMode mode;
        /// <summary>
        /// The `NearestMode` to use for the layer when using `InterpolationMode.NearestMode`.
        /// </summary>
        public NearestMode nearestMode;

        /// <summary>
        /// Initializes and returns an instance of `Resize` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="scalesOrSizes">The name to use for the 1D scales or sizes tensor of the layer depending on the `scaleMode`.</param>
        /// <param name="scaleMode">The `ScaleMode` to use for the layer.</param>
        /// <param name="mode">The `InterpolationMode` to use for the layer.</param>
        /// <param name="coordTransformMode">The `CoordTransformMode` to use for the layer.</param>
        /// <param name="nearestMode">The `NearestMode` to use for the layer when using `InterpolationMode.NearestMode`.</param>
        public Resize(string name, string input, string scalesOrSizes, ScaleMode scaleMode, InterpolationMode mode, CoordTransformMode coordTransformMode, NearestMode nearestMode)
        {
            this.name = name;
            inputs = new[] { input, scalesOrSizes };
            this.scaleMode = scaleMode;
            this.coordTransformMode = coordTransformMode;
            this.mode = mode;
            this.nearestMode = nearestMode;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var dataType = inputTensors[0].dataType;
            var shapeX = inputTensors[0].shape;

            if (scaleMode == ScaleMode.Sizes)
            {
                PartialTensor sizes = inputTensors[1];
                sizes.shape.DeclareRank(1);
                shapeX.DeclareRank(sizes.shape[0]);

                if (sizes.isPartiallyKnown)
                    return new PartialTensor(dataType, sizes.ToSymbolicTensorShape());

                return new PartialTensor(dataType, SymbolicTensorShape.UnknownOfRankLike(shapeX));
            }

            var scales = inputTensors[1];
            scales.shape.DeclareRank(1);
            shapeX.DeclareRank(scales.shape[0]);

            if (!scales.isPartiallyKnown)
                return new PartialTensor(dataType, SymbolicTensorShape.UnknownOfRankLike(shapeX));

            shapeX.DeclareRank(scales.length);

            var shapeOut = SymbolicTensorShape.UnknownOfRank(shapeX.rank);

            for (var i = 0; i < shapeOut.rank; i++)
            {
                var dimX = shapeX[i];
                var scale = scales[i];
                if (!scale.isFloatValue)
                    shapeOut[i] = SymbolicTensorDim.Unknown;
                else if (scale.floatValue == 1f)
                    shapeOut[i] = dimX;
                else if (dimX.isValue)
                    shapeOut[i] = new SymbolicTensorDim(Mathf.RoundToInt(dimX.value * scale.floatValue));
                else
                    shapeOut[i] = SymbolicTensorDim.Unknown;
            }

            return new PartialTensor(dataType, shapeOut);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            if (scaleMode == ScaleMode.Sizes)
            {
                var inputShape = inputTensors[0].shape;
                Span<float> scales = stackalloc float[inputShape.rank];

                var sizes = inputTensors[1].ToReadOnlySpan<int>();

                for (var i = 0; i < scales.Length; i++)
                {
                    scales[i] = sizes[i] / (float)inputShape[i];
                }
                var O = ctx.backend.NewOutputTensorFloat(ShapeInference.Resize(inputTensors[0].shape, scales));
                if (O.shape.HasZeroDims())
                    return O;
                ctx.backend.Resize(inputTensors[0] as TensorFloat, O, scales, mode, nearestMode, coordTransformMode);
                return O;
            }
            else
            {
                var scales = inputTensors[1].ToReadOnlySpan<float>();
                var O = ctx.backend.NewOutputTensorFloat(ShapeInference.Resize(inputTensors[0].shape, scales));
                if (O.shape.HasZeroDims())
                    return O;
                ctx.backend.Resize(inputTensors[0] as TensorFloat, O, scales, mode, nearestMode, coordTransformMode);
                return O;
            }
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, coordTransformMode: {coordTransformMode}, nearestMode: {nearestMode}";
        }

        internal override string profilerTag => "Resize";
    }

    /// <summary>
    /// Represents a `Slice` layer. The layer calculates the output tensor by slicing the input tensor along given axes with given starts, ends and steps.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(1, 2, 3, 4)]
    public class Slice : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `Slice` layer with given starts and ends. The layer slices the first axes of the input with step 1.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="starts">The name to use for the 1D starts tensor of the layer.</param>
        /// <param name="ends">The name to use for the 1D ends tensor of the layer.</param>
        public Slice(string name, string input, string starts, string ends)
        {
            this.name = name;
            inputs = new[] { input, starts, ends };
        }

        /// <summary>
        /// Initializes and returns an instance of `Slice` layer with given starts, ends and axes. The layer uses step 1.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="starts">The name to use for the 1D starts tensor of the layer.</param>
        /// <param name="ends">The name to use for the 1D ends tensor of the layer.</param>
        /// <param name="axes">The name to use for the 1D axes tensor of the layer.</param>
        public Slice(string name, string input, string starts, string ends, string axes)
        {
            this.name = name;
            inputs = new[] { input, starts, ends, axes };
        }

        /// <summary>
        /// Initializes and returns an instance of `Slice` layer with given starts, ends, axes, and steps.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="starts">The name to use for the 1D starts tensor of the layer.</param>
        /// <param name="ends">The name to use for the 1D ends tensor of the layer.</param>
        /// <param name="axes">The name to use for the 1D axes tensor of the layer.</param>
        /// <param name="steps">The name to use for the 1D steps tensor of the layer.</param>
        public Slice(string name, string input, string starts, string ends, string axes, string steps)
        {
            this.name = name;
            inputs = new[] { input, starts, ends, axes, steps };
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var dataType = inputTensors[0].dataType;
            var shapeData = inputTensors[0].shape;
            if (!shapeData.hasRank)
                return new PartialTensor(dataType);

            var data = inputTensors[0];
            var starts = inputTensors[1];
            var ends = inputTensors[2];
            var axes = (inputTensors.Length > 3 ? inputTensors[3] : null) ?? PartialTensor.Range(0, shapeData.rank);
            var steps = (inputTensors.Length > 4 ? inputTensors[4] : null) ?? PartialTensor.Ones(starts.shape);

            if (data.isPartiallyKnown && data.shape.rank == 1 && starts[0].isIntValue && ends[0].isIntValue && steps[0].isIntValue)
            {
                var dim = data.shape[0].value;
                var start = starts[0].intValue;
                var end = ends[0].intValue;
                var step = steps[0].intValue;

                var clampAdjustDirection = step < 0 ? -1 : 0;

                start = start < 0 ? dim + start : start;
                start = Mathf.Clamp(start, 0, dim + clampAdjustDirection);

                end = end < 0 ? dim + end : end;
                end = Mathf.Clamp(end, clampAdjustDirection, dim);

                var length = (int)Math.Ceiling((end - start) / (double)step);
                length = Mathf.Max(length, 0);

                var tensorOut = new PartialTensor(dataType, new SymbolicTensorShape(length));

                for (var i = 0; i < length; i++)
                {
                    tensorOut[i] = data[start + i * step];
                }

                return tensorOut;
            }

            if (!axes.isPartiallyKnown)
                return new PartialTensor(dataType, SymbolicTensorShape.UnknownOfRank(shapeData.rank));

            var shapeOut = new SymbolicTensorShape(shapeData);

            for (var i = 0; i < axes.length; i++)
            {
                var axisElement = axes[i];
                if (!axisElement.isIntValue)
                {
                    shapeOut = SymbolicTensorShape.UnknownOfRank(shapeData.rank);
                    continue;
                }
                var axis = shapeOut.Axis(axisElement.intValue);
                shapeOut[axis] = shapeData[axis].Slice(starts[i], ends[i], steps[i]);
            }

            return new PartialTensor(dataType, shapeOut);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var starts = inputTensors[1].ToReadOnlySpan<int>();
            var ends = inputTensors[2].ToReadOnlySpan<int>();
            var axes = inputTensors.Length > 3 && inputTensors[3] != null ? inputTensors[3].ToReadOnlySpan<int>() : null;
            var steps = inputTensors.Length > 4 && inputTensors[4] != null ? inputTensors[4].ToReadOnlySpan<int>() : null;
            var O = ctx.backend.NewOutputTensor(inputTensors[0].shape.Slice(starts, ends, axes, steps), inputTensors[0].dataType);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.Slice(inputTensors[0], O, starts, axes, steps);
            return O;
        }

        internal override string profilerTag => "Slice";
    }

    /// <summary>
    /// Represents a `SpaceToDepth` layer. The layer computes the output tensor by permuting data from blocks of spatial data into depth.
    /// </summary>
    [Serializable]
    public class SpaceToDepth : Layer
    {
        /// <summary>
        /// The size of the spatial blocks to move into depth.
        /// </summary>
        public int blocksize;

        /// <summary>
        /// Initializes and returns an instance of `SpaceToDepth` layer with given starts, ends, axes, and steps.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="blocksize">The size of the spatial blocks to move into depth.</param>
        public SpaceToDepth(string name, string input, int blocksize)
        {
            this.name = name;
            inputs = new[] { input };
            this.blocksize = blocksize;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var shapeX = inputTensors[0].shape;
            shapeX.DeclareRank(4);
            return new PartialTensor(inputTensors[0].dataType, new SymbolicTensorShape(shapeX[0], shapeX[1] * (blocksize * blocksize), shapeX[2] / blocksize, shapeX[3] / blocksize));
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorFloat(ShapeInference.SpaceToDepth(inputTensors[0].shape, blocksize));
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.SpaceToDepth(inputTensors[0] as TensorFloat, O, blocksize);
            return O;
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, blocksize: {blocksize}";
        }

        internal override string profilerTag => "SpaceToDepth";
    }

    /// <summary>
    /// Represents a `Split` layer. The layer computes the output tensors by splitting the input tensor along a single given axis.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(1)]
    public class Split : Layer
    {
        /// <summary>
        /// The axis along which to split.
        /// </summary>
        public int axis;
        /// <summary>
        /// The number of outputs along which to split the input tensor if no split tensor is used.
        /// </summary>
        public int numOutputs;

        /// <summary>
        /// Initializes and returns an instance of `Split` layer where the input tensor is split equally.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="outputs">The names to use for all of the output tensors of the layer.</param>
        /// <param name="axis">The axis along which to split.</param>
        /// <param name="numOutputs">The number of outputs to split the input tensor into.</param>
        public Split(string name, string input, string[] outputs, int axis, int numOutputs)
        {
            this.name = name;
            inputs = new[] { input };
            Logger.AssertIsTrue(outputs.Length >= 1, "Split.InputError: output array must have length at least 1");
            this.outputs = outputs;
            this.axis = axis;
            this.numOutputs = numOutputs;
            Logger.AssertIsTrue(numOutputs >= outputs.Length, "Split.InputError: numOutputs must be at least the length of output array");
        }

        /// <summary>
        /// Initializes and returns an instance of `Split` layer where the input tensor is split according to the split tensor.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="split">The name to use for the 1D split tensor of the layer.</param>
        /// <param name="outputs">The names to use for all of the output tensors of the layer.</param>
        /// <param name="axis">The axis along which to split.</param>
        public Split(string name, string input, string split, string[] outputs, int axis)
        {
            this.name = name;
            inputs = new[] { input, split };
            Logger.AssertIsTrue(outputs.Length >= 1, "Split.InputError: output array must have length at least 1");
            this.outputs = outputs;
            this.axis = axis;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            PartialTensor partialSplit;
            if (inputs.Length == 2)
            {
                partialSplit = inputTensors[1];
            }
            else
            {
                partialSplit = new PartialTensor(DataType.Int, new SymbolicTensorShape(numOutputs));

                var dim = inputTensors[0].shape[axis];
                if (dim.isParam && numOutputs == 1)
                {
                    partialSplit[0] = new PartialTensorElement(dim.param);
                }
                else if (dim.isValue)
                {
                    var splitLength = Mathf.CeilToInt(dim.value / (float)numOutputs);
                    for (var i = 0; i < numOutputs - 1; i++)
                    {
                        partialSplit[i] = new PartialTensorElement(splitLength);
                    }

                    // final split length is the (possible smaller) remainder along the axis
                    var lastSplitLength = dim.value - (splitLength * (numOutputs - 1));
                    Logger.AssertIsTrue(lastSplitLength >= 0, "Split.InputError: split axis too small for numOutputs");
                    partialSplit[numOutputs - 1] = new PartialTensorElement(lastSplitLength);
                }
            }

            var shapeOut = new SymbolicTensorShape(inputTensors[0].shape);
            if (shapeOut.hasRank)
                shapeOut[axis] = (SymbolicTensorDim)partialSplit[0];
            for (var i = 1; i < outputs.Length; i++)
            {
                var outputShape = new SymbolicTensorShape(inputTensors[0].shape);
                outputShape[axis] = (SymbolicTensorDim)partialSplit[i];
                ctx.AddPartialTensor(outputs[i], new PartialTensor(inputTensors[0].dataType, outputShape));
            }

            return new PartialTensor(inputTensors[0].dataType, shapeOut);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            Tensor firstOutput = null;
            var dim = inputTensors[0].shape[axis];
            ReadOnlySpan<int> split = null;
            var equalSplitLength = 0;
            if (inputTensors.Length > 1 && inputTensors[1] != null)
                split = inputTensors[1].ToReadOnlySpan<int>();
            else
                equalSplitLength = (int)Math.Ceiling(dim / (double)numOutputs);
            var start = 0;
            for (var i = 0; i < outputs.Length; i++)
            {
                var end = start + (split != null ? split[i] : equalSplitLength);
                end = Math.Min(end, dim);
                var O = ctx.backend.NewOutputTensor(inputTensors[0].shape.Split(axis, start, end), inputTensors[0].dataType);
                if (!O.shape.HasZeroDims())
                    ctx.backend.Split(inputTensors[0], O, axis, start);
                if (i == 0)
                    firstOutput = O;
                else
                    ctx.vars.Store(outputs[i], O);
                start = end;
            }
            return firstOutput;
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, axis: {axis}, numOutputs: {numOutputs}";
        }

        internal override string profilerTag => "Split";
    }

    /// <summary>
    /// Represents a `Squeeze` layer. The layer computes the output tensor by reshaping the input tensor by removing dimensions of size 1.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(1)]
    public class Squeeze : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `Squeeze` layer where the layer squeezes all the axes of size 1 from the input.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Squeeze(string name, string input)
        {
            this.name = name;
            inputs = new[] { input };
        }

        /// <summary>
        /// Initializes and returns an instance of `Squeeze` layer where the layer squeezes the specified axes of size 1 from the input.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="axes">The name to use for the 1D axes tensor of the layer.</param>
        public Squeeze(string name, string input, string axes)
        {
            this.name = name;
            inputs = new[] { input, axes };
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            if (inputs.Length == 2)
            {
                var X = inputTensors[0];
                var axes = inputTensors[1];
                if (!axes.isPartiallyKnown)
                    return X.Reshape(SymbolicTensorShape.UnknownShape);
                if (!axes.IsFullyKnown())
                    return X.Reshape(SymbolicTensorShape.UnknownOfRank(X.shape.rank - axes.length));
                return X.Reshape(X.shape.Squeeze(axes));
            }

            return inputTensors[0].Reshape(inputTensors[0].shape.Squeeze());
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            TensorShape shape;
            if (inputTensors.Length > 1 && inputTensors[1] != null)
            {
                var axes = inputTensors[1].ToReadOnlySpan<int>();
                shape = inputTensors[0].shape.Squeeze(axes);
            }
            else
            {
                shape = inputTensors[0].shape.Squeeze();
            }
            return ctx.backend.ShallowReshape(inputTensors[0], shape, AllocScope.LayerOutput);
        }

        internal override string profilerTag => "Squeeze";
    }

    /// <summary>
    /// Represents a `Tile` layer. The layer computes the output tensor by repeating the input layer a given number of times along each axis.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(1)]
    public class Tile : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `Tile` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="repeats">The name to use for the 1D repeats tensor of the layer.</param>
        public Tile(string name, string input, string repeats)
        {
            this.name = name;
            inputs = new[] { input, repeats };
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var dataType = inputTensors[0].dataType;
            var shapeX = inputTensors[0].shape;
            var repeats = inputTensors[1];
            repeats.shape.DeclareRank(1);

            if (!repeats.isPartiallyKnown)
            {
                if (repeats.shape[0].isValue && !shapeX.hasRank)
                    shapeX = SymbolicTensorShape.UnknownOfRank(repeats.shape[0].value);
                Logger.AssertIsFalse(repeats.shape[0] != shapeX.rank, "Tile.InputError: repeats value must be equal to input rank");
                return new PartialTensor(dataType, SymbolicTensorShape.UnknownOfRankLike(shapeX));
            }

            shapeX.DeclareRank(repeats.length);

            var shapeOut = new SymbolicTensorShape(shapeX);
            for (var i = 0; i < shapeOut.rank; i++)
            {
                shapeOut[i] *= (SymbolicTensorDim)repeats[i];
            }
            return new PartialTensor(dataType, shapeOut);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var repeats = inputTensors[1].ToReadOnlySpan<int>();
            var O = ctx.backend.NewOutputTensor(inputTensors[0].shape.Tile(repeats), inputTensors[0].dataType);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.Tile(inputTensors[0], O, repeats);
            return O;
        }

        internal override string profilerTag => "Tile";
    }

    /// <summary>
    /// Represents a `Transpose` layer. The layer computes the output tensor by permuting the axes and data of the input tensor according to the given permutations.
    /// </summary>
    [Serializable]
    public class Transpose : Layer
    {
        /// <summary>
        /// The axes to sample the output tensor from in the input tensor.
        ///
        /// If this is `null`, the layer reverses the dimensions of the input tensor in the output tensor.
        /// </summary>
        public int[] permutations;

        /// <summary>
        /// Initializes and returns an instance of `Transpose` layer with permutations as an array of integers.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="permutations">The axes to sample the output tensor from in the input tensor.</param>
        public Transpose(string name, string input, int[] permutations)
        {
            this.name = name;
            inputs = new[] { input };
            this.permutations = permutations;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var shapeX = inputTensors[0].shape;
            if (permutations != null)
                shapeX.DeclareRank(permutations.Length);

            if (!shapeX.hasRank)
                return new PartialTensor(inputTensors[0].dataType);

            var shapeOut = SymbolicTensorShape.UnknownOfRank(shapeX.rank);

            if (permutations == null || permutations.Length == 0)
            {
                // reverse axes
                for (var i = 0; i < shapeX.rank; i++)
                {
                    shapeOut[i] = shapeX[shapeX.rank - 1 - i];
                }
            }
            else
            {
                uint axesBitMask = 0;
                for (var i = 0; i < permutations.Length; i++)
                {
                    var axis = shapeX.Axis(permutations[i]);
                    Logger.AssertIsTrue(((axesBitMask >> axis) & 1U) == 0, "Transpose.ValueError: permutation must be a permutation of the axis (0, rank-1)");
                    axesBitMask |= 1U << axis;
                    shapeOut[i] = shapeX[axis];
                }
            }

            return new PartialTensor(inputTensors[0].dataType, shapeOut);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            if (permutations == null)
            {
                var O = ctx.backend.NewOutputTensor(inputTensors[0].shape.Transpose(), inputTensors[0].dataType);
                if (O.shape.HasZeroDims())
                    return O;
                ctx.backend.Transpose(inputTensors[0], O);
                return O;
            }
            else
            {
                var O = ctx.backend.NewOutputTensor(inputTensors[0].shape.Transpose(permutations), inputTensors[0].dataType);
                if (O.shape.HasZeroDims())
                    return O;
                ctx.backend.Transpose(inputTensors[0], O, permutations);
                return O;
            }
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            if (permutations == null)
                return base.ToString();
            else
                return $"{base.ToString()}, permutations: [{string.Join(", ", permutations)}]";
        }

        internal override string profilerTag => "Transpose";
    }

    /// <summary>
    /// Represents a `Trilu` layer. The layer computes the output tensor by retaining the upper or lower triangular values from an input matrix or matrix batch and setting the other values to zero.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(1)]
    public class Trilu : Layer
    {
        /// <summary>
        /// The lower or upper mode for the operation.
        /// </summary>
        public TriluMode mode;

        /// <summary>
        /// Initializes and returns an instance of `Trilu` layer with no k offset value.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="mode">The lower or upper mode for the operation.</param>
        public Trilu(string name, string input, TriluMode mode)
        {
            this.name = name;
            inputs = new[] { input };
            this.mode = mode;
        }

        /// <summary>
        /// Initializes and returns an instance of `Trilu` layer with k offset value.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="k">The name to use for the scalar k offset tensor of the layer.</param>
        /// <param name="mode">The lower or upper mode for the operation.</param>
        public Trilu(string name, string input, string k, TriluMode mode)
        {
            this.name = name;
            inputs = new[] { input, k };
            this.mode = mode;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            return new PartialTensor(inputTensors[0].dataType, inputTensors[0].shape);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var k = inputTensors.Length > 1 && inputTensors[1] != null ? inputTensors[1].ToReadOnlySpan<int>()[0] : 0;
            var O = ctx.backend.NewOutputTensor(inputTensors[0].shape, inputTensors[0].dataType);
            if (O.shape.HasZeroDims())
                return O;
            if (mode == TriluMode.Upper)
                ctx.backend.Triu(inputTensors[0], O, k);
            else
                ctx.backend.Tril(inputTensors[0], O, k);
            return O;
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, mode: {mode}";
        }

        internal override string profilerTag => "Trilu";
    }

    /// <summary>
    /// Represents an `Unsqueeze` layer. The layer computes the output tensor by reshaping the input tensor by adding dimensions of size 1 at the given axes.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(1)]
    public class Unsqueeze : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `Unsqueeze` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="axes">The name to use for the 1D axes tensor of the layer.</param>
        public Unsqueeze(string name, string input, string axes)
        {
            this.name = name;
            inputs = new[] { input, axes };
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            return inputTensors[0].Reshape(inputTensors[0].shape.Unsqueeze(inputTensors[1]));
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var shape = inputTensors[0].shape.Unsqueeze(inputTensors[1].ToReadOnlySpan<int>());
            return ctx.backend.ShallowReshape(inputTensors[0], shape, AllocScope.LayerOutput);
        }

        internal override string profilerTag => "Unsqueeze";
    }
}
