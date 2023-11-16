using System;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents the abstract base class for reduction layers.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(1)]
    public abstract class Reduce : Layer
    {
        /// <summary>
        /// Whether to keep the axis dimension in the output tensor.
        /// </summary>
        public bool keepdims;
        /// <summary>
        /// Whether to perform an identity operation if the input axes tensor is empty.
        ///
        /// If this is `false` and the input axes tensor is empty then the reduction is applied on all axes of the input tensor.
        /// </summary>
        public bool noopWithEmptyAxes;

        /// <summary>
        /// Initializes and returns an instance of `Reduce` reduction layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="inputs">The name of the input tensor and optionally the axes tensor. If no axes tensor is provided, the layer performs the reduction according to the value of `noopWithEmptyAxes`.</param>
        /// <param name="keepdims">Whether to keep the axis dimension in the output tensor. The default value is `true`.</param>
        /// <param name="noopWithEmptyAxes">Whether to perform an identity operation if the input axes tensor is empty. The default value is `false`.</param>
        protected Reduce(string name, string[] inputs, bool keepdims = true, bool noopWithEmptyAxes = false)
        {
            this.name = name;
            this.inputs = inputs;
            this.keepdims = keepdims;
            this.noopWithEmptyAxes = noopWithEmptyAxes;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var dataType = inputTensors[0].dataType;
            var shapeX = inputTensors[0].shape;
            var axes = inputTensors.Length == 2 ? inputTensors[1] : null;
            var shapeAxes = axes?.shape ?? new SymbolicTensorShape(SymbolicTensorDim.Zero);
            if (axes != null && axes.isPartiallyKnown && axes.length != 0)
            {
                var reducedShape = new SymbolicTensorShape(shapeX);
                if (!axes.IsFullyKnown() && reducedShape.hasRank)
                {
                    // replace any non 0 or 1 dims with unknown (0 and 1 stay the same whether reduced or not)
                    for (var i = 0; i < reducedShape.rank; i++)
                    {
                        if (reducedShape[i] == 0 || reducedShape[i] == 1)
                            continue;
                        reducedShape[i] = SymbolicTensorDim.Unknown;
                    }
                }

                for (var i = 0; i < axes.length; i++)
                {
                    if (!axes[i].isIntValue)
                        continue;
                    var axis = axes[i].intValue;
                    // reducing on a zero axis will result in a zero rather than a one
                    if (shapeX[axis].isValue)
                        reducedShape[axis] = shapeX[axis].value == 0 ? SymbolicTensorDim.Zero : SymbolicTensorDim.One;
                    else
                        reducedShape[axis] = SymbolicTensorDim.Unknown;
                }

                var tensorOut = new PartialTensor(dataType, reducedShape);
                if (!keepdims)
                {
                    tensorOut = tensorOut.Reshape(!axes.IsFullyKnown() ? SymbolicTensorShape.UnknownOfRank(tensorOut.shape.rank - axes.length) : tensorOut.shape.Squeeze(axes));
                }

                return tensorOut;
            }

            if (shapeAxes.IsFullyKnown())
            {
                // empty axes
                if (shapeAxes[0].value == 0)
                {
                    if (noopWithEmptyAxes)
                        return new PartialTensor(dataType, shapeX);

                    return new PartialTensor(dataType, keepdims ? SymbolicTensorShape.OnesLike(shapeX) : new SymbolicTensorShape());
                }

                return new PartialTensor(dataType, keepdims ? SymbolicTensorShape.UnknownOfRankLike(shapeX) : SymbolicTensorShape.UnknownShape);
            }

            return new PartialTensor(dataType, keepdims && !noopWithEmptyAxes ? SymbolicTensorShape.UnknownOfRankLike(shapeX) : SymbolicTensorShape.UnknownShape);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, keepdims: {keepdims}, noopWithEmptyAxes: {noopWithEmptyAxes}";
        }
    }

    /// <summary>
    /// Represents a `ReduceL1` reduction layer along the given axes: f(x1, x2 ... xn) = |x1| + |x2| + ... + |xn|.
    /// </summary>
    [Serializable]
    public class ReduceL1 : Reduce
    {
        /// <summary>
        /// Initializes and returns an instance of `ReduceL1` reduction layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="inputs">The name of the input tensor and optionally the axes tensor. If no axes tensor is provided, the layer performs the reduction according to the value of `noopWithEmptyAxes`.</param>
        /// <param name="keepdims">Whether to keep the axis dimension in the output tensor. The default value is `true`.</param>
        /// <param name="noopWithEmptyAxes">Whether to perform an identity operation if the input axes tensor is empty. The default value is `false`.</param>
        public ReduceL1(string name, string[] inputs, bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(name, inputs, keepdims, noopWithEmptyAxes) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var axes = inputTensors.Length > 1 && inputTensors[1] != null ? inputTensors[1].ToReadOnlySpan<int>() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
                return inputTensors[0].ShallowCopy();
            var O = ctx.backend.NewOutputTensor(inputTensors[0].shape.Reduce(axes, keepdims), inputTensors[0].dataType);
            if (O.shape.HasZeroDims())
                return O;
            if (inputTensors[0] is TensorInt)
                ctx.backend.ReduceL1(inputTensors[0] as TensorInt, O as TensorInt, axes, keepdims);
            else
                ctx.backend.ReduceL1(inputTensors[0] as TensorFloat, O as TensorFloat, axes, keepdims);
            return O;
        }

        internal override string profilerTag => "ReduceL1";
    }

    /// <summary>
    /// Represents a `ReduceL2` reduction layer along the given axes: f(x1, x2 ... xn) = sqrt(x1² + x2² + ... + xn²).
    /// </summary>
    [Serializable]
    public class ReduceL2 : Reduce
    {
        /// <summary>
        /// Initializes and returns an instance of `ReduceL2` reduction layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="inputs">The name of the input tensor and optionally the axes tensor. If no axes tensor is provided, the layer performs the reduction according to the value of `noopWithEmptyAxes`.</param>
        /// <param name="keepdims">Whether to keep the axis dimension in the output tensor. The default value is `true`.</param>
        /// <param name="noopWithEmptyAxes">Whether to perform an identity operation if the input axes tensor is empty. The default value is `false`.</param>
        public ReduceL2(string name, string[] inputs, bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(name, inputs, keepdims, noopWithEmptyAxes) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var axes = inputTensors.Length > 1 && inputTensors[1] != null ? inputTensors[1].ToReadOnlySpan<int>() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
                return inputTensors[0].ShallowCopy();
            var O = ctx.backend.NewOutputTensor(inputTensors[0].shape.Reduce(axes, keepdims), inputTensors[0].dataType);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.ReduceL2(inputTensors[0] as TensorFloat, O as TensorFloat, axes, keepdims);
            return O;
        }

        internal override string profilerTag => "ReduceL2";
    }

    /// <summary>
    /// Represents a `ReduceLogSum` reduction layer along the given axes: f(x1, x2 ... xn) = log(x1 + x2 + ... + xn).
    /// </summary>
    [Serializable]
    public class ReduceLogSum : Reduce
    {
        /// <summary>
        /// Initializes and returns an instance of `ReduceLogSum` reduction layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="inputs">The name of the input tensor and optionally the axes tensor. If no axes tensor is provided, the layer performs the reduction according to the value of `noopWithEmptyAxes`.</param>
        /// <param name="keepdims">Whether to keep the axis dimension in the output tensor. The default value is `true`.</param>
        /// <param name="noopWithEmptyAxes">Whether to perform an identity operation if the input axes tensor is empty. The default value is `false`.</param>
        public ReduceLogSum(string name, string[] inputs, bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(name, inputs, keepdims, noopWithEmptyAxes) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var axes = inputTensors.Length > 1 && inputTensors[1] != null ? inputTensors[1].ToReadOnlySpan<int>() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
                return inputTensors[0].ShallowCopy();
            var O = ctx.backend.NewOutputTensor(inputTensors[0].shape.Reduce(axes, keepdims), inputTensors[0].dataType);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.ReduceLogSum(inputTensors[0] as TensorFloat, O as TensorFloat, axes, keepdims);
            return O;
        }

        internal override string profilerTag => "ReduceLogSum";
    }

    /// <summary>
    /// Represents a `ReduceLogSumExp` reduction layer along the given axes: f(x1, x2 ... xn) = log(e^x1 + e^x2 + ... + e^xn).
    /// </summary>
    [Serializable]
    public class ReduceLogSumExp : Reduce
    {
        /// <summary>
        /// Initializes and returns an instance of `ReduceLogSumExp` reduction layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="inputs">The name of the input tensor and optionally the axes tensor. If no axes tensor is provided, the layer performs the reduction according to the value of `noopWithEmptyAxes`.</param>
        /// <param name="keepdims">Whether to keep the axis dimension in the output tensor. The default value is `true`.</param>
        /// <param name="noopWithEmptyAxes">Whether to perform an identity operation if the input axes tensor is empty. The default value is `false`.</param>
        public ReduceLogSumExp(string name, string[] inputs, bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(name, inputs, keepdims, noopWithEmptyAxes) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var axes = inputTensors.Length > 1 && inputTensors[1] != null ? inputTensors[1].ToReadOnlySpan<int>() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
                return inputTensors[0].ShallowCopy();
            var O = ctx.backend.NewOutputTensor(inputTensors[0].shape.Reduce(axes, keepdims), inputTensors[0].dataType);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.ReduceLogSumExp(inputTensors[0] as TensorFloat, O as TensorFloat, axes, keepdims);
            return O;
        }

        internal override string profilerTag => "ReduceLogSumExp";
    }

    /// <summary>
    /// Represents a `ReduceMax` reduction layer along the given axes: f(x1, x2 ... xn) = max(x1, x2, ... , xn).
    /// </summary>
    [Serializable]
    public class ReduceMax : Reduce
    {
        /// <summary>
        /// Initializes and returns an instance of `ReduceMax` reduction layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="inputs">The name of the input tensor and optionally the axes tensor. If no axes tensor is provided, the layer performs the reduction according to the value of `noopWithEmptyAxes`.</param>
        /// <param name="keepdims">Whether to keep the axis dimension in the output tensor. The default value is `true`.</param>
        /// <param name="noopWithEmptyAxes">Whether to perform an identity operation if the input axes tensor is empty. The default value is `false`.</param>
        public ReduceMax(string name, string[] inputs, bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(name, inputs, keepdims, noopWithEmptyAxes) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var axes = inputTensors.Length > 1 && inputTensors[1] != null ? inputTensors[1].ToReadOnlySpan<int>() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
                return inputTensors[0].ShallowCopy();
            var O = ctx.backend.NewOutputTensor(inputTensors[0].shape.Reduce(axes, keepdims), inputTensors[0].dataType);
            if (O.shape.HasZeroDims())
                return O;
            if (inputTensors[0] is TensorInt)
                ctx.backend.ReduceMax(inputTensors[0] as TensorInt, O as TensorInt, axes, keepdims);
            else
                ctx.backend.ReduceMax(inputTensors[0] as TensorFloat, O as TensorFloat, axes, keepdims);
            return O;
        }

        internal override string profilerTag => "ReduceMax";
    }

    /// <summary>
    /// Represents a `ReduceMean` reduction layer along the given axes: f(x1, x2 ... xn) = (x1 + x2 + ... + xn) / n.
    /// </summary>
    [Serializable]
    public class ReduceMean : Reduce
    {
        /// <summary>
        /// Initializes and returns an instance of `ReduceMean` reduction layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="inputs">The name of the input tensor and optionally the axes tensor. If no axes tensor is provided, the layer performs the reduction according to the value of `noopWithEmptyAxes`.</param>
        /// <param name="keepdims">Whether to keep the axis dimension in the output tensor. The default value is `true`.</param>
        /// <param name="noopWithEmptyAxes">Whether to perform an identity operation if the input axes tensor is empty. The default value is `false`.</param>
        public ReduceMean(string name, string[] inputs, bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(name, inputs, keepdims, noopWithEmptyAxes) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var axes = inputTensors.Length > 1 && inputTensors[1] != null ? inputTensors[1].ToReadOnlySpan<int>() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
                return inputTensors[0].ShallowCopy();
            var O = ctx.backend.NewOutputTensor(inputTensors[0].shape.Reduce(axes, keepdims), inputTensors[0].dataType);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.ReduceMean(inputTensors[0] as TensorFloat, O as TensorFloat, axes, keepdims);
            return O;
        }

        internal override string profilerTag => "ReduceMean";
    }

    /// <summary>
    /// Represents a `ReduceMin` reduction layer along the given axes: f(x1, x2 ... xn) = min(x1, x2, ... , xn).
    /// </summary>
    [Serializable]
    public class ReduceMin : Reduce
    {
        /// <summary>
        /// Initializes and returns an instance of `ReduceMin` reduction layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="inputs">The name of the input tensor and optionally the axes tensor. If no axes tensor is provided, the layer performs the reduction according to the value of `noopWithEmptyAxes`.</param>
        /// <param name="keepdims">Whether to keep the axis dimension in the output tensor. The default value is `true`.</param>
        /// <param name="noopWithEmptyAxes">Whether to perform an identity operation if the input axes tensor is empty. The default value is `false`.</param>
        public ReduceMin(string name, string[] inputs, bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(name, inputs, keepdims, noopWithEmptyAxes) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var axes = inputTensors.Length > 1 && inputTensors[1] != null ? inputTensors[1].ToReadOnlySpan<int>() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
                return inputTensors[0].ShallowCopy();
            var O = ctx.backend.NewOutputTensor(inputTensors[0].shape.Reduce(axes, keepdims), inputTensors[0].dataType);
            if (O.shape.HasZeroDims())
                return O;
            if (inputTensors[0] is TensorInt)
                ctx.backend.ReduceMin(inputTensors[0] as TensorInt, O as TensorInt, axes, keepdims);
            else
                ctx.backend.ReduceMin(inputTensors[0] as TensorFloat, O as TensorFloat, axes, keepdims);
            return O;
        }

        internal override string profilerTag => "ReduceMin";
    }

    /// <summary>
    /// Represents a `ReduceProd` reduction layer along the given axes: f(x1, x2 ... xn) = x1 * x2 * ... * xn.
    /// </summary>
    [Serializable]
    public class ReduceProd : Reduce
    {
        /// <summary>
        /// Initializes and returns an instance of `ReduceProd` reduction layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="inputs">The name of the input tensor and optionally the axes tensor. If no axes tensor is provided, the layer performs the reduction according to the value of `noopWithEmptyAxes`.</param>
        /// <param name="keepdims">Whether to keep the axis dimension in the output tensor. The default value is `true`.</param>
        /// <param name="noopWithEmptyAxes">Whether to perform an identity operation if the input axes tensor is empty. The default value is `false`.</param>
        public ReduceProd(string name, string[] inputs, bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(name, inputs, keepdims, noopWithEmptyAxes) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var axes = inputTensors.Length > 1 && inputTensors[1] != null ? inputTensors[1].ToReadOnlySpan<int>() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
                return inputTensors[0].ShallowCopy();
            var O = ctx.backend.NewOutputTensor(inputTensors[0].shape.Reduce(axes, keepdims), inputTensors[0].dataType);
            if (O.shape.HasZeroDims())
                return O;
            if (inputTensors[0] is TensorInt)
                ctx.backend.ReduceProd(inputTensors[0] as TensorInt, O as TensorInt, axes, keepdims);
            else
                ctx.backend.ReduceProd(inputTensors[0] as TensorFloat, O as TensorFloat, axes, keepdims);
            return O;
        }

        internal override string profilerTag => "ReduceProd";
    }

    /// <summary>
    /// Represents a `ReduceSum` reduction layer along the given axes: f(x1, x2 ... xn) = x1 + x2 + ... + xn.
    /// </summary>
    [Serializable]
    public class ReduceSum : Reduce
    {
        /// <summary>
        /// Initializes and returns an instance of `ReduceSum` reduction layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="inputs">The name of the input tensor and optionally the axes tensor. If no axes tensor is provided, the layer performs the reduction according to the value of `noopWithEmptyAxes`.</param>
        /// <param name="keepdims">Whether to keep the axis dimension in the output tensor. The default value is `true`.</param>
        /// <param name="noopWithEmptyAxes">Whether to perform an identity operation if the input axes tensor is empty. The default value is `false`.</param>
        public ReduceSum(string name, string[] inputs, bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(name, inputs, keepdims, noopWithEmptyAxes) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var axes = inputTensors.Length > 1 && inputTensors[1] != null ? inputTensors[1].ToReadOnlySpan<int>() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
                return inputTensors[0].ShallowCopy();
            var O = ctx.backend.NewOutputTensor(inputTensors[0].shape.Reduce(axes, keepdims), inputTensors[0].dataType);
            if (O.shape.HasZeroDims())
                return O;
            if (inputTensors[0] is TensorInt)
                ctx.backend.ReduceSum(inputTensors[0] as TensorInt, O as TensorInt, axes, keepdims);
            else
                ctx.backend.ReduceSum(inputTensors[0] as TensorFloat, O as TensorFloat, axes, keepdims);
            return O;
        }

        internal override string profilerTag => "ReduceSum";
    }

    /// <summary>
    /// Represents a `ReduceSumSquare` reduction layer along the given axes: f(x1, x2 ... xn) = x1² + x2² + ... + xn².
    /// </summary>
    [Serializable]
    public class ReduceSumSquare : Reduce
    {
        /// <summary>
        /// Initializes and returns an instance of `ReduceSumSquare` reduction layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="inputs">The name of the input tensor and optionally the axes tensor. If no axes tensor is provided, the layer performs the reduction according to the value of `noopWithEmptyAxes`.</param>
        /// <param name="keepdims">Whether to keep the axis dimension in the output tensor. The default value is `true`.</param>
        /// <param name="noopWithEmptyAxes">Whether to perform an identity operation if the input axes tensor is empty. The default value is `false`.</param>
        public ReduceSumSquare(string name, string[] inputs, bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(name, inputs, keepdims, noopWithEmptyAxes) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var axes = inputTensors.Length > 1 && inputTensors[1] != null ? inputTensors[1].ToReadOnlySpan<int>() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
                return inputTensors[0].ShallowCopy();
            var O = ctx.backend.NewOutputTensor(inputTensors[0].shape.Reduce(axes, keepdims), inputTensors[0].dataType);
            if (O.shape.HasZeroDims())
                return O;
            if (inputTensors[0] is TensorInt)
                ctx.backend.ReduceSumSquare(inputTensors[0] as TensorInt, O as TensorInt, axes, keepdims);
            else
                ctx.backend.ReduceSumSquare(inputTensors[0] as TensorFloat, O as TensorFloat, axes, keepdims);
            return O;
        }

        internal override string profilerTag => "ReduceSumSquare";
    }
}
