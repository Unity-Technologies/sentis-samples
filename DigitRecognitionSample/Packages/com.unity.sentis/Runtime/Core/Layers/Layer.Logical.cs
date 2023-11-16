using System;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents an element-wise comparison layer.
    /// </summary>
    [Serializable]
    public abstract class Comparison : Broadcast
    {
        /// <summary>
        /// Initializes and returns an instance of `Comparison` logical operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="a">The name to use for the first input tensor of the layer.</param>
        /// <param name="b">The name to use for the second input tensor of the layer.</param>
        protected Comparison(string name, string a, string b)
            : base(name, a, b) { }

        /// <inheritdoc/>
        internal override DataType InferPartialDataType(PartialTensor[] inputTensors)
        {
            return DataType.Int;
        }
    }

    /// <summary>
    /// Represents an element-wise `And` logical operation layer: f(a, b) = a &amp; b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    public class And : Broadcast
    {
        /// <summary>
        /// Initializes and returns an instance of `And` logical operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="a">The name to use for the first input tensor of the layer.</param>
        /// <param name="b">The name to use for the second input tensor of the layer.</param>
        public And(string name, string a, string b)
            : base(name, a, b) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorInt(TensorShapeHelper.BroadcastShape(inputTensors[0], inputTensors[1]));
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.And(inputTensors[0] as TensorInt, inputTensors[1] as TensorInt, O);
            return O;
        }

        internal override string profilerTag => "And";
    }

    /// <summary>
    /// Represents a `Compress` logical layer that selects slices of an input tensor along a given axis according to a condition tensor.
    /// If you don't provide an axis, the layer flattens the input tensor.
    /// </summary>
    [Serializable]
    public class Compress : Layer
    {
        /// <summary>
        /// Whether to perform the `Compress` along an axis. If `false`, the layer flattens the input tensor.
        /// </summary>
        public bool hasAxis;
        /// <summary>
        /// The axis along which to apply the Compress when `hasAxis` is `true`.
        /// </summary>
        public int axis;

        /// <summary>
        /// Initializes and returns an instance of `Compress` logical layer without an axis. The layer flattens the input tensor.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="condition">The name to use for the condition tensor of the layer.</param>
        public Compress(string name, string input, string condition)
        {
            this.name = name;
            inputs = new[] { input, condition };
        }

        /// <summary>
        /// Initializes and returns an instance of `Compress` logical layer along an axis.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="condition">The name to use for the condition tensor of the layer.</param>
        /// <param name="axis">The axis along which to apply the `Compress`.</param>
        public Compress(string name, string input, string condition, int axis)
        {
            this.name = name;
            inputs = new[] { input, condition };
            hasAxis = true;
            this.axis = axis;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var dataType = inputTensors[0].dataType;
            var shapeX = inputTensors[0].shape;
            var condition = inputTensors[1];
            var isZero = shapeX.Length() * condition.shape.Length() == 0;
            if (hasAxis)
            {
                var shapeOut = shapeX;
                shapeOut[axis] = isZero ? SymbolicTensorDim.Zero : SymbolicTensorDim.Unknown;
                return new PartialTensor(dataType, shapeOut);
            }

            return new PartialTensor(dataType, new SymbolicTensorShape(isZero ? SymbolicTensorDim.Zero : SymbolicTensorDim.Unknown));
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var X = inputTensors[0];
            if (!hasAxis)
            {
                var flattenedShape = new TensorShape(X.shape.length);
                X = ctx.backend.NewTensor(flattenedShape, X.dataType, AllocScope.InternalToLayer);
                ctx.backend.Reshape(inputTensors[0], X);
            }

            var condition = inputTensors[1] as TensorInt;
            var numCondition = condition.shape.length;

            var indices = ctx.backend.NewTempTensorInt(condition.shape);
            ArrayTensorData.Pin(indices, clearOnInit: false);

            var numIndices = 0;
            for (var i = 0; i < numCondition; i++)
            {
                if (condition[i] == 0)
                    continue;
                indices[numIndices] = i;
                numIndices++;
            }

            var O = ctx.backend.NewOutputTensor(ShapeInference.Compress(X.shape, numIndices, axis), X.dataType);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.CompressWithIndices(X, indices, O, numIndices, axis);
            return O;
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, hasAxis: {hasAxis}, axis: {axis}";
        }

        internal override string profilerTag => "Compress";
    }

    /// <summary>
    /// Represents an element-wise `Equal` logical operation layer: f(a, b) = 1 if a == b, otherwise f(x) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    public class Equal : Comparison
    {
        /// <summary>
        /// Initializes and returns an instance of `Equal` logical operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="a">The name to use for the first input tensor of the layer.</param>
        /// <param name="b">The name to use for the second input tensor of the layer.</param>
        public Equal(string name, string a, string b)
            : base(name, a, b) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorInt(TensorShapeHelper.BroadcastShape(inputTensors[0], inputTensors[1]));
            if (O.shape.HasZeroDims())
                return O;
            if (inputTensors[0] is TensorInt)
                ctx.backend.Equal(inputTensors[0] as TensorInt, inputTensors[1] as TensorInt, O);
            else
                ctx.backend.Equal(inputTensors[0] as TensorFloat, inputTensors[1] as TensorFloat, O);
            return O;
        }

        internal override string profilerTag => "Equal";
    }

    /// <summary>
    /// Represents an element-wise `Greater` logical operation layer: f(a, b) = 1 if a > b, otherwise f(x) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    public class Greater : Comparison
    {
        /// <summary>
        /// Initializes and returns an instance of `Greater` logical operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="a">The name to use for the first input tensor of the layer.</param>
        /// <param name="b">The name to use for the second input tensor of the layer.</param>
        public Greater(string name, string a, string b)
            : base(name, a, b) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorInt(TensorShapeHelper.BroadcastShape(inputTensors[0], inputTensors[1]));
            if (O.shape.HasZeroDims())
                return O;
            if (inputTensors[0] is TensorInt)
                ctx.backend.Greater(inputTensors[0] as TensorInt, inputTensors[1] as TensorInt, O);
            else
                ctx.backend.Greater(inputTensors[0] as TensorFloat, inputTensors[1] as TensorFloat, O);
            return O;
        }

        internal override string profilerTag => "Greater";
    }

    /// <summary>
    /// Represents an element-wise `GreaterOrEqual` logical operation layer: f(a, b) = 1 if a >= b, otherwise f(a,b) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    public class GreaterOrEqual : Comparison
    {
        /// <summary>
        /// Initializes and returns an instance of `GreaterOrEqual` logical operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="a">The name to use for the first input tensor of the layer.</param>
        /// <param name="b">The name to use for the second input tensor of the layer.</param>
        public GreaterOrEqual(string name, string a, string b)
            : base(name, a, b) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorInt(TensorShapeHelper.BroadcastShape(inputTensors[0], inputTensors[1]));
            if (O.shape.HasZeroDims())
                return O;
            if (inputTensors[0] is TensorInt)
                ctx.backend.GreaterOrEqual(inputTensors[0] as TensorInt, inputTensors[1] as TensorInt, O);
            else
                ctx.backend.GreaterOrEqual(inputTensors[0] as TensorFloat, inputTensors[1] as TensorFloat, O);
            return O;
        }

        internal override string profilerTag => "GreaterOrEqual";
    }

    /// <summary>
    /// Represents an element-wise `IsInf` logical layer: f(x) = 1 elementwise if x is +Inf and detectPositive, or x is -Inf and `detectNegative` is true. Otherwise f(x) = 0.
    /// </summary>
    [Serializable]
    public class IsInf : Layer
    {
        /// <summary>
        /// Whether to detect negative infinities in the `IsInf` function.
        /// </summary>
        public bool detectNegative;
        /// <summary>
        /// Whether to detect positive infinities in the `IsInf` function.
        /// </summary>
        public bool detectPositive;

        /// <summary>
        /// Initializes and returns an instance of `IsInf` logical layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="detectNegative">Whether to detect negative infinities in the `IsInf` function.</param>
        /// <param name="detectPositive">Whether to detect positive infinities in the `IsInf` function.</param>
        public IsInf(string name, string input, bool detectNegative, bool detectPositive)
        {
            this.name = name;
            inputs = new[] { input };
            this.detectNegative = detectNegative;
            this.detectPositive = detectPositive;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            return new PartialTensor(DataType.Int, inputTensors[0].shape);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorInt(inputTensors[0].shape);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.IsInf(inputTensors[0] as TensorFloat, O, detectNegative, detectPositive);
            return O;
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, detectNegative: {detectNegative}, detectPositive: {detectPositive}";
        }

        internal override string profilerTag => "IsInf";
    }

    /// <summary>
    /// Represents an element-wise `IsNaN` logical layer: f(x) = 1 if x is NaN, otherwise f(x) = 0.
    /// </summary>
    [Serializable]
    public class IsNaN : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `IsNaN` logical layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public IsNaN(string name, string input)
        {
            this.name = name;
            inputs = new[] { input };
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            return new PartialTensor(DataType.Int, inputTensors[0].shape);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorInt(inputTensors[0].shape);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.IsNaN(inputTensors[0] as TensorFloat, O);
            return O;
        }

        internal override string profilerTag => "IsNaN";
    }

    /// <summary>
    /// Represents an element-wise `Less` logical operation layer: f(a, b) = 1 if a &lt; b, otherwise f(x) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    public class Less : Comparison
    {
        /// <summary>
        /// Initializes and returns an instance of `Less` logical operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="a">The name to use for the first input tensor of the layer.</param>
        /// <param name="b">The name to use for the second input tensor of the layer.</param>
        public Less(string name, string a, string b)
            : base(name, a, b) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorInt(TensorShapeHelper.BroadcastShape(inputTensors[0], inputTensors[1]));
            if (O.shape.HasZeroDims())
                return O;
            if (inputTensors[0] is TensorInt)
                ctx.backend.Less(inputTensors[0] as TensorInt, inputTensors[1] as TensorInt, O);
            else
                ctx.backend.Less(inputTensors[0] as TensorFloat, inputTensors[1] as TensorFloat, O);
            return O;
        }

        internal override string profilerTag => "Less";
    }

    /// <summary>
    /// Represents an element-wise `LessOrEqual` logical operation layer: f(a, b) = 1 if a &lt;= b, otherwise f(a,b) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    public class LessOrEqual : Comparison
    {
        /// <summary>
        /// Initializes and returns an instance of `LessOrEqual` logical operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="a">The name to use for the first input tensor of the layer.</param>
        /// <param name="b">The name to use for the second input tensor of the layer.</param>
        public LessOrEqual(string name, string a, string b)
            : base(name, a, b) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorInt(TensorShapeHelper.BroadcastShape(inputTensors[0], inputTensors[1]));
            if (O.shape.HasZeroDims())
                return O;
            if (inputTensors[0] is TensorInt)
                ctx.backend.LessOrEqual(inputTensors[0] as TensorInt, inputTensors[1] as TensorInt, O);
            else
                ctx.backend.LessOrEqual(inputTensors[0] as TensorFloat, inputTensors[1] as TensorFloat, O);
            return O;
        }

        internal override string profilerTag => "LessOrEqual";
    }

    /// <summary>
    /// Represents an element-wise `Not` logical layer: f(x) = ~x.
    /// </summary>
    [Serializable]
    public class Not : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `Not` logical layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Not(string name, string input)
        {
            this.name = name;
            this.inputs = new[] { input };
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            return new PartialTensor(DataType.Int, inputTensors[0].shape);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorInt(inputTensors[0].shape);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.Not(inputTensors[0] as TensorInt, O);
            return O;
        }

        internal override string profilerTag => "Not";
    }

    /// <summary>
    /// Represents an element-wise `Or` logical operation layer: f(a, b) = a | b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    public class Or : Broadcast
    {
        /// <summary>
        /// Initializes and returns an instance of `Or` logical operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="a">The name to use for the first input tensor of the layer.</param>
        /// <param name="b">The name to use for the second input tensor of the layer.</param>
        public Or(string name, string a, string b)
            : base(name, a, b) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorInt(TensorShapeHelper.BroadcastShape(inputTensors[0], inputTensors[1]));
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.Or(inputTensors[0] as TensorInt, inputTensors[1] as TensorInt, O);
            return O;
        }

        internal override string profilerTag => "Or";
    }

    /// <summary>
    /// Represents an element-wise `Xor` logical operation layer: f(a, b) = a ^ b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    public class Xor : Broadcast
    {
        /// <summary>
        /// Initializes and returns an instance of `Xor` logical operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="a">The name to use for the first input tensor of the layer.</param>
        /// <param name="b">The name to use for the second input tensor of the layer.</param>
        public Xor(string name, string a, string b)
            : base(name, a, b) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorInt(TensorShapeHelper.BroadcastShape(inputTensors[0], inputTensors[1]));
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.Xor(inputTensors[0] as TensorInt, inputTensors[1] as TensorInt, O);
            return O;
        }

        internal override string profilerTag => "Xor";
    }

    /// <summary>
    /// Represents an element-wise `Where` logical operation layer: f(condition, a, b) = a if `condition`, otherwise f(condition, a, b) = b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    public class Where : Broadcast
    {
        /// <summary>
        /// Initializes and returns an instance of `Where` logical operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="condition">The name to use for the condition input tensor of the layer.</param>
        /// <param name="input1">The name to use for the first input tensor of the layer.</param>
        /// <param name="input2">The name to use for the second input tensor of the layer.</param>
        public Where(string name, string condition, string input1, string input2)
            : base(name, condition, input1, input2) { }

        /// <inheritdoc/>
        internal override DataType InferPartialDataType(PartialTensor[] inputTensors)
        {
            return inputTensors[1].dataType;
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensor(inputTensors[0].shape.Broadcast(inputTensors[1].shape.Broadcast(inputTensors[2].shape)), inputTensors[1].dataType);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.Where(inputTensors[0] as TensorInt, inputTensors[1], inputTensors[2], O);
            return O;
        }

        internal override string profilerTag => "Where";
    }
}
