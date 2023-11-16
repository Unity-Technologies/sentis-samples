using System;
using System.Linq;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents an element-wise `Abs` math layer: f(x) = |x|.
    /// </summary>
    [Serializable]
    public class Abs : Activation
    {
        /// <summary>
        /// Initializes and returns an instance of `Abs` math layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Abs(string name, string input)
            : base(name, input) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensor(inputTensors[0].shape, inputTensors[0].dataType);
            if (O.shape.HasZeroDims())
                return O;
            if (inputTensors[0] is TensorInt)
                ctx.backend.Abs(inputTensors[0] as TensorInt, O as TensorInt);
            else
                ctx.backend.Abs(inputTensors[0] as TensorFloat, O as TensorFloat);
            return O;
        }

        internal override string profilerTag => "Abs";
    }

    /// <summary>
    /// Represents an element-wise `Add` math operation layer: f(a, b) = a + b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    public class Add : Broadcast
    {
        /// <summary>
        /// Initializes and returns an instance of `Add` math operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="a">The name to use for the first input tensor of the layer.</param>
        /// <param name="b">The name to use for the second input tensor of the layer.</param>
        public Add(string name, string a, string b)
            : base(name, a, b) { }

        /// <inheritdoc/>
        internal override Func<PartialTensorElement, PartialTensorElement, PartialTensorElement> InferPartialOp => (a, b) => a + b;

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensor(TensorShapeHelper.BroadcastShape(inputTensors[0], inputTensors[1]), inputTensors[0].dataType);
            if (O.shape.HasZeroDims())
                return O;
            if (inputTensors[0] is TensorInt)
                ctx.backend.Add(inputTensors[0] as TensorInt, inputTensors[1] as TensorInt, O as TensorInt);
            else
                ctx.backend.Add(inputTensors[0] as TensorFloat, inputTensors[1] as TensorFloat, O as TensorFloat);
            return O;
        }

        internal override string profilerTag => "Add";
    }

    /// <summary>
    /// Represents an element-wise `Ceil` math layer: f(x) = ceil(x).
    /// </summary>
    [Serializable]
    public class Ceil : Activation
    {
        /// <summary>
        /// Initializes and returns an instance of `Ceil` math layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Ceil(string name, string input)
            : base(name, input) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorFloat(inputTensors[0].shape);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.Ceil(inputTensors[0] as TensorFloat, O);
            return O;
        }

        internal override string profilerTag => "Ceil";
    }

    /// <summary>
    /// Represents an element-wise `Clip` math layer: f(x, xmin, xmax) = min(max(x, xmin), xmax)
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(1, 2)]
    public class Clip : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `Clip` math layer with no min or max tensors.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Clip(string name, string input)
        {
            this.name = name;
            this.inputs = new[] { input };
        }

        /// <summary>
        /// Initializes and returns an instance of `Clip` math layer with no max tensor.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="min">The name to use for the min scalar tensor of the layer.</param>
        public Clip(string name, string input, string min)
        {
            this.name = name;
            this.inputs = new[] { input, min };
        }

        /// <summary>
        /// Initializes and returns an instance of `Clip` math layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="min">The name to use for the min scalar tensor of the layer.</param>
        /// <param name="max">The name to use for the max scalar tensor of the layer.</param>
        public Clip(string name, string input, string min, string max)
        {
            this.name = name;
            this.inputs = new[] { input, min, max };
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            return new PartialTensor(DataType.Float, inputTensors[0].shape);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var min = inputTensors.Length > 1 && inputTensors[1] != null ? inputTensors[1].ToReadOnlySpan<float>()[0] : float.MinValue;
            var max = inputTensors.Length > 2 && inputTensors[2] != null ? inputTensors[2].ToReadOnlySpan<float>()[0] : float.MaxValue;
            var O = ctx.backend.NewOutputTensorFloat(inputTensors[0].shape);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.Clip(inputTensors[0] as TensorFloat, O, min, max);
            return O;
        }

        internal override string profilerTag => "Clip";
    }

    /// <summary>
    /// Represents a `CumSum` math layer that performs the cumulative sum along a given axis.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(1)]
    public class CumSum : Layer
    {
        /// <summary>
        /// Whether to perform the cumulative sum from the end of the axis.
        /// </summary>
        public bool reverse;
        /// <summary>
        /// Whether to include the respective input element in the cumulative sum.
        /// </summary>
        public bool exclusive;

        /// <summary>
        /// Initializes and returns an instance of `CumSum` math layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="axis">The name to use for the axis scalar tensor along which to perform the cumulative sum.</param>
        /// <param name="reverse">Whether to perform the cumulative sum from the end of the axis.</param>
        /// <param name="exclusive">Whether to include the respective input element in the cumulative sum.</param>
        public CumSum(string name, string input, string axis, bool reverse, bool exclusive)
        {
            this.name = name;
            this.inputs = new[] { input, axis };
            this.reverse = reverse;
            this.exclusive = exclusive;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            return new PartialTensor(inputTensors[0].dataType, inputTensors[0].shape);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensor(inputTensors[0].shape, inputTensors[0].dataType);
            if (O.shape.HasZeroDims())
                return O;
            var axis = inputTensors[1].ToReadOnlySpan<int>()[0];
            if (inputTensors[0] is TensorInt)
                ctx.backend.CumSum(inputTensors[0] as TensorInt, O as TensorInt, axis, reverse, exclusive);
            else
                ctx.backend.CumSum(inputTensors[0] as TensorFloat, O as TensorFloat, axis, reverse, exclusive);
            return O;
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, reverse: {reverse}, exclusive: {exclusive}";
        }

        internal override string profilerTag => "Hardmax";
    }

    /// <summary>
    /// Represents a `Dense` math operation layer which performs a matrix multiplication operation: f(x, w, b) = X x W + B.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    public class Dense : FusedActivation
    {
        /// <summary>
        /// Initializes and returns an instance of `Dense` math operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the first input tensor of the layer.</param>
        /// <param name="weights">The name to use for the weights input tensor of the layer.</param>
        /// <param name="bias">The name to use for the bias input tensor of the layer.</param>
        /// <param name="fusedActivation">The fusable activation to apply to the output tensor of the layer.</param>
        public Dense(string name, string input, string weights, string bias, FusableActivation fusedActivation = FusableActivation.None)
        {
            this.name = name;
            inputs = new[] { input, weights, bias };
            this.fusedActivation = fusedActivation;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var shapeB = inputTensors[2].shape;
            var shapeOut = inputTensors[0].shape.MatMul(inputTensors[1].shape);
            if (shapeOut.hasRank)
                shapeOut[-1] = SymbolicTensorDim.MaxDefinedDim(shapeB[0], shapeOut[-1]);
            return new PartialTensor(DataType.Float, shapeOut);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorFloat(inputTensors[0].shape.MatMul(inputTensors[1].shape));
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.Dense(inputTensors[0] as TensorFloat, inputTensors[1] as TensorFloat, inputTensors[2] as TensorFloat, O, fusedActivation);
            return O;
        }

        internal override string profilerTag => "Dense";
    }

    /// <summary>
    /// Represents an element-wise `Div` math operation layer: f(a, b) = a / b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    public class Div : Broadcast
    {
        /// <summary>
        /// Initializes and returns an instance of `Div` math operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="a">The name to use for the numerator input tensor of the layer.</param>
        /// <param name="b">The name to use for the denominator input tensor of the layer.</param>
        public Div(string name, string a, string b)
            : base(name, a, b) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensor(TensorShapeHelper.BroadcastShape(inputTensors[0], inputTensors[1]), inputTensors[0].dataType);
            if (O.shape.HasZeroDims())
                return O;
            if (inputTensors[0] is TensorInt)
                ctx.backend.Div(inputTensors[0] as TensorInt, inputTensors[1] as TensorInt, O as TensorInt);
            else
                ctx.backend.Div(inputTensors[0] as TensorFloat, inputTensors[1] as TensorFloat, O as TensorFloat);
            return O;
        }

        internal override string profilerTag => "Div";
    }

    /// <summary>
    /// Represents an `Einsum` math operation layer.
    /// </summary>
    /// <description>
    /// The Einsum operator evaluates algebraic tensor operations on a sequence of tensors, using the Einstein summation convention. The equation string contains a comma-separated sequence of lower case letters. Each term corresponds to an operand tensor, and the characters within the terms correspond to operands dimensions.
    /// This sequence may be followed by "->" to separate the left and right hand side of the equation. If the equation contains "->" followed by the right-hand side, the explicit (not classical) form of the Einstein summation is performed, and the right-hand side indices indicate output tensor dimensions. In other cases, output indices are (implicitly) set to the alphabetically sorted sequence of indices appearing exactly once in the equation.
    /// When a dimension character is repeated in the left-hand side, it represents summation along the dimension.
    /// The equation may contain ellipsis ("...") to enable broadcasting. Ellipsis must indicate a fixed number of dimensions. Specifically, every occurrence of ellipsis in the equation must represent the same number of dimensions. The right-hand side may contain exactly one ellipsis. In implicit mode, the ellipsis dimensions are set to the beginning of the output. The equation string may contain space (U+0020) character.
    /// </description>
    [Serializable]
    public class Einsum : Layer
    {
        /// <summary>
        /// The equation of the Einstein summation as a comma-separated list of subscript labels.
        /// </summary>
        public string equation;

        TensorShape[] operandShapes;
        TensorIndex[] operandIndices;

        /// <summary>
        /// Initializes and returns an instance of `Einsum` math operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="inputs">The names to use for the input tensors of the layer.</param>
        /// <param name="equation">The equation of the Einstein summation as a comma-separated list of subscript labels.</param>
        public Einsum(string name, string[] inputs, string equation)
        {
            this.name = name;
            this.inputs = inputs;
            this.equation = equation;
            operandShapes = new TensorShape[inputs.Length];
            operandIndices = new TensorIndex[inputs.Length];
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var operandIndices = new TensorIndex[inputTensors.Length];
            var shape = EinsumHelper.ParseEquationStringShape(equation, inputTensors.Select(i => i.shape).ToArray(), ref operandIndices, out _, out _);
            return new PartialTensor(DataType.Float, shape);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            for (var i = 0; i < inputTensors.Length; i++)
                operandShapes[i] = inputTensors[i].shape;
            EinsumHelper.ParseEquationString(equation, operandShapes, ref operandIndices, out var outputIndices, out var outputShape, out var sumIndices, out var sumShape, out var numIndices);
            var O = ctx.backend.NewOutputTensorFloat(outputShape);
            if (O.shape.HasZeroDims())
                return O;
            if (inputTensors.Length > 2)
                CPUBackend.EinsumND(Array.ConvertAll(inputTensors, i => i as TensorFloat), O, operandShapes, operandIndices, outputIndices, outputShape, sumIndices, sumShape, numIndices);
            else
                ctx.backend.Einsum(Array.ConvertAll(inputTensors, i => i as TensorFloat), O, operandIndices, outputIndices, sumIndices, sumShape);
            return O;
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, equation: {equation}";
        }

        internal override string profilerTag => "Einsum";
    }

    /// <summary>
    /// Represents an element-wise `Exp` math layer: f(x) = e^{x}.
    /// </summary>
    [Serializable]
    public class Exp : Activation
    {
        /// <summary>
        /// Initializes and returns an instance of `Exp` math layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Exp(string name, string input)
            : base(name, input) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorFloat(inputTensors[0].shape);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.Exp(inputTensors[0] as TensorFloat, O);
            return O;
        }

        internal override string profilerTag => "Exp";
    }

    /// <summary>
    /// Represents an element-wise `Floor` math layer: f(x) = floor(x).
    /// </summary>
    [Serializable]
    public class Floor : Activation
    {
        /// <summary>
        /// Initializes and returns an instance of `Floor` math layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Floor(string name, string input)
            : base(name, input) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorFloat(inputTensors[0].shape);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.Floor(inputTensors[0] as TensorFloat, O);
            return O;
        }

        internal override string profilerTag => "Floor";
    }

    /// <summary>
    /// Represents an element-wise `Log` math layer: f(x) = log(x).
    /// </summary>
    [Serializable]
    public class Log : Activation
    {
        /// <summary>
        /// Initializes and returns an instance of `Log` math layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Log(string name, string input)
            : base(name, input) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorFloat(inputTensors[0].shape);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.Log(inputTensors[0] as TensorFloat, O);
            return O;
        }

        internal override string profilerTag => "Log";
    }

    /// <summary>
    /// Represents a `MatMul` math operation layer which performs a matrix multiplication operation: f(a, b) = a x b.
    /// </summary>
    [Serializable]
    public class MatMul : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `MatMul` math layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input0">The name to use for the first input tensor of the layer.</param>
        /// <param name="input1">The name to use for the second input tensor of the layer.</param>
        public MatMul(string name, string input0, string input1)
        {
            this.name = name;
            this.inputs = new[] { input0, input1 };
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            return new PartialTensor(inputTensors[0].dataType, inputTensors[0].shape.MatMul(inputTensors[1].shape));
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorFloat(inputTensors[0].shape.MatMul(inputTensors[1].shape));
            if (O.shape.HasZeroDims())
                return O;
            if (inputTensors[0].shape.HasZeroDims() || inputTensors[1].shape.HasZeroDims())
                ctx.backend.MemSet(O, 0.0f);
            else
                ctx.backend.MatMul(inputTensors[0] as TensorFloat, inputTensors[1] as TensorFloat, O);
            return O;
        }

        internal override string profilerTag => "MatMul";
    }

    /// <summary>
    /// Represents a `MatMul2D` math operation layer which performs a matrix multiplication operation with optional transposes: f(a, b) = a' x b'.
    /// </summary>
    [Serializable]
    public class MatMul2D : Layer
    {
        /// <summary>
        /// Whether to transpose the first input before performing the matrix multiplication.
        /// </summary>
        public bool transposeA;
        /// <summary>
        /// Whether to transpose the second input before performing the matrix multiplication.
        /// </summary>
        public bool transposeB;

        /// <summary>
        /// Initializes and returns an instance of `MatMul2D` math layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input0">The name to use for the first input tensor of the layer.</param>
        /// <param name="transpose0">Whether to transpose the first input before performing the matrix multiplication.</param>
        /// <param name="input1">The name to use for the second input tensor of the layer.</param>
        /// <param name="transpose1">Whether to transpose the second input before performing the matrix multiplication.</param>
        public MatMul2D(string name, string input0, bool transpose0, string input1, bool transpose1)
        {
            this.name = name;
            this.inputs = new[] { input0, input1 };
            this.transposeA = transpose0;
            this.transposeB = transpose1;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var shapeX = inputTensors[0].shape;
            var shapeY = inputTensors[1].shape;

            shapeX.DeclareRank(2);
            shapeY.DeclareRank(2);

            var mulXDim = transposeA ? shapeX[0] : shapeX[1];
            var mulYDim = transposeB ? shapeY[1] : shapeY[0];
            Logger.AssertIsFalse(mulXDim != mulYDim, "MatMul2D.ValueError: failed, dims not equal");

            var shapeOut = new SymbolicTensorShape(transposeA ? shapeX[1] : shapeX[0], transposeB ? shapeY[0] : shapeY[1]);
            return new PartialTensor(inputTensors[0].dataType, shapeOut);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorFloat(ShapeInference.Gemm(inputTensors[0].shape, inputTensors[1].shape, transposeA, transposeB));
            if (O.shape.HasZeroDims())
                return O;
            if (inputTensors[0].shape.HasZeroDims() || inputTensors[1].shape.HasZeroDims())
                ctx.backend.MemSet(O, 0.0f);
            else
                ctx.backend.MatMul2D(inputTensors[0] as TensorFloat, inputTensors[1] as TensorFloat, O, transposeA, transposeB);
            return O;
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, transposeA: {transposeA}, transposeB: {transposeB}";
        }

        internal override string profilerTag => "MatMul2D";
    }

    /// <summary>
    /// Represents an element-wise `Max` math operation layer: f(x1, x2 ... xn) = max(x1, x2 ... xn).
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    public class Max : Broadcast
    {
        /// <summary>
        /// Initializes and returns an instance of `Max` math operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="inputs">The array of names to use for the input tensors of the layer.</param>
        public Max(string name, string[] inputs)
            : base(name, inputs) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensor(TensorShapeHelper.BroadcastShape(inputTensors), inputTensors[0].dataType);
            if (O.shape.HasZeroDims())
                return O;
            if (inputTensors[0] is TensorInt)
                ctx.backend.Max(Array.ConvertAll(inputTensors, i => i as TensorInt), O as TensorInt);
            else
                ctx.backend.Max(Array.ConvertAll(inputTensors, i => i as TensorFloat), O as TensorFloat);
            return O;
        }

        internal override string profilerTag => "Max";
    }

    /// <summary>
    /// Represents an element-wise `Mean` math operation layer: f(x1, x2 ... xn) = (x1 + x2 ... xn) / n.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    public class Mean : Broadcast
    {
        /// <summary>
        /// Initializes and returns an instance of `Mean` math operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="inputs">The array of names to use for the input tensors of the layer.</param>
        public Mean(string name, string[] inputs)
            : base(name, inputs) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorFloat(TensorShapeHelper.BroadcastShape(inputTensors));
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.Mean(Array.ConvertAll(inputTensors, i => i as TensorFloat), O);
            return O;
        }

        internal override string profilerTag => "Mean";
    }

    /// <summary>
    /// Represents an element-wise `Min` math operation layer: f(x1, x2 ... xn) = min(x1, x2 ... xn).
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    public class Min : Broadcast
    {
        /// <summary>
        /// Initializes and returns an instance of `Min` math operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="inputs">The array of names to use for the input tensors of the layer.</param>
        public Min(string name, string[] inputs)
            : base(name, inputs) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensor(TensorShapeHelper.BroadcastShape(inputTensors), inputTensors[0].dataType);
            if (O.shape.HasZeroDims())
                return O;
            if (inputTensors[0] is TensorInt)
                ctx.backend.Min(Array.ConvertAll(inputTensors, i => i as TensorInt), O as TensorInt);
            else
                ctx.backend.Min(Array.ConvertAll(inputTensors, i => i as TensorFloat), O as TensorFloat);
            return O;
        }

        internal override string profilerTag => "Min";
    }

    /// <summary>
    /// Represents an element-wise `Max` math operation layer: f(a, b) = a % b.
    ///
    /// If fmod is false the sign of the remainder is the same as that of the divisor as in Python.
    ///
    /// If fmod is true the sign of the remainder is the same as that of the dividend as in C#.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    public class Mod : Broadcast
    {
        /// <summary>
        /// Whether to have the sign of the remainder the same as that of the dividend rather than that of the divisor.
        /// </summary>
        public bool fmod;

        /// <summary>
        /// Initializes and returns an instance of `Mod` math operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="a">The name to use for the divisor input tensor of the layer.</param>
        /// <param name="b">The name to use for the dividend input tensor of the layer.</param>
        /// <param name="fmod">Whether to have the sign of the remainder the same as that of the dividend rather than that of the divisor.</param>
        public Mod(string name, string a, string b, bool fmod = false)
            : base(name, a, b)
        {
            this.fmod = fmod;
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensor(TensorShapeHelper.BroadcastShape(inputTensors[0], inputTensors[1]), inputTensors[0].dataType);
            if (O.shape.HasZeroDims())
                return O;
            if (!fmod)
                ctx.backend.Mod(inputTensors[0] as TensorInt, inputTensors[1] as TensorInt, O as TensorInt);
            else if (inputTensors[0] is TensorInt)
                ctx.backend.FMod(inputTensors[0] as TensorInt, inputTensors[1] as TensorInt, O as TensorInt);
            else
                ctx.backend.FMod(inputTensors[0] as TensorFloat, inputTensors[1] as TensorFloat, O as TensorFloat);
            return O;
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, fmod: {fmod}";
        }

        internal override string profilerTag => "Mod";
    }

    /// <summary>
    /// Represents an element-wise `Mul` math operation layer: f(a, b) = a * b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    public class Mul : Broadcast
    {
        /// <summary>
        /// Initializes and returns an instance of `Mul` math operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="a">The name to use for the first input tensor of the layer.</param>
        /// <param name="b">The name to use for the second input tensor of the layer.</param>
        public Mul(string name, string a, string b)
            : base(name, a, b) { }

        /// <inheritdoc/>
        internal override Func<PartialTensorElement, PartialTensorElement, PartialTensorElement> InferPartialOp => (a, b) => a * b;

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensor(TensorShapeHelper.BroadcastShape(inputTensors[0], inputTensors[1]), inputTensors[0].dataType);
            if (O.shape.HasZeroDims())
                return O;
            if (inputTensors[0] is TensorInt)
                ctx.backend.Mul(inputTensors[0] as TensorInt, inputTensors[1] as TensorInt, O as TensorInt);
            else
                ctx.backend.Mul(inputTensors[0] as TensorFloat, inputTensors[1] as TensorFloat, O as TensorFloat);
            return O;
        }

        internal override string profilerTag => "Mul";
    }

    /// <summary>
    /// Represents an element-wise `Neg` math layer: f(x) = -x.
    /// </summary>
    [Serializable]
    public class Neg : Activation
    {
        /// <summary>
        /// Initializes and returns an instance of `Neg` math layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Neg(string name, string input)
            : base(name, input) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensor(inputTensors[0].shape, inputTensors[0].dataType);
            if (O.shape.HasZeroDims())
                return O;
            if (inputTensors[0] is TensorInt)
                ctx.backend.Neg(inputTensors[0] as TensorInt, O as TensorInt);
            else
                ctx.backend.Neg(inputTensors[0] as TensorFloat, O as TensorFloat);
            return O;
        }

        internal override string profilerTag => "Neg";
    }

    /// <summary>
    /// Represents an element-wise `Pow` math operation layer: f(a, b) = pow(a, b).
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    public class Pow : Broadcast
    {
        /// <summary>
        /// Initializes and returns an instance of `Pow` math operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="a">The name to use for the first input tensor of the layer.</param>
        /// <param name="b">The name to use for the second input tensor of the layer.</param>
        public Pow(string name, string a, string b)
            : base(name, a, b) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorFloat(TensorShapeHelper.BroadcastShape(inputTensors[0], inputTensors[1]));
            if (O.shape.HasZeroDims())
                return O;
            if (inputTensors[1] is TensorInt)
                ctx.backend.Pow(inputTensors[0] as TensorFloat, inputTensors[1] as TensorInt, O);
            else
                ctx.backend.Pow(inputTensors[0] as TensorFloat, inputTensors[1] as TensorFloat, O);
            return O;
        }

        internal override string profilerTag => "Pow";
    }

    /// <summary>
    /// Represents an element-wise `Reciprocal` math layer: f(x) = 1 / x.
    /// </summary>
    [Serializable]
    public class Reciprocal : Activation
    {
        /// <summary>
        /// Initializes and returns an instance of `Reciprocal` math layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Reciprocal(string name, string input)
            : base(name, input) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorFloat(inputTensors[0].shape);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.Reciprocal(inputTensors[0] as TensorFloat, O);
            return O;
        }

        internal override string profilerTag => "Reciprocal";
    }

    /// <summary>
    /// Represents an element-wise `Round` math layer: f(x) = round(x).
    /// </summary>
    [Serializable]
    public class Round : Activation
    {
        /// <summary>
        /// Initializes and returns an instance of `Round` math layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Round(string name, string input)
            : base(name, input) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorFloat(inputTensors[0].shape);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.Round(inputTensors[0] as TensorFloat, O);
            return O;
        }

        internal override string profilerTag => "Round";
    }

    /// <summary>
    /// Represents an element-wise `Mad` math operation: multiplies and adds bias to a tensor: f(T, s, b) = s * T + b.
    /// </summary>
    [Serializable]
    public class ScalarMad : Activation
    {
        /// <summary>
        /// Input scalar for multiplication.
        /// </summary>
        public float s;
        /// <summary>
        /// Input bias for addition.
        /// </summary>
        public float b;

        /// <summary>
        /// Initializes and returns an instance of `ScalarMad` math layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="s">The value of the scale for the scalarmad function.</param>
        /// <param name="b">The value of the bias for the scalarmad function.</param>
        public ScalarMad(string name, string input, float s, float b)
            : base(name, input)
        {
            this.s = s;
            this.b = b;
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorFloat(inputTensors[0].shape);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.ScalarMad(inputTensors[0] as TensorFloat, O, s, b);
            return O;
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, s: {s}, b: {b}";
        }

        internal override string profilerTag => "ScalarMad";
    }

    /// <summary>
    /// Represents an element-wise `Shrink` math layer: f(x) = x + bias if x &lt; lambd. f(x) = x - bias if x &gt; lambd. Otherwise f(x) = 0.
    /// </summary>
    [Serializable]
    public class Shrink : Layer
    {
        /// <summary>
        /// The value of the bias for the shrink function.
        /// </summary>
        public float bias;
        /// <summary>
        /// The value of lambda for the shrink function.
        /// </summary>
        public float lambd;

        /// <summary>
        /// Initializes and returns an instance of `Shrink` math layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="bias">The value of the bias for the shrink function.</param>
        /// <param name="lambd">The value of lambda for the shrink function.</param>
        public Shrink(string name, string input, float bias, float lambd)
        {
            this.name = name;
            inputs = new[] { input };
            this.bias = bias;
            this.lambd = lambd;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            return new PartialTensor(inputTensors[0].dataType, inputTensors[0].shape);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorFloat(inputTensors[0].shape);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.Shrink(inputTensors[0] as TensorFloat, O, bias, lambd);
            return O;
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, bias: {bias}, lambd: {lambd}";
        }

        internal override string profilerTag => "Shrink";
    }

    /// <summary>
    /// Represents an element-wise `Sign` math layer: f(x) = 1 if x > 0. f(x) = -1 if x &lt; 0. Otherwise f(x) = 0.
    /// </summary>
    [Serializable]
    public class Sign : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `Sign` math layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Sign(string name, string input)
        {
            this.name = name;
            this.inputs = new[] { input };
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            return new PartialTensor(inputTensors[0].dataType, inputTensors[0].shape);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensor(inputTensors[0].shape, inputTensors[0].dataType);
            if (O.shape.HasZeroDims())
                return O;
            if (inputTensors[0] is TensorInt)
                ctx.backend.Sign(inputTensors[0] as TensorInt, O as TensorInt);
            else
                ctx.backend.Sign(inputTensors[0] as TensorFloat, O as TensorFloat);
            return O;
        }

        internal override string profilerTag => "Sign";
    }

    /// <summary>
    /// Represents an element-wise `Sqrt` math layer: f(x) = sqrt(x).
    /// </summary>
    [Serializable]
    public class Sqrt : Activation
    {
        /// <summary>
        /// Initializes and returns an instance of `Sqrt` math layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Sqrt(string name, string input)
            : base(name, input) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorFloat(inputTensors[0].shape);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.Sqrt(inputTensors[0] as TensorFloat, O);
            return O;
        }

        internal override string profilerTag => "Sqrt";
    }

    /// <summary>
    /// Represents an element-wise `Square` math layer: f(x) = x * x.
    /// </summary>
    [Serializable]
    public class Square : Activation
    {
        /// <summary>
        /// Initializes and returns an instance of `Square` math layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Square(string name, string input)
            : base(name, input) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorFloat(inputTensors[0].shape);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.Square(inputTensors[0] as TensorFloat, O);
            return O;
        }

        internal override string profilerTag => "Square";
    }

    /// <summary>
    /// Represents an element-wise `Sub` math operation layer: f(a, b) = a - b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    public class Sub : Broadcast
    {
        /// <summary>
        /// Initializes and returns an instance of `Sub` math operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="a">The name to use for the first input tensor of the layer.</param>
        /// <param name="b">The name to use for the second input tensor of the layer.</param>
        public Sub(string name, string a, string b)
            : base(name, a, b) { }

        /// <inheritdoc/>
        internal override Func<PartialTensorElement, PartialTensorElement, PartialTensorElement> InferPartialOp => (a, b) => a - b;

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensor(TensorShapeHelper.BroadcastShape(inputTensors[0], inputTensors[1]), inputTensors[0].dataType);
            if (O.shape.HasZeroDims())
                return O;
            if (inputTensors[0] is TensorInt)
                ctx.backend.Sub(inputTensors[0] as TensorInt, inputTensors[1] as TensorInt, O as TensorInt);
            else
                ctx.backend.Sub(inputTensors[0] as TensorFloat, inputTensors[1] as TensorFloat, O as TensorFloat);
            return O;
        }

        internal override string profilerTag => "Sub";
    }

    /// <summary>
    /// Represents an element-wise `Sum` math operation layer: f(x1, x2 ... xn) = x1 + x2 ... xn.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    public class Sum : Broadcast
    {
        /// <summary>
        /// Initializes and returns an instance of `Sum` math operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="inputs">The array of names to use for the input tensors of the layer.</param>
        public Sum(string name, string[] inputs)
            : base(name, inputs) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorFloat(TensorShapeHelper.BroadcastShape(inputTensors));
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.Sum(Array.ConvertAll(inputTensors, i => i as TensorFloat), O);
            return O;
        }

        internal override string profilerTag => "Sum";
    }
}
