using System;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents an element-wise `Acos` trigonometric layer: f(x) = acos(x).
    /// </summary>
    [Serializable]
    public class Acos : Activation
    {
        /// <summary>
        /// Initializes and returns an instance of Acos trigonometric layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Acos(string name, string input)
            : base(name, input) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorFloat(inputTensors[0].shape);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.Acos(inputTensors[0] as TensorFloat, O);
            return O;
        }

        internal override string profilerTag => "Acos";
    }

    /// <summary>
    /// Represents an element-wise `Acosh` trigonometric layer: f(x) = acosh(x).
    /// </summary>
    [Serializable]
    public class Acosh : Activation
    {
        /// <summary>
        /// Initializes and returns an instance of Acosh trigonometric layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Acosh(string name, string input)
            : base(name, input) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorFloat(inputTensors[0].shape);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.Acosh(inputTensors[0] as TensorFloat, O);
            return O;
        }

        internal override string profilerTag => "Acosh";
    }

    /// <summary>
    /// Represents an element-wise `Asin` trigonometric layer: f(x) = asin(x).
    /// </summary>
    [Serializable]
    public class Asin : Activation
    {
        /// <summary>
        /// Initializes and returns an instance of Asin trigonometric layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Asin(string name, string input)
            : base(name, input) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorFloat(inputTensors[0].shape);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.Asin(inputTensors[0] as TensorFloat, O);
            return O;
        }

        internal override string profilerTag => "Asin";
    }

    /// <summary>
    /// Represents an element-wise `Asinh` trigonometric layer: f(x) = asinh(x).
    /// </summary>
    [Serializable]
    public class Asinh : Activation
    {
        /// <summary>
        /// Initializes and returns an instance of Asinh trigonometric layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Asinh(string name, string input)
            : base(name, input) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorFloat(inputTensors[0].shape);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.Asinh(inputTensors[0] as TensorFloat, O);
            return O;
        }

        internal override string profilerTag => "Asinh";
    }

    /// <summary>
    /// Represents an element-wise `Atan` trigonometric layer: f(x) = atan(x).
    /// </summary>
    [Serializable]
    public class Atan : Activation
    {
        /// <summary>
        /// Initializes and returns an instance of Atan trigonometric layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Atan(string name, string input)
            : base(name, input) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorFloat(inputTensors[0].shape);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.Atan(inputTensors[0] as TensorFloat, O);
            return O;
        }

        internal override string profilerTag => "Atan";
    }

    /// <summary>
    /// Represents an element-wise `Atanh` trigonometric layer: f(x) = atanh(x).
    /// </summary>
    [Serializable]
    public class Atanh : Activation
    {
        /// <summary>
        /// Initializes and returns an instance of Atanh trigonometric layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Atanh(string name, string input)
            : base(name, input) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorFloat(inputTensors[0].shape);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.Atanh(inputTensors[0] as TensorFloat, O);
            return O;
        }

        internal override string profilerTag => "Atanh";
    }

    /// <summary>
    /// Represents an element-wise `Cos` trigonometric layer: f(x) = cos(x).
    /// </summary>
    [Serializable]
    public class Cos : Activation
    {
        /// <summary>
        /// Initializes and returns an instance of Cos trigonometric layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Cos(string name, string input)
            : base(name, input) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorFloat(inputTensors[0].shape);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.Cos(inputTensors[0] as TensorFloat, O);
            return O;
        }

        internal override string profilerTag => "Cos";
    }

    /// <summary>
    /// Represents an element-wise `Cosh` trigonometric layer: f(x) = cosh(x).
    /// </summary>
    [Serializable]
    public class Cosh : Activation
    {
        /// <summary>
        /// Initializes and returns an instance of Cosh trigonometric layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Cosh(string name, string input)
            : base(name, input) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorFloat(inputTensors[0].shape);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.Cosh(inputTensors[0] as TensorFloat, O);
            return O;
        }

        internal override string profilerTag => "Cosh";
    }

    /// <summary>
    /// Represents an element-wise `Sin` trigonometric layer: f(x) = sin(x).
    /// </summary>
    [Serializable]
    public class Sin : Activation
    {
        /// <summary>
        /// Initializes and returns an instance of Sin trigonometric layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Sin(string name, string input)
            : base(name, input) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorFloat(inputTensors[0].shape);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.Sin(inputTensors[0] as TensorFloat, O);
            return O;
        }

        internal override string profilerTag => "Sin";
    }

    /// <summary>
    /// Represents an element-wise `Sinh` trigonometric layer: f(x) = sinh(x).
    /// </summary>
    [Serializable]
    public class Sinh : Activation
    {
        /// <summary>
        /// Initializes and returns an instance of Sinh trigonometric layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Sinh(string name, string input)
            : base(name, input) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorFloat(inputTensors[0].shape);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.Sinh(inputTensors[0] as TensorFloat, O);
            return O;
        }

        internal override string profilerTag => "Sinh";
    }

    /// <summary>
    /// Represents an element-wise `Tan` trigonometric layer: f(x) = tan(x).
    /// </summary>
    [Serializable]
    public class Tan : Activation
    {
        /// <summary>
        /// Initializes and returns an instance of Tan trigonometric layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Tan(string name, string input)
            : base(name, input) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorFloat(inputTensors[0].shape);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.Tan(inputTensors[0] as TensorFloat, O);
            return O;
        }

        internal override string profilerTag => "Tan";
    }
}
