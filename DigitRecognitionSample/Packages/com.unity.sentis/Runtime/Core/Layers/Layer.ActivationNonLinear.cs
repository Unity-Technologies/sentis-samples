using System;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents a `LogSoftmax` activation layer along an axis: f(x, axis) = log(Softmax(x, axis)).
    /// </summary>
    [Serializable]
    public class LogSoftmax : Activation
    {
        /// <summary>
        /// The axis along which to apply the `LogSoftmax` activation function.
        /// </summary>
        public int axis;

        /// <summary>
        /// Initializes and returns an instance of `LogSoftmax` activation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="axis">The axis along which to apply the `LogSoftmax` activation function.</param>
        public LogSoftmax(string name, string input, int axis = -1)
            : base(name, input)
        {
            this.axis = axis;
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorFloat(inputTensors[0].shape);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.LogSoftmax(inputTensors[0] as TensorFloat, O, axis);
            return O;
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, axis: {axis}";
        }

        internal override string profilerTag => "LogSoftmax";
    }

    /// <summary>
    /// Represents a `Softmax` activation layer along an axis: f(x, axis) = exp(X) / ReduceSum(exp(X), axis).
    /// </summary>
    [Serializable]
    public class Softmax : Activation
    {
        /// <summary>
        /// The axis along which to apply the `Softmax` activation function.
        /// </summary>
        public int axis;

        /// <summary>
        /// Initializes and returns an instance of `Softmax` activation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="axis">The axis along which to apply the `Softmax` activation function.</param>
        public Softmax(string name, string input, int axis = -1)
            : base(name, input)
        {
            this.axis = axis;
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorFloat(inputTensors[0].shape);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.Softmax(inputTensors[0] as TensorFloat, O, axis);
            return O;
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, axis: {axis}";
        }

        internal override string profilerTag => "Softmax";
    }
}
