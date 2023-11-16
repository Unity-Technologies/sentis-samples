using System;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents a `ConstantOfShape` layer. This generates a tensor with the shape given by the `input` tensor and filled with a given value.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(0)]
    public class ConstantOfShape : Layer
    {
        /// <summary>
        /// The data type of the layer as a `DataType`.
        /// </summary>
        public DataType dataType;
        /// <summary>
        /// The float value to use to fill the output tensor. The layer only uses this when the `dataType` equals `DataType.Float`.
        /// </summary>
        public float floatValue;
        /// <summary>
        /// The int value to use to fill the output tensor. The layer only uses this when the `dataType` equals `DataType.Int`.
        /// </summary>
        public int intValue;

        /// <summary>
        /// Initializes and returns an instance of `ConstantOfShape` layer with a float value.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the shape tensor of the layer.</param>
        /// <param name="value">The float value to use to fill the output tensor.</param>
        public ConstantOfShape(string name, string input, float value)
        {
            this.name = name;
            this.inputs = new[] { input };
            this.floatValue = value;
            this.dataType = DataType.Float;
        }

        /// <summary>
        /// Initializes and returns an instance of `ConstantOfShape` layer with an int value.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the shape tensor of the layer.</param>
        /// <param name="value">The int value to use to fill the output tensor.</param>
        public ConstantOfShape(string name, string input, int value)
        {
            this.name = name;
            this.inputs = new[] { input };
            this.intValue = value;
            this.dataType = DataType.Int;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var shape = inputTensors[0].ToSymbolicTensorShape();
            var tensorOut = new PartialTensor(dataType, shape);
            if (!tensorOut.isPartiallyKnown)
                return tensorOut;
            for (var i = 0; i < tensorOut.length; i++)
            {
                tensorOut[i] = dataType == DataType.Float ? new PartialTensorElement(floatValue) : new PartialTensorElement(intValue);
            }

            return tensorOut;
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            TensorShape shape = new TensorShape(inputTensors[0].ToReadOnlySpan<int>());
            var O = ctx.backend.NewOutputTensor(shape, dataType);
            if (O.shape.HasZeroDims())
                return O;
            if (dataType == DataType.Int)
                ctx.backend.MemSet(O as TensorInt, intValue);
            else
                ctx.backend.MemSet(O as TensorFloat, floatValue);
            return O;
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"ConstantOfShape{dataType.ToString()} - name: {name}, value: {floatValue}";
        }

        internal override string profilerTag => "ConstantOfShape";
    }

    /// <summary>
    /// Represents a `OneHot` layer. This generates a one-hot tensor with a given `depth`, `indices` and `values`.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(1, 2)]
    public class OneHot : Layer
    {
        /// <summary>
        /// The axis along which the layer adds the one-hot representation.
        /// </summary>
        public int axis;

        /// <summary>
        /// Initializes and returns an instance of `OneHot` layer with a float value.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="indices">The name to use for the indices tensor of the layer.</param>
        /// <param name="depth">The name to use for the scalar depth tensor of the layer.</param>
        /// <param name="values">The name to use for the two-element off/on values tensor of the layer.</param>
        /// <param name="axis">The axis along which the layer adds the one-hot representation.</param>
        public OneHot(string name, string indices, string depth, string values, int axis)
        {
            this.name = name;
            inputs = new[] { indices, depth, values };
            this.axis = axis;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var dataType = inputTensors[2].dataType;
            var shapeX = inputTensors[0].shape;
            if (!shapeX.hasRank)
                return new PartialTensor(dataType);

            var shapeOut = shapeX.Unsqueeze(axis);
            shapeOut[axis] = (SymbolicTensorDim)inputTensors[1][0];

            return new PartialTensor(dataType, shapeOut);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var depth = inputTensors[1].ToReadOnlySpan<int>()[0];
            var O = ctx.backend.NewOutputTensor(ShapeInference.OneHot(inputTensors[0].shape, axis, depth), inputTensors[2].dataType);
            if (O.shape.HasZeroDims())
                return O;
            if (inputTensors[2].dataType == DataType.Int)
            {
                var values = inputTensors[2].ToReadOnlySpan<int>();
                ctx.backend.OneHot(inputTensors[0] as TensorInt, O as TensorInt, axis, depth, values[0], values[1]);
            }
            else
            {
                var values = inputTensors[2].ToReadOnlySpan<float>();
                ctx.backend.OneHot(inputTensors[0] as TensorInt, O as TensorFloat, axis, depth, values[0], values[1]);
            }
            return O;
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, axis: {axis}";
        }

        internal override string profilerTag => "OneHot";
    }

    /// <summary>
    /// Represents a `Range` layer. This generates a 1D output tensor where the values form an arithmetic progression defined by the `start`, `limit` and `delta` scalar input tensors.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(0, 1, 2)]
    public class Range : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `Range` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="start">The name to use for the scalar start value tensor of the layer.</param>
        /// <param name="limit">The name to use for the scalar limit value tensor of the layer.</param>
        /// <param name="delta">The name to use for the scalar delta value tensor of the layer.</param>
        public Range(string name, string start, string limit, string delta)
        {
            this.name = name;
            this.inputs = new[] { start, limit, delta };
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var start = inputTensors[0];
            var limit = inputTensors[1];
            var delta = inputTensors[2];

            inputTensors[0].shape.DeclareRank(0);
            inputTensors[1].shape.DeclareRank(0);
            inputTensors[2].shape.DeclareRank(0);

            var shape = SymbolicTensorShape.UnknownOfRank(1);

            if (start[0] == 0 && delta[0] == 1)
                shape[0] = (SymbolicTensorDim)limit[0];

            return new PartialTensor(inputTensors[0].dataType, shape);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            if (inputTensors[0] is TensorInt)
            {
                int start = inputTensors[0].ToReadOnlySpan<int>()[0];
                int limit = inputTensors[1].ToReadOnlySpan<int>()[0];
                int delta = inputTensors[2].ToReadOnlySpan<int>()[0];
                var O = ctx.backend.NewOutputTensorInt(ShapeInference.Range(start, limit, delta));
                if (O.shape.HasZeroDims())
                    return O;
                ctx.backend.Range(O, start, delta);
                return O;
            }
            else
            {
                float start = inputTensors[0].ToReadOnlySpan<float>()[0];
                float limit = inputTensors[1].ToReadOnlySpan<float>()[0];
                float delta = inputTensors[2].ToReadOnlySpan<float>()[0];
                var O = ctx.backend.NewOutputTensorFloat(ShapeInference.Range(start, limit, delta));
                if (O.shape.HasZeroDims())
                    return O;
                ctx.backend.Range(O, start, delta);
                return O;
            }
        }

        internal override string profilerTag => "Range";
    }
}
