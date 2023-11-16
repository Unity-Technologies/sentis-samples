using System;
using UnityEngine;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents a `Shape` layer. This computes the shape of an input tensor as a 1D `TensorInt`.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.NoDataDependencyInputs(0)]
    public class Shape : Layer
    {
        /// <summary>
        /// The inclusive start axis for slicing the shape of the input tensor.
        ///
        /// If this is negative then the axes of the tensor are counted from the back.
        /// </summary>
        public int start;
        /// <summary>
        /// The exclusive end axis for slicing the shape of the input tensor.
        ///
        /// If this is negative then the dimensions of the tensor are counted from the back.
        /// </summary>
        public int end;

        /// <summary>
        /// Initializes and returns an instance of `Shape` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer. The layer does not use the values of this tensor in the computation.</param>
        /// <param name="start">The inclusive start axis for slicing the shape of the input tensor. The default value is 0.</param>
        /// <param name="end">The exclusive end axis for slicing the shape of the input tensor. The default value is 8.</param>
        public Shape(string name, string input, int start = 0, int end = TensorShape.maxRank)
        {
            this.name = name;
            inputs = new[] { input };
            this.start = start;
            this.end = end;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            if (start == end)
                return new PartialTensor(DataType.Int, new SymbolicTensorShape(new SymbolicTensorDim(0)));

            var shapeX = inputTensors[0].shape;

            if (!shapeX.hasRank)
                return new PartialTensor(DataType.Int, SymbolicTensorShape.UnknownOfRank(1));

            var startX = start < 0 ? start + shapeX.rank : start;
            var endX = end < 0 ? end + shapeX.rank : end;
            startX = Mathf.Clamp(startX, 0, shapeX.rank);
            endX = Mathf.Clamp(endX, 0, shapeX.rank);

            Logger.AssertIsTrue(endX >= startX, "PartialTensorFromSymbolicShape.InputError: start value cannot be greater than end value for shape slicing");

            var tensorOut = new PartialTensor(DataType.Int, new SymbolicTensorShape(endX - startX));
            for (var i = startX; i < endX; i++)
            {
                tensorOut[i - startX] = (PartialTensorElement)shapeX[i];
            }

            return tensorOut;
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var shapeX = inputTensors[0].shape;
            var startX = start < 0 ? start + shapeX.rank : start;
            var endX = end < 0 ? end + shapeX.rank : end;
            startX = Mathf.Clamp(startX, 0, shapeX.rank);
            endX = Mathf.Clamp(endX, 0, shapeX.rank);

            Logger.AssertIsTrue(endX >= startX, "Shape.InputError: start value cannot be greater than end value for shape slicing");
            var O = ctx.backend.NewOutputTensorInt(new TensorShape(endX - startX));
            ArrayTensorData.Pin(O, clearOnInit: false);
            for (var i = startX; i < endX; i++)
                O[i - startX] = shapeX[i];
            return O;
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, start: {start}, end: {end}";
        }

        internal override string profilerTag => "Shape";
    }

    /// <summary>
    /// Represents a `Size` layer. This computes the number of elements of an input tensor as a scalar `TensorInt`.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.NoDataDependencyInputs(0)]
    public class Size : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `Size` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer. The layer does not use the values of this tensor in the computation.</param>
        public Size(string name, string input)
        {
            this.name = name;
            inputs = new[] { input };
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            return new PartialTensor(DataType.Int, new SymbolicTensorShape())
            {
                [0] = (PartialTensorElement)inputTensors[0].shape.Length()
            };
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorInt(new TensorShape());
            ArrayTensorData.Pin(O, clearOnInit: false);
            O[0] = inputTensors[0].shape.length;
            return O;
        }

        internal override string profilerTag => "Size";
    }
}
