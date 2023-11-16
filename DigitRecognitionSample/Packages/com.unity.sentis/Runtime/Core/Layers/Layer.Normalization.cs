using System;
using UnityEngine;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents an element-wise `ScaleBias` normalization layer: f(x, s, b) = x * s + b.
    /// </summary>
    [Serializable]
    public class ScaleBias : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `ScaleBias` normalization layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="scale">The name to use for the scale tensor of the layer.</param>
        /// <param name="bias">The name to use for the bias tensor of the layer.</param>
        public ScaleBias(string name, string input, string scale, string bias)
        {
            this.name = name;
            inputs = new[] { input, scale, bias };
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var dataType = inputTensors[0].dataType;
            var shapeX = inputTensors[0].shape;
            var shapeScale = inputTensors[1].shape;
            var shapeBias = inputTensors[2].shape;
            var c = SymbolicTensorDim.Unknown;
            shapeScale.DeclareRank(1);
            c = SymbolicTensorDim.MaxDefinedDim(c, shapeScale[0]);
            shapeBias.DeclareRank(1);
            c = SymbolicTensorDim.MaxDefinedDim(c, shapeBias[0]);
            if (!shapeX.hasRank)
                return new PartialTensor(dataType);

            Logger.AssertIsTrue(shapeX.hasRank ? shapeX.rank >= 2 : true, "RankError: incorrect rank, expecting at least {0}, got {1}", 2, shapeX.rank);

            var shapeOut = new SymbolicTensorShape(shapeX);
            shapeOut[1] = SymbolicTensorDim.MaxDefinedDim(shapeOut[1], c);
            return new PartialTensor(dataType, shapeOut);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorFloat(inputTensors[0].shape);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.ScaleBias(inputTensors[0] as TensorFloat, inputTensors[1] as TensorFloat, inputTensors[2] as TensorFloat, O);
            return O;
        }

        internal override string profilerTag => "ScaleBias";
    }

    /// <summary>
    /// Represents an `InstanceNormalization` normalization layer. This computes the mean variance on the spatial dims of the input tensor and normalizes them according to `scale` and `bias` tensors.
    /// </summary>
    [Serializable]
    public class InstanceNormalization : Layer
    {
        /// <summary>
        /// The epsilon value the layer uses to avoid division by zero.
        /// </summary>
        public float epsilon;

        /// <summary>
        /// Initializes and returns an instance of `InstanceNormalization` normalization layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="scale">The name to use for the scale tensor of the layer.</param>
        /// <param name="bias">The name to use for the bias tensor of the layer.</param>
        /// <param name="epsilon">The epsilon value the layer uses to avoid division by zero. The default value is 1e-5f.</param>
        public InstanceNormalization(string name, string input, string scale, string bias, float epsilon = 1e-5f)
        {
            this.name = name;
            inputs = new[] { input, scale, bias };
            if (epsilon == 0)
                epsilon = Mathf.Epsilon; // safety check to prevent division by zero
            this.epsilon = epsilon;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var dataType = inputTensors[0].dataType;
            var shapeX = inputTensors[0].shape;
            var shapeScale = inputTensors[1].shape;
            var shapeBias = inputTensors[2].shape;
            var c = SymbolicTensorDim.Unknown;
            shapeScale.DeclareRank(1);
            c = SymbolicTensorDim.MaxDefinedDim(c, shapeScale[0]);
            shapeBias.DeclareRank(1);
            c = SymbolicTensorDim.MaxDefinedDim(c, shapeBias[0]);
            if (!shapeX.hasRank)
                return new PartialTensor(dataType);

            Logger.AssertIsTrue(shapeX.hasRank ? shapeX.rank >= 2 : true, "RankError: incorrect rank, expecting at least {0}, got {1}", 2, shapeX.rank);
            shapeScale.DeclareRank(1);

            var shapeOut = new SymbolicTensorShape(shapeX);
            shapeOut[1] = SymbolicTensorDim.MaxDefinedDim(shapeOut[1], c);
            return new PartialTensor(dataType, shapeOut);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            // @TODO: support other types of Normalization at test time.
            // Currently supported only pool=1 (InstanceNormalization)
            var O = ctx.backend.NewOutputTensorFloat(inputTensors[0].shape);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.InstanceNormalization(inputTensors[0] as TensorFloat, inputTensors[1] as TensorFloat, inputTensors[2] as TensorFloat, O, epsilon);
            return O;
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, epsilon: {epsilon}";
        }

        internal override string profilerTag => "Normalization";
    }

    /// <summary>
    /// Represents an `LayerNormalization` normalization layer. This computes the mean variance on the last dimension of the input tensor and normalizes it according to `scale` and `bias` tensors.
    /// </summary>
    [Serializable]
    public class LayerNormalization : Layer
    {
        /// <summary>
        /// The epsilon value the layer uses to avoid division by zero.
        /// </summary>
        public float epsilon;

        /// <summary>
        /// Initializes and returns an instance of `LayerNormalization` normalization layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="scale">The name to use for the scale tensor of the layer.</param>
        /// <param name="bias">The name to use for the bias tensor of the layer.</param>
        /// <param name="epsilon">The epsilon value the layer uses to avoid division by zero. The default value is 1e-5f.</param>
        public LayerNormalization(string name, string input, string scale, string bias, float epsilon = 1e-5f)
        {
            this.name = name;
            inputs = new[] { input, scale, bias };

            if (epsilon == 0)
                epsilon = Mathf.Epsilon; // safety check to prevent division by zero
            this.epsilon = epsilon;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var shapeX = inputTensors[0].shape;

            if (!shapeX.hasRank)
                return new PartialTensor(inputTensors[0].dataType, SymbolicTensorShape.UnknownShape);

            Logger.AssertIsTrue(shapeX.rank >= 1, "RankError: incorrect rank, expecting at least {0}, got {1}", 1, shapeX.rank);

            var shapeScale = inputTensors[1].shape;
            var shapeBias = inputTensors[2].shape;
            shapeScale.DeclareRank(1);
            shapeBias.DeclareRank(1);

            var shape = new SymbolicTensorShape(shapeX);
            shape[-1] = SymbolicTensorDim.MaxDefinedDim(shape[-1], SymbolicTensorDim.MaxDefinedDim(shapeScale[0], shapeBias[0]));
            return new PartialTensor(inputTensors[0].dataType, shape);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorFloat(inputTensors[0].shape);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.LayerNormalization(inputTensors[0] as TensorFloat, inputTensors[1] as TensorFloat, inputTensors[2] as TensorFloat, O, epsilon);
            return O;
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, epsilon: {epsilon}";
        }

        internal override string profilerTag => "LayerNormalization";
    }

    /// <summary>
    /// Represents an `BatchNormalization` normalization layer. This computes the mean variance on the second dimension of the input tensor and normalizes it according to `scale` and `bias` tensors.
    /// </summary>
    [Serializable]
    public class BatchNormalization : Layer
    {
        /// <summary>
        /// The epsilon value the layer uses to avoid division by zero.
        /// </summary>
        public float epsilon;

        /// <summary>
        /// Initializes and returns an instance of `BatchNormalization` normalization layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="scale">The name to use for the scale tensor of the layer.</param>
        /// <param name="bias">The name to use for the bias tensor of the layer.</param>
        /// <param name="mean">The name to use for the mean tensor of the layer.</param>
        /// <param name="variance">The name to use for the variance tensor of the layer.</param>
        /// <param name="epsilon">The epsilon value the layer uses to avoid division by zero. The default value is 1e-5f.</param>
        public BatchNormalization(string name, string input, string scale, string bias, string mean, string variance, float epsilon = 1e-5f)
        {
            this.name = name;
            inputs = new[] { input, scale, bias, mean, variance };
            this.epsilon = epsilon;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var shapeX = inputTensors[0].shape;

            if (!shapeX.hasRank)
                return new PartialTensor(inputTensors[0].dataType, SymbolicTensorShape.UnknownShape);

            Logger.AssertIsTrue(shapeX.rank >= 1, "RankError: incorrect rank, expecting at least {0}, got {1}", 1, shapeX.rank);

            SymbolicTensorShape shapeScale = inputTensors[1].shape, shapeBias = inputTensors[2].shape, shapeMean = inputTensors[3].shape, shapeVar = inputTensors[4].shape;
            shapeScale.DeclareRank(1);
            shapeBias.DeclareRank(1);
            shapeMean.DeclareRank(1);
            shapeVar.DeclareRank(1);

            var shape = new SymbolicTensorShape(shapeX);
            if (shapeX.rank > 1)
                shape[1] = SymbolicTensorDim.MaxDefinedDim(shape[1], SymbolicTensorDim.MaxDefinedDim(shapeScale[0], SymbolicTensorDim.MaxDefinedDim(shapeBias[0], SymbolicTensorDim.MaxDefinedDim(shapeMean[0], shapeVar[0]))));
            return new PartialTensor(inputTensors[0].dataType, shape);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorFloat(inputTensors[0].shape);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.BatchNormalization(inputTensors[0] as TensorFloat, inputTensors[1] as TensorFloat, inputTensors[2] as TensorFloat, inputTensors[3] as TensorFloat, inputTensors[4] as TensorFloat, O, epsilon);
            return O;
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, epsilon: {epsilon}";
        }

        internal override string profilerTag => "BatchNormalization";
    }

    /// <summary>
    /// Represents an `LRN` local response normalization layer. This normalizes the input tensor over local input regions.
    /// </summary>
    [Serializable]
    public class LRN : Layer
    {
        /// <summary>
        /// The scaling parameter to use for the normalization.
        /// </summary>
        public float alpha;
        /// <summary>
        /// The exponent to use for the normalization.
        /// </summary>
        public float beta;
        /// <summary>
        /// The bias value to use for the normalization.
        /// </summary>
        public float bias;
        /// <summary>
        /// The number of channels to sum over.
        /// </summary>
        public int count;

        /// <summary>
        /// Initializes and returns an instance of `LRN` local response normalization layer layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="alpha">The scaling parameter to use for the normalization.</param>
        /// <param name="beta">The exponent to use for the normalization.</param>
        /// <param name="bias">The bias value to use for the normalization.</param>
        /// <param name="count">The number of channels to sum over.</param>
        public LRN(string name, string input, float alpha, float beta, float bias, int count)
        {
            this.name = name;
            inputs = new[] { input };
            this.alpha = alpha;
            this.beta = beta;
            this.bias = bias;
            this.count = count;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var shapeX = inputTensors[0].shape;
            Logger.AssertIsTrue(shapeX.hasRank ? shapeX.rank >= 2 : true, "RankError: incorrect rank, expecting at least {0}, got {1}", 2, shapeX.rank);

            return new PartialTensor(inputTensors[0].dataType, shapeX);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorFloat(inputTensors[0].shape);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.LRN(inputTensors[0] as TensorFloat, O, alpha, beta, bias, count);
            return O;
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, alpha: {alpha}, beta: {beta}, bias: {bias}, count: {count}";
        }

        internal override string profilerTag => "LRN";
    }
}
