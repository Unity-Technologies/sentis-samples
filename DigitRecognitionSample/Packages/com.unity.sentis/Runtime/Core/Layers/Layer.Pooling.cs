using System;
using UnityEngine;
using UnityEngine.Serialization;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Infer the output partial tensor from the input partial tensors.
    ///
    /// If the layer has more than one output, output partial tensors are saved to 'ctx'.
    /// </summary>
    [Serializable]
    public abstract class LocalPool : Layer
    {
        /// <summary>
        /// The size of the kernel along each spatial axis.
        /// </summary>
        public int[] kernelShape;
        /// <summary>
        /// The stride along each spatial axis.
        ///
        /// If this is `null` the layer uses a default of [1, 1, ..., 1].
        /// </summary>
        public int[] strides;
        /// <summary>
        /// The lower and upper padding values for each spatial dimension. For example [pad_left, pad_right] for 1D, or [pad_top, pad_bottom, pad_left, pad_right] for 2D.
        ///
        /// If this is `null` the layer uses a default of [0, 0, ..., 0].
        /// </summary>
        public int[] pads;
        /// <summary>
        /// The auto padding mode of the pool as an `AutoPad`.
        /// </summary>
        public AutoPad autopad;

        /// <summary>
        /// Initializes and returns an instance of `AveragePool` pooling layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="kernelShape">The size of the kernel along each spatial axis.</param>
        /// <param name="strides">The stride along each spatial axis.</param>
        /// <param name="pads">The lower and upper padding values for each spatial dimension, [pad_left, pad_right] for 1D, [pad_top, pad_bottom, pad_left, pad_right] for 2D, etc.</param>
        /// <param name="autopad">The auto padding mode of the pool as an `AutoPad`. The default value is `AutoPad.NotSet`.</param>
        public LocalPool(string name, string input, int[] kernelShape, int[] strides, int[] pads, AutoPad autopad = AutoPad.NotSet)
        {
            this.name = name;
            inputs = new[] { input };
            this.kernelShape = kernelShape;
            this.strides = strides;
            if (this.strides == null)
            {
                this.strides = new int[this.kernelShape.Length];
                for (var i = 0; i < this.strides.Length; i++)
                {
                    this.strides[i] = 1;
                }
            }
            this.pads = pads;
            this.pads ??= new int[2 * this.kernelShape.Length];
            this.autopad = autopad;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var dataType = inputTensors[0].dataType;
            var shapeX = inputTensors[0].shape;
            shapeX.DeclareRank(2 + kernelShape.Length);

            Logger.AssertIsTrue(strides == null || shapeX.rank - 2 == strides.Length, "Pool.InputError: strides must have same number of values as spatial dimensions or be null");
            Logger.AssertIsTrue(pads == null || (shapeX.rank - 2) * 2 == pads.Length, "Pool.InputError: padding must have twice the number of values as spatial dimensions or be null");

            var shapeOut = new SymbolicTensorShape(shapeX);

            for (var i = 2; i < shapeOut.rank; i++)
            {
                var s = strides == null ? 1 : strides[i - 2];
                var p = (pads == null || autopad != AutoPad.NotSet) ? 0 : (pads[i - 2] + pads[i - 2 + (shapeX.rank - 2)]);
                shapeOut[i] = shapeX[i].Pool(kernelShape[i - 2], s, p, 1, false, autopad);
            }

            return new PartialTensor(dataType, shapeOut);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, kernelShape: [{string.Join(", ", kernelShape)}], strides: [{string.Join(", ", strides)}], pads: [{string.Join(", ", pads)}], autopad: {autopad}";
        }
    }

    /// <summary>
    /// Represents a global pooling layer.
    /// </summary>
    [Serializable]
    public abstract class GlobalPool : Layer
    {
        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var dataType = inputTensors[0].dataType;
            var shapeX = inputTensors[0].shape;
            if (!shapeX.hasRank)
                return new PartialTensor(dataType);

            Logger.AssertIsTrue(shapeX.hasRank ? shapeX.rank >= 3 : true, "RankError: incorrect rank, expecting at least {0}, got {1}", 3, shapeX.rank);

            var shapeOut = new SymbolicTensorShape(shapeX);

            for (var i = 2; i < shapeOut.rank; i++)
            {
                shapeOut[i] = SymbolicTensorDim.One;
            }

            return new PartialTensor(dataType, shapeOut);
        }
    }

    /// <summary>
    /// Represents an `AveragePool` pooling layer. This calculates an output tensor by pooling the mean values of the input tensor across its spatial dimensions according to the given pool and stride values.
    /// </summary>
    [Serializable]
    public class AveragePool : LocalPool
    {
        /// <summary>
        /// Initializes and returns an instance of `AveragePool` pooling layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="kernelShape">The size of the kernel along each spatial axis.</param>
        /// <param name="strides">The stride along each spatial axis.</param>
        /// <param name="pads">The lower and upper padding values for each spatial dimension, [pad_left, pad_right] for 1D, [pad_top, pad_bottom, pad_left, pad_right] for 2D, etc.</param>
        /// <param name="autopad">The auto padding mode of the pool as an `AutoPad`. The default value is `AutoPad.NotSet`.</param>
        public AveragePool(string name, string input, int[] kernelShape, int[] strides, int[] pads, AutoPad autopad = AutoPad.NotSet)
            : base(name, input, kernelShape, strides, pads, autopad) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            ShapeInference.UpdatePadForPoolAutoPadding(inputTensors[0].shape, kernelShape, strides, pads, false, autopad);
            var O = ctx.backend.NewOutputTensorFloat(ShapeInference.ApplyPool(inputTensors[0].shape, kernelShape, strides, pads));
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.AveragePool(inputTensors[0] as TensorFloat, O, kernelShape, strides, pads);
            return O;
        }

        internal override string profilerTag => "AveragePool";
    }

    /// <summary>
    /// Represents a `GlobalAveragePool` pooling layer. This calculates an output tensor by pooling the mean values of the input tensor across all of its spatial dimensions. The spatial dimensions of the output are size 1.
    /// </summary>
    [Serializable]
    public class GlobalAveragePool : GlobalPool
    {
        /// <summary>
        /// Initializes and returns an instance of `GlobalAveragePool` pooling layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public GlobalAveragePool(string name, string input)
        {
            this.name = name;
            inputs = new[] { input };
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorFloat(ShapeInference.GlobalPool(inputTensors[0].shape));
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.GlobalAveragePool(inputTensors[0] as TensorFloat, O);
            return O;
        }

        internal override string profilerTag => "GlobalAveragePool";
    }

    /// <summary>
    /// Represents a `GlobalMaxPool` pooling layer. This calculates an output tensor by pooling the maximum values of the input tensor across all of its spatial dimensions. The spatial dimensions of the output are size 1.
    /// </summary>
    [Serializable]
    public class GlobalMaxPool : GlobalPool
    {
        /// <summary>
        /// Initializes and returns an instance of `GlobalMaxPool` pooling layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public GlobalMaxPool(string name, string input)
        {
            this.name = name;
            inputs = new[] { input };
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorFloat(ShapeInference.GlobalPool(inputTensors[0].shape));
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.GlobalMaxPool(inputTensors[0] as TensorFloat, O);
            return O;
        }

        internal override string profilerTag => "GlobalMaxPool";
    }

    /// <summary>
    /// Represents a `MaxPool` pooling layer. This calculates an output tensor by pooling the maximum values of the input tensor across its spatial dimensions according to the given pool and stride values.
    /// </summary>
    [Serializable]
    public class MaxPool : LocalPool
    {
        /// <summary>
        /// Initializes and returns an instance of `MaxPool` pooling layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="kernelShape">The size of the kernel along each spatial axis.</param>
        /// <param name="strides">The stride along each spatial axis.</param>
        /// <param name="pads">The lower and upper padding values for each spatial dimension. For example [pad_left, pad_right] for 1D, or [pad_top, pad_bottom, pad_left, pad_right] for 2D.</param>
        /// <param name="autopad">The auto padding mode of the pool as an `AutoPad`. The default value is `AutoPad.NotSet`.</param>
        public MaxPool(string name, string input, int[] kernelShape, int[] strides, int[] pads, AutoPad autopad = AutoPad.NotSet)
            : base(name, input, kernelShape, strides, pads, autopad) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            ShapeInference.UpdatePadForPoolAutoPadding(inputTensors[0].shape, kernelShape, strides, pads, false, autopad);
            var O = ctx.backend.NewOutputTensorFloat(ShapeInference.ApplyPool(inputTensors[0].shape, kernelShape, strides, pads));
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.MaxPool(inputTensors[0] as TensorFloat, O, kernelShape, strides, pads);
            return O;
        }

        internal override string profilerTag => "MaxPool";
    }
}
