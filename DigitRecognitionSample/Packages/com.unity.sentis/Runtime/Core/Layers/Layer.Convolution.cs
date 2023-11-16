using System;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Options for auto padding in image layers.
    /// </summary>
    public enum AutoPad
    {
        /// <summary>
        /// Use explicit padding.
        /// </summary>
        NotSet,
        /// <summary>
        /// Use no padding.
        /// </summary>
        Valid,
        /// <summary>
        /// Use equal or almost equal padding on both sides. When the padding is odd, add the extra value to the end.
        /// </summary>
        SameUpper,
        /// <summary>
        /// Use equal or almost equal padding on both sides. When the padding is odd, add the extra value to the start.
        /// </summary>
        SameLower,
    }

    /// <summary>
    /// Represents a `Conv` convolution layer, which applies a convolution filter to an input tensor.
    /// </summary>
    [Serializable]
    public class Conv : FusedActivation
    {
        /// <summary>
        /// The auto padding mode of the convolution as an `AutoPad`.
        /// </summary>
        public AutoPad autoPad;
        /// <summary>
        /// The dilation value of each spatial dimension of the filter.
        ///
        /// If this is `null` the layer uses a default of [1, 1, ..., 1].
        /// </summary>
        public int[] dilations;
        /// <summary>
        /// The number of groups that input channels and output channels are divided into.
        /// </summary>
        public int group;
        /// <summary>
        /// The lower and upper padding values for each spatial dimension of the filter, [pad_left, pad_right] for 1D, [pad_top, pad_bottom, pad_left, pad_right] for 2D, etc.
        ///
        /// If this is `null` the layer uses a default of [0, 0, ..., 0].
        /// </summary>
        public int[] pads;
        /// <summary>
        /// The stride value for each spatial dimension of the filter.
        ///
        /// If this is `null` the layer uses a default of [1, 1, ..., 1].
        /// </summary>
        public int[] strides;
        /// <summary>
        /// The shape of the kernel as an integer array.
        /// </summary>
        public int[] kernelShape;

        /// <summary>
        /// Initializes and returns an instance of `Conv` convolution layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="X">The name to use for the input tensor of the layer.</param>
        /// <param name="W">The name to use for the filter tensor of the layer.</param>
        /// <param name="B">The name to use for the optional bias tensor of the layer.</param>
        /// <param name="group">The number of groups that input channels and output channels are divided into.</param>
        /// <param name="strides">The optional stride value for each spatial dimension of the filter.</param>
        /// <param name="pads">The optional lower and upper padding values for each spatial dimension of the filter.</param>
        /// <param name="dilations">The optional dilation value of each spatial dimension of the filter.</param>
        /// <param name="autoPad">The auto padding mode of the convolution as an `AutoPad`.</param>
        /// <param name="kernelShape">The shape of the kernel as an integer array.</param>
        /// <param name="fusedActivation">The fused activation type to apply after the convolution. The default value is `None`.</param>
        public Conv(string name, string X, string W, string B, int group, int[] strides, int[] pads, int[] dilations, AutoPad autoPad = AutoPad.NotSet, int[] kernelShape = null, FusableActivation fusedActivation = FusableActivation.None)
        {
            this.name = name;
            inputs = new[] { X, W, B };
            this.autoPad = autoPad;
            this.dilations = dilations ?? new[] { 1, 1, 1, 1, 1, 1 };
            this.group = group;
            this.pads = pads ?? new int[12];
            this.strides = strides ?? new[] { 1, 1, 1, 1, 1, 1 };
            this.kernelShape = kernelShape;
            this.fusedActivation = fusedActivation;
        }

        /// <summary>
        /// Initializes and returns an instance of `Conv` convolution layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="X">The name to use for the input tensor of the layer.</param>
        /// <param name="W">The name to use for the filter tensor of the layer.</param>
        /// <param name="group">The number of groups that input channels and output channels are divided into.</param>
        /// <param name="strides">The optional stride value for each spatial dimension of the filter.</param>
        /// <param name="pads">The optional lower and upper padding values for each spatial dimension of the filter.</param>
        /// <param name="dilations">The optional dilation value of each spatial dimension of the filter.</param>
        /// <param name="autoPad">The auto padding mode of the convolution as an `AutoPad`.</param>
        /// <param name="kernelShape">The shape of the kernel as an integer array.</param>
        /// <param name="fusedActivation">The fused activation type to apply after the convolution. The default value is `None`.</param>
        public Conv(string name, string X, string W, int group, int[] strides, int[] pads, int[] dilations, AutoPad autoPad = AutoPad.NotSet, int[] kernelShape = null, FusableActivation fusedActivation = FusableActivation.None)
        {
            this.name = name;
            inputs = new[] { X, W };
            this.autoPad = autoPad;
            this.dilations = dilations ?? new[] { 1, 1, 1, 1, 1, 1 };
            this.group = group;
            this.pads = pads ?? new int[12];
            this.strides = strides ?? new[] { 1, 1, 1, 1, 1, 1 };
            this.kernelShape = kernelShape;
            this.fusedActivation = fusedActivation;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var shapeX = inputTensors[0].shape;
            var shapeKernel = inputTensors[1].shape;
            for (var i = 0; kernelShape != null && i < kernelShape.Length; i++)
            {
                shapeKernel[i + 2] = SymbolicTensorDim.MaxDefinedDim(shapeKernel[i + 2], new SymbolicTensorDim(kernelShape[i]));
            }
            if (!shapeX.hasRank)
                return new PartialTensor(DataType.Float);

            shapeKernel.DeclareRank(shapeX.rank);

            var shapeOut = SymbolicTensorShape.UnknownOfRank(shapeX.rank);

            shapeOut[0] = shapeX[0];
            shapeOut[1] = shapeKernel[0];

            if (inputs.Length == 3)
            {
                var shapeBias = inputTensors[2]?.shape ?? SymbolicTensorShape.UnknownShape;
                shapeBias.DeclareRank(1);
                shapeOut[1] = SymbolicTensorDim.MaxDefinedDim(shapeOut[1], shapeBias[0]);
            }

            for (var i = 2; i < shapeOut.rank; i++)
            {
                var stride = strides == null ? 1 : strides[i - 2];
                var pad = pads == null || autoPad != AutoPad.NotSet ? 0 : pads[i - 2] + pads[i - 2 + (shapeX.rank - 2)];
                var dilation = dilations == null ? 1 : dilations[i - 2];
                var dimX = shapeX[i];
                var dimKernel = shapeKernel[i];
                if (dimKernel.isValue)
                    shapeOut[i] = dimX.Pool(dimKernel.value, stride, pad, dilation, false, autoPad);
                else if (dimKernel.isParam && (autoPad is AutoPad.SameLower || autoPad is AutoPad.SameUpper))
                    shapeOut[i] = dimX.Pool(0, stride, pad, dilation, false, autoPad);
                else
                    shapeOut[i] = SymbolicTensorDim.Unknown;
            }

            return new PartialTensor(DataType.Float, shapeOut);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var numSpatialDims = inputTensors[0].shape.rank - 2;
            var stridesSpan = strides.AsSpan(0, numSpatialDims);
            var padsSpan = pads.AsSpan(0, 2 * numSpatialDims);
            var dilationsSpan = dilations.AsSpan(0, numSpatialDims);
            ShapeInference.UpdatePadForConvAutoPadding(inputTensors[0].shape, inputTensors[1].shape, stridesSpan, dilationsSpan, autoPad, padsSpan);
            var shapeO = ShapeInference.Conv(inputTensors[0].shape, inputTensors[1].shape, group, stridesSpan, padsSpan, dilationsSpan);
            var O = ctx.backend.NewOutputTensorFloat(shapeO);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.Conv(inputTensors[0] as TensorFloat, inputTensors[1] as TensorFloat, inputTensors.Length > 2 ? inputTensors[2] as TensorFloat : null, O, group, stridesSpan, padsSpan, dilationsSpan, fusedActivation);
            return O;
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, group: {group}, strides: [{(string.Join(", ", strides))}], pads: [{(string.Join(", ", pads))}], dilations: [{(string.Join(", ", dilations))}], autoPad: {autoPad}, kernelShape: [{(kernelShape == null ? "null" : string.Join(", ", kernelShape))}], fusedActivation: {fusedActivation}";
        }

        internal override string profilerTag => "Conv";
    }

    /// <summary>
    /// Represents a `ConvTranspose` transpose convolution layer, which applies a convolution filter to an input tensor.
    /// </summary>
    [Serializable]
    public class ConvTranspose : FusedActivation
    {
        /// <summary>
        /// The auto padding mode of the transpose convolution.
        /// </summary>
        public AutoPad autoPad;
        /// <summary>
        /// The output padding value for each spatial dimension in the filter.
        ///
        /// The layer adds the output padding to the side with higher coordinate indices in the output tensor.
        ///
        /// If this is `null` the layer uses a default of [0, 0, ..., 0].
        /// </summary>
        public int[] outputPadding;
        /// <summary>
        /// The lower and upper padding values for each spatial dimension of the filter. For example [pad_left, pad_right] for 1D, or [pad_top, pad_bottom, pad_left, pad_right] for 2D.
        ///
        /// If this is `null` the layer uses a default of [0, 0, ..., 0].
        /// </summary>
        public int[] pads;
        /// <summary>
        /// The stride value for each spatial dimension of the filter.
        ///
        /// If this is `null` the layer uses a default of [1, 1, ..., 1].
        /// </summary>
        public int[] strides;
        /// <summary>
        /// The shape of the kernel.
        /// </summary>
        public int[] kernelShape;

        /// <summary>
        /// Initializes and returns an instance of `ConvTranspose` convolution layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="kernel">The name to use for the filter tensor of the layer.</param>
        /// <param name="bias">The name to use for the optional bias tensor of the layer.</param>
        /// <param name="strides">The optional stride value for each spatial dimension of the filter.</param>
        /// <param name="pads">The optional lower and upper padding values for each spatial dimension of the filter.</param>
        /// <param name="autoPad">The auto padding mode of the convolution.</param>
        /// <param name="outputPadding">The output padding value for each spatial dimension in the filter.</param>
        /// <param name="kernelShape">The shape of the kernel as an integer array.</param>
        /// <param name="fusedActivation">The fused activation type to apply after the convolution. The default value is `None`.</param>
        public ConvTranspose(string name, string input, string kernel, string bias, int[] strides, int[] pads, AutoPad autoPad, int[] outputPadding, int[] kernelShape = null, FusableActivation fusedActivation = FusableActivation.None)
        {
            this.name = name;
            inputs = new[] { input, kernel, bias };
            this.autoPad = autoPad;
            this.outputPadding = outputPadding ?? new int[6];
            this.pads = pads ?? new int[12];
            this.strides = strides ?? new[] { 1, 1, 1, 1, 1, 1 };
            this.kernelShape = kernelShape;
            this.fusedActivation = fusedActivation;
        }

        /// <summary>
        /// Initializes and returns an instance of `ConvTranspose` convolution layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="kernel">The name to use for the filter tensor of the layer.</param>
        /// <param name="strides">The optional stride value for each spatial dimension of the filter.</param>
        /// <param name="pads">The optional lower and upper padding values for each spatial dimension of the filter.</param>
        /// <param name="autoPad">The auto padding mode of the convolution.</param>
        /// <param name="outputPadding">The output padding value for each spatial dimension in the filter.</param>
        /// <param name="kernelShape">The shape of the kernel as an integer array.</param>
        /// <param name="fusedActivation">The fused activation type to apply after the convolution. The default value is `None`.</param>
        public ConvTranspose(string name, string input, string kernel, int[] strides, int[] pads, AutoPad autoPad, int[] outputPadding, int[] kernelShape = null, FusableActivation fusedActivation = FusableActivation.None)
        {
            this.name = name;
            inputs = new[] { input, kernel };
            this.autoPad = autoPad;
            this.outputPadding = outputPadding ?? new int[6];
            this.pads = pads ?? new int[12];
            this.strides = strides ?? new[] { 1, 1, 1, 1, 1, 1 };
            this.kernelShape = kernelShape;
            this.fusedActivation = fusedActivation;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var shapeX = inputTensors[0].shape;
            var shapeKernel = inputTensors[1].shape;
            for (var i = 0; kernelShape != null && i < kernelShape.Length; i++)
            {
                shapeKernel[i + 2] = SymbolicTensorDim.MaxDefinedDim(shapeKernel[i + 2], new SymbolicTensorDim(kernelShape[i]));
            }
            if (!shapeX.hasRank)
                return new PartialTensor(DataType.Float);

            shapeKernel.DeclareRank(shapeX.rank);

            var shapeOut = SymbolicTensorShape.Ones(shapeX.rank);

            shapeOut[0] = shapeX[0];
            shapeOut[1] = shapeKernel[1];

            if (inputs.Length == 3)
            {
                var shapeBias = inputTensors[2]?.shape ?? SymbolicTensorShape.UnknownShape;
                shapeBias.DeclareRank(1);
                shapeOut[1] = SymbolicTensorDim.MaxDefinedDim(shapeOut[1], shapeBias[0]);
            }

            for (var i = 2; i < shapeOut.rank; i++)
            {
                var stride = strides == null ? 1 : strides[i - 2];
                var pad = pads == null || autoPad != AutoPad.NotSet ? 0 : pads[i - 2] + pads[i - 2 + (shapeX.rank - 2)];
                var dilation = 1;
                var outputPad = outputPadding == null ? 0 : outputPadding[i - 2];
                var dimX = shapeX[i];
                var dimKernel = shapeKernel[i];
                if (autoPad == AutoPad.NotSet)
                    shapeOut[i] = stride * (dimX - 1) + outputPad + (dimKernel - 1) * dilation + 1 - pad;
                else
                    shapeOut[i] = dimX * stride;
            }

            return new PartialTensor(DataType.Float, shapeOut);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var numSpatialDims = inputTensors[0].shape.rank - 2;
            var stridesSpan = strides.AsSpan(0, numSpatialDims);
            var padsSpan = pads.AsSpan(0, 2 * numSpatialDims);
            var outputPaddingSpan = outputPadding.AsSpan(0, numSpatialDims);
            ShapeInference.UpdatePadForConvTransAutoPadding(inputTensors[0].shape, inputTensors[1].shape, stridesSpan, autoPad, outputPaddingSpan, padsSpan);
            var shapeO = ShapeInference.ConvTranspose(inputTensors[0].shape, inputTensors[1].shape, stridesSpan, padsSpan, outputPaddingSpan);
            var O = ctx.backend.NewOutputTensorFloat(shapeO);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.ConvTranspose(inputTensors[0] as TensorFloat, inputTensors[1] as TensorFloat, inputTensors.Length > 2 ? inputTensors[2] as TensorFloat : null, O, stridesSpan, padsSpan, outputPaddingSpan, fusedActivation);
            return O;
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, strides: [{(string.Join(", ", strides))}], pads: [{(string.Join(", ", pads))}], outputPadding: [{(string.Join(", ", outputPadding))}], autoPad, {autoPad}, kernelShape: [{(kernelShape == null ? "null" : string.Join(", ", kernelShape))}], fusedActivation: {fusedActivation}";
        }

        internal override string profilerTag => "ConvTranspose";
    }
}
