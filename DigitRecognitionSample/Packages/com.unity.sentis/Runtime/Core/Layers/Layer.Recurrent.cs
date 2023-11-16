using System;
using System.Linq;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Options for the direction of a recurrent layer.
    /// </summary>
    public enum RnnDirection
    {
        /// <summary>
        /// Use only forward direction in the calculation.
        /// </summary>
        Forward = 0,
        /// <summary>
        /// Use only reverse direction in the calculation.
        /// </summary>
        Reverse = 1,
        /// <summary>
        /// Use both forward and reverse directions in the calculation.
        /// </summary>
        Bidirectional = 2,
    }

    /// <summary>
    /// Options for activation functions to apply in a recurrent layer.
    /// </summary>
    public enum RnnActivation
    {
        /// <summary>
        /// Use `Relu` activation: f(x) = max(0, x).
        /// </summary>
        Relu = 0,
        /// <summary>
        /// Use `Tanh` activation: f(x) = (1 - e^{-2x}) / (1 + e^{-2x}).
        /// </summary>
        Tanh = 1,
        /// <summary>
        /// Use `Sigmoid` activation: f(x) = 1 / (1 + e^{-x}).
        /// </summary>
        Sigmoid = 2,
        /// <summary>
        /// Use `Affine` activation: f(x) = alpha * x + beta.
        /// </summary>
        Affine = 3,
        /// <summary>
        /// Use `LeakyRelu` activation: f(x) = x if x >= 0, otherwise f(x) = alpha * x.
        /// </summary>
        LeakyRelu = 4,
        /// <summary>
        /// Use `ThresholdedRelu` activation: f(x) = x if x >= alpha, otherwise f(x) = 0.
        /// </summary>
        ThresholdedRelu = 5,
        /// <summary>
        /// Use `ScaledTanh` activation: f(x) = alpha * tanh(beta * x).
        /// </summary>
        ScaledTanh = 6,
        /// <summary>
        /// Use `HardSigmoid` activation: f(x) = clamp(alpha * x + beta, 0, 1).
        /// </summary>
        HardSigmoid = 7,
        /// <summary>
        /// Use `Elu` activation: f(x) = x if x >= 0, otherwise f(x) = alpha * (e^x - 1).
        /// </summary>
        Elu = 8,
        /// <summary>
        /// Use `Softsign` activation: f(x) = x / (1 + |x|).
        /// </summary>
        Softsign = 9,
        /// <summary>
        /// Use `Softplus` activation: f(x) = log(1 + e^x).
        /// </summary>
        Softplus = 10,
    }

    /// <summary>
    /// Options for the layout of the tensor in a recurrent layer.
    /// </summary>
    public enum RnnLayout
    {
        /// <summary>
        /// Use layout with sequence as the first dimension of the tensors.
        /// </summary>
        SequenceFirst = 0,
        /// <summary>
        /// Use layout with batch as the first dimension of the tensors.
        /// </summary>
        BatchFirst = 1,
    }

    /// <summary>
    /// Represents an `LSTM` recurrent layer. This generates an output tensor by computing a one-layer LSTM (long short-term memory) on an input tensor.
    /// </summary>
    [Serializable]
    public class LSTM : Layer
    {
        /// <summary>
        /// The number of neurons in the hidden layer of the LSTM.
        /// </summary>
        public int hiddenSize;
        /// <summary>
        /// The direction of the LSTM as an `RnnDirection`.
        /// </summary>
        public RnnDirection direction;
        /// <summary>
        /// The activation functions of the LSTM as an array of `RnnActivation`.
        /// </summary>
        public RnnActivation[] activations;
        /// <summary>
        /// The alpha values of the activation functions of the LSTM.
        /// </summary>
        public float[] activationAlpha;
        /// <summary>
        /// The beta values of the activation functions of the LSTM.
        /// </summary>
        public float[] activationBeta;
        /// <summary>
        /// The cell clip threshold of the LSTM.
        /// </summary>
        public float clip;
        /// <summary>
        /// Whether to forget the input values in the LSTM. If this is `false` the input and forget gates are coupled.
        /// </summary>
        public bool inputForget;
        /// <summary>
        /// The layout of the tensors as an `RnnLayout`.
        /// </summary>
        public RnnLayout layout;
        /// <summary>
        /// The number of directions of the LSTM inferred from the `direction`.
        /// </summary>
        public int NumDirections => direction == RnnDirection.Bidirectional ? 2 : 1;

        /// <summary>
        /// Initializes and returns an instance of a Long Short-Term Memory Network (`LSTM`) recurrent layer.
        /// </summary>
        /// <param name="name">The name to use for the first output tensor of the layer.</param>
        /// <param name="inputs">The names to use for the input tensors of the layer [X, W, R, (B, sequenceLens, initialH, initialC, P)].
        ///
        /// X is the name of the input sequences tensor.
        ///
        /// W is the name of the weights tensor for the gates of the LSTM.
        ///
        /// R is the name of the recurrent weights tensor for the gates of the LSTM.
        ///
        /// B is the name of the optional bias tensor for the input gate of the LSTM.
        ///
        /// sequenceLens is the name of the optional 1D tensor specifying the lengths of the sequences in a batch.
        ///
        /// initialH is the name of the optional initial values tensor of the hidden neurons of the LSTM. If this is `null` then 0 is used.
        ///
        /// initialC is the name of the optional initial values tensor of the cells of the LSTM. If this is `null` then 0 is used.
        ///
        /// P is the name of the optional weight tensor for the peepholes of the LSTM. If this is `null` then 0 is used./// </param>
        /// <param name="outputs">The names for the output tensors of the layer [Y, Y_h, Y_c].
        ///
        /// Y is the name of the concatenated intermediate output values tensor of the hidden neurons.
        ///
        /// Y_h is the name of the last output values tensor of the hidden neurons.
        ///
        /// Y_c is the name of the last output values tensor of the cells.</param>
        /// <param name="hiddenSize">The number of neurons in the hidden layer of the LSTM.</param>
        /// <param name="direction">The direction of the LSTM as an `RnnDirection`.</param>
        /// <param name="activations">The activation functions of the LSTM as an array of `RnnActivation`. If this is `null` then the LSTM uses the corresponding defaults for the given activations.</param>
        /// <param name="activationAlpha">The alpha values of the activation functions of the LSTM.
        ///
        /// If this is `null` then the LSTM uses [0, 0, 0...].</param>
        /// <param name="activationBeta">The beta values of the activation functions of the LSTM.
        ///
        /// If this is `null` then the LSTM uses the corresponding defaults for the given activations.</param>
        /// <param name="clip">The cell clip threshold of the LSTM. The default value is `float.MaxValue`.</param>
        /// <param name="inputForget">Whether to forget the input values in the LSTM. If this is `false` the input and forget gates are coupled. The default value is `false`.</param>
        /// <param name="layout">The layout of the tensors as an `RnnLayout`. The default value is RnnLayout.SequenceFirst.</param>
        public LSTM(string name, string[] inputs, string[] outputs, int hiddenSize, RnnDirection direction, RnnActivation[] activations = null, float[] activationAlpha = null, float[] activationBeta = null, float clip = float.MaxValue, bool inputForget = false, RnnLayout layout = RnnLayout.SequenceFirst)
        {
            this.name = name;
            this.inputs = inputs;
            this.outputs = outputs;
            this.hiddenSize = hiddenSize;
            this.direction = direction;
            this.activations = new RnnActivation[3 * NumDirections];
            this.activationAlpha = new float[3 * NumDirections];
            this.activationBeta = new float[3 * NumDirections];
            for (var i = 0; i < 3 * NumDirections; i++)
            {
                this.activations[i] = i % 3 == 0 ? RnnActivation.Sigmoid : RnnActivation.Tanh;
                if (activations != null && i < activations.Length)
                    this.activations[i] = activations[i];
                switch (this.activations[i])
                {
                    case RnnActivation.Affine:
                        this.activationAlpha[i] = 1.0f;
                        break;
                    case RnnActivation.LeakyRelu:
                        this.activationAlpha[i] = 0.01f;
                        break;
                    case RnnActivation.ThresholdedRelu:
                        this.activationAlpha[i] = 1.0f;
                        break;
                    case RnnActivation.ScaledTanh:
                        this.activationAlpha[i] = 1.0f;
                        this.activationBeta[i] = 1.0f;
                        break;
                    case RnnActivation.HardSigmoid:
                        this.activationAlpha[i] = 0.2f;
                        this.activationBeta[i] = 0.5f;
                        break;
                    case RnnActivation.Elu:
                        this.activationAlpha[i] = 1.0f;
                        break;
                }
                if (activationAlpha != null && i < activationAlpha.Length)
                    this.activationAlpha[i] = activationAlpha[i];
                if (activationBeta != null && i < activationBeta.Length)
                    this.activationBeta[i] = activationBeta[i];
            }

            this.clip = clip;
            this.inputForget = inputForget;
            this.layout = layout;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var shapeX = inputTensors[0].shape;
            var shapeW = inputTensors[1].shape;
            var shapeR = inputTensors[2].shape;

            var seqLength = SymbolicTensorDim.Unknown;
            var batchSize = SymbolicTensorDim.Unknown;

            shapeX.DeclareRank(3);
            shapeW.DeclareRank(3);
            shapeR.DeclareRank(3);

            seqLength = SymbolicTensorDim.MaxDefinedDim(seqLength, layout == RnnLayout.SequenceFirst ? shapeX[0] : shapeX[1]);
            batchSize = SymbolicTensorDim.MaxDefinedDim(batchSize, layout == RnnLayout.SequenceFirst ? shapeX[1] : shapeX[0]);

            if (inputTensors.Length > 3 && inputTensors[3] != null && inputTensors[3].shape is var shapeB)
                shapeB.DeclareRank(2);

            if (inputTensors.Length > 4 && inputTensors[4] != null && inputTensors[4].shape is var shapeSequenceLens)
            {
                shapeSequenceLens.DeclareRank(1);
                batchSize = SymbolicTensorDim.MaxDefinedDim(batchSize, shapeSequenceLens[0]);
            }

            if (inputTensors.Length > 5 && inputTensors[5] != null && inputTensors[5].shape is var shapeInitialH)
            {
                shapeInitialH.DeclareRank(3);
                batchSize = SymbolicTensorDim.MaxDefinedDim(batchSize, layout == RnnLayout.SequenceFirst ? shapeInitialH[1] : shapeInitialH[0]);
            }

            if (inputTensors.Length > 6 && inputTensors[6] != null && inputTensors[6].shape is var shapeInitialC)
            {
                shapeInitialC.DeclareRank(3);
                batchSize = SymbolicTensorDim.MaxDefinedDim(batchSize, layout == RnnLayout.SequenceFirst ? shapeInitialC[1] : shapeInitialC[0]);
            }

            if (inputTensors.Length > 7 && inputTensors[7] != null && inputTensors[7].shape is var shapeP)
                shapeP.DeclareRank(2);

            var numDirectionsDim = new SymbolicTensorDim(NumDirections);
            var hiddenSizeDim = new SymbolicTensorDim(hiddenSize);

            if (layout == RnnLayout.SequenceFirst)
            {
                if (outputs.Length > 1 && !string.IsNullOrEmpty(outputs[1]))
                    ctx.AddPartialTensor(outputs[1], new PartialTensor(DataType.Float, new SymbolicTensorShape(numDirectionsDim, batchSize, hiddenSizeDim)));
                if (outputs.Length > 2 && !string.IsNullOrEmpty(outputs[2]))
                    ctx.AddPartialTensor(outputs[2], new PartialTensor(DataType.Float, new SymbolicTensorShape(numDirectionsDim, batchSize, hiddenSizeDim)));
                return new PartialTensor(DataType.Float, new SymbolicTensorShape(seqLength, numDirectionsDim, batchSize, hiddenSizeDim));
            }

            if (outputs.Length > 1 && !string.IsNullOrEmpty(outputs[1]))
                ctx.AddPartialTensor(outputs[1], new PartialTensor(DataType.Float, new SymbolicTensorShape(batchSize, numDirectionsDim, hiddenSizeDim)));
            if (outputs.Length > 2 && !string.IsNullOrEmpty(outputs[2]))
                ctx.AddPartialTensor(outputs[2], new PartialTensor(DataType.Float, new SymbolicTensorShape(batchSize, numDirectionsDim, hiddenSizeDim)));
            return new PartialTensor(DataType.Float, new SymbolicTensorShape(batchSize, seqLength, numDirectionsDim, hiddenSizeDim));
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var X = inputTensors[0] as TensorFloat;
            var W = inputTensors[1] as TensorFloat;
            var R = inputTensors[2] as TensorFloat;
            var B = inputTensors.Length > 3 ? inputTensors[3] as TensorFloat : null;
            var sequenceLens = inputTensors.Length > 4 ? inputTensors[4] as TensorInt : null;
            var initialH = inputTensors.Length > 5 ? inputTensors[5] as TensorFloat : null;
            var initialC = inputTensors.Length > 6 ? inputTensors[6] as TensorFloat : null;
            var P = inputTensors.Length > 7 ? inputTensors[7] as TensorFloat : null;

            ShapeInference.LSTM(X.shape, W.shape, R.shape, layout, out var shapeY, out var shapeY_h, out var shapeY_c);
            var Y = ctx.backend.NewOutputTensorFloat(shapeY);
            var Y_h = ctx.backend.NewOutputTensorFloat(shapeY_h);
            var Y_c = ctx.backend.NewOutputTensorFloat(shapeY_c);
            if (Y.shape.HasZeroDims())
            {
                if (outputs.Length > 1 && !string.IsNullOrEmpty(outputs[1]))
                    ctx.vars.Store(outputs[1], Y_h);
                if (outputs.Length > 2 && !string.IsNullOrEmpty(outputs[2]))
                    ctx.vars.Store(outputs[2], Y_c);
                return Y;
            }

            ctx.backend.LSTM(X, W, R, B, sequenceLens, initialH, initialC, P, Y, Y_h, Y_c, direction, activations, activationAlpha, activationBeta, inputForget, clip, layout);

            if (outputs.Length > 1 && !string.IsNullOrEmpty(outputs[1]))
                ctx.vars.Store(outputs[1], Y_h);
            if (outputs.Length > 2 && !string.IsNullOrEmpty(outputs[2]))
                ctx.vars.Store(outputs[2], Y_c);
            return Y;
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, outputs: [{string.Join(", ", outputs)}], hiddenSize: {hiddenSize}, direction: {direction}, activations: [{string.Join(", ", activations)}], activationAlpha: [{string.Join(", ", activationAlpha)}], activationBeta: [{string.Join(", ", activationBeta)}], clip: {clip}, inputForget: {inputForget}, layout: {layout}";
        }

        internal override string profilerTag { get { return "LSTM"; } }
    }
}
