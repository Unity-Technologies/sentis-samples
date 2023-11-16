using System;
using UnityEngine;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Options for the reduction operation to use in a scatter layer.
    /// </summary>
    public enum ScatterReductionMode
    {
        /// <summary>
        /// Use no reduction.
        /// </summary>
        None,
        /// <summary>
        /// Use the addition operator when reducing.
        /// </summary>
        Add,
        /// <summary>
        /// Use the multiplication operator when reducing.
        /// </summary>
        Mul,
    }

    /// <summary>
    /// Represents a reduction which calculates indices.
    /// </summary>
    [Serializable]
    public abstract class ArgReduce : Layer
    {
        /// <summary>
        /// The axis along which to perform the operation.
        /// </summary>
        public int axis;
        /// <summary>
        /// Whether to keep the axis dimension in the output tensor.
        /// </summary>
        public bool keepdims;
        /// <summary>
        /// Whether to perform the operation from the back of the axis.
        /// </summary>
        public bool selectLastIndex;

        /// <summary>
        /// Initializes and returns an instance of `ArgReduce` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="axis">The axis along which to perform the operation.</param>
        /// <param name="keepdims">Whether to keep the axis dimension in the output tensor. The default value is `true`.</param>
        /// <param name="selectLastIndex">Whether to perform the operation from the back of the axis. The default value is `false`.</param>
        protected ArgReduce(string name, string input, int axis, bool keepdims = true, bool selectLastIndex = false)
        {
            this.name = name;
            this.inputs = new[] { input };
            this.axis = axis;
            this.keepdims = keepdims;
            this.selectLastIndex = selectLastIndex;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var shapeX = inputTensors[0].shape;
            if (!shapeX.hasRank)
                return new PartialTensor(DataType.Int);

            var reducedShape = new SymbolicTensorShape(shapeX);

            // reducing on a zero axis will result in a zero rather than a one
            if (shapeX[axis].isValue)
                reducedShape[axis] = shapeX[axis].value == 0 ? SymbolicTensorDim.Zero : SymbolicTensorDim.One;
            else
                reducedShape[axis] = SymbolicTensorDim.Unknown;

            var shapeOut = !keepdims ? reducedShape.Squeeze(axis) : reducedShape;
            return new PartialTensor(DataType.Int, shapeOut);
        }
    }

    /// <summary>
    /// Represents an `ArgMax` layer. This computes the indices of the maximum elements of the input tensor along a given axis.
    /// </summary>
    [Serializable]
    public class ArgMax : ArgReduce
    {
        /// <summary>
        /// Initializes and returns an instance of `ArgMax` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="axis">The axis along which to perform the operation.</param>
        /// <param name="keepdims">Whether to keep the axis dimension in the output tensor. The default value is `true`.</param>
        /// <param name="selectLastIndex">Whether to perform the operation from the back of the axis. The default value is `false`.</param>
        public ArgMax(string name, string input, int axis, bool keepdims = true, bool selectLastIndex = false)
            : base(name, input, axis, keepdims, selectLastIndex) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var shapeO = inputTensors[0].shape.Reduce(axis, keepdims);
            var O = ctx.backend.NewOutputTensorInt(shapeO);
            if (O.shape.HasZeroDims())
                return O;
            if (inputTensors[0] is TensorInt)
                ctx.backend.ArgMax(inputTensors[0] as TensorInt, O, axis, keepdims, selectLastIndex);
            else
                ctx.backend.ArgMax(inputTensors[0] as TensorFloat, O, axis, keepdims, selectLastIndex);
            return O;
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, axis: {axis}, keepdims: {keepdims}, selectLastIndex: {selectLastIndex}";
        }

        internal override string profilerTag => "ArgMax";
    }

    /// <summary>
    /// Represents an `ArgMin` layer. This computes the indices of the minimum elements of the input tensor along a given axis.
    /// </summary>
    [Serializable]
    public class ArgMin : ArgReduce
    {
        /// <summary>
        /// Initializes and returns an instance of `ArgMin` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="axis">The axis along which to perform the operation.</param>
        /// <param name="keepdims">Whether to keep the axis dimension in the output tensor. The default value is `true`.</param>
        /// <param name="selectLastIndex">Whether to perform the operation from the back of the axis. The default value is `false`.</param>
        public ArgMin(string name, string input, int axis, bool keepdims = true, bool selectLastIndex = false)
            : base(name, input, axis, keepdims, selectLastIndex) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var shapeO = inputTensors[0].shape.Reduce(axis, keepdims);
            var O = ctx.backend.NewOutputTensorInt(shapeO);
            if (O.shape.HasZeroDims())
                return O;
            if (inputTensors[0] is TensorInt)
                ctx.backend.ArgMin(inputTensors[0] as TensorInt, O, axis, keepdims, selectLastIndex);
            else
                ctx.backend.ArgMin(inputTensors[0] as TensorFloat, O, axis, keepdims, selectLastIndex);
            return O;
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, axis: {axis}, keepdims: {keepdims}, selectLastIndex: {selectLastIndex}";
        }

        internal override string profilerTag => "ArgMin";
    }

    /// <summary>
    /// Represents a `Gather` layer. This takes values from the input tensor indexed by the indices tensor along a given axis and concatenates them.
    /// </summary>
    [Serializable]
    public class Gather : Layer
    {
        /// <summary>
        /// The axis along which to perform the gather.
        /// </summary>
        public int axis;

        /// <summary>
        /// Initializes and returns an instance of `Gather` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="indices">The name to use for the indices tensor of the layer.</param>
        /// <param name="axis">The axis along which to perform the gather.</param>
        public Gather(string name, string input, string indices, int axis)
        {
            this.name = name;
            inputs = new[] { input, indices };
            this.axis = axis;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var dataType = inputTensors[0].dataType;
            var input = inputTensors[0];
            var indices = inputTensors[1];
            if (dataType == DataType.Int && axis == 0 && input.isPartiallyKnown && indices.isPartiallyKnown && input.shape.rank == 1 && indices.shape.rank <= 1)
            {
                var tensorOut = new PartialTensor(dataType, indices.shape);
                for (var i = 0; i < indices.length; i++)
                {
                    var index = indices[i];
                    index = index < 0 ? index + input.length : index;
                    tensorOut[i] = input[index];
                }

                return tensorOut;
            }

            var shapeX = input.shape;
            var shapeIndices = indices.shape;
            if (!shapeX.hasRank)
                return new PartialTensor(dataType);

            Logger.AssertIsTrue(shapeX.hasRank ? shapeX.rank >= 1 : true, "RankError: incorrect rank, expecting at least {0}, got {1}", 1, shapeX.rank);

            if (!shapeIndices.hasRank)
                return new PartialTensor(dataType);

            var axisX = shapeX.Axis(axis);

            var shapeOut = SymbolicTensorShape.UnknownOfRank(shapeX.rank - 1 + shapeIndices.rank);

            for (var i = 0; i < shapeOut.rank; i++)
            {
                if (i < axisX)
                    shapeOut[i] = shapeX[i];
                else if (i < axisX + shapeIndices.rank)
                    shapeOut[i] = shapeIndices[i - axisX];
                else
                    shapeOut[i] = shapeX[i - shapeOut.rank];
            }

            return new PartialTensor(dataType, shapeOut);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensor(ShapeInference.Gather(inputTensors[0].shape, inputTensors[1].shape, axis), inputTensors[0].dataType);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.Gather(inputTensors[0], inputTensors[1] as TensorInt, O, axis);
            return O;
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, axis: {axis}";
        }

        internal override string profilerTag => "Gather";
    }

    /// <summary>
    /// Represents a `GatherElements` layer. This takes values from the input tensor indexed by the `indices` tensor along a given axis.
    /// </summary>
    [Serializable]
    public class GatherElements : Layer
    {
        /// <summary>
        /// The axis along which to perform the gather.
        /// </summary>
        public int axis;

        /// <summary>
        /// Initializes and returns an instance of `GatherElements` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="indices">The name to use for the indices tensor of the layer.</param>
        /// <param name="axis">The axis along which to perform the gather.</param>
        public GatherElements(string name, string input, string indices, int axis)
        {
            this.name = name;
            inputs = new[] { input, indices };
            this.axis = axis;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var shapeX = inputTensors[0].shape;
            var shapeIndices = inputTensors[1].shape;
            Logger.AssertIsTrue(shapeX.hasRank ? shapeX.rank >= 1 : true, "RankError: incorrect rank, expecting at least {0}, got {1}", 1, shapeX.rank);

            if (shapeX.hasRank)
            {
                shapeX.Axis(axis);
                shapeIndices.DeclareRank(shapeX.rank);
            }

            return new PartialTensor(inputTensors[0].dataType, shapeIndices);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensor(inputTensors[1].shape, inputTensors[0].dataType);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.GatherElements(inputTensors[0], inputTensors[1] as TensorInt, O, axis);
            return O;
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, axis: {axis}";
        }

        internal override string profilerTag => "GatherElements";
    }

    /// <summary>
    /// Represents a `GatherND` layer. This takes slices of values from the batched input tensor indexed by the `indices` tensor.
    /// </summary>
    [Serializable]
    public class GatherND : Layer
    {
        /// <summary>
        /// The number of batch dimensions of the input tensor. The gather begins at the next dimension.
        /// </summary>
        public int batchDims;

        /// <summary>
        /// Initializes and returns an instance of `GatherND` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="indices">The name to use for the indices tensor of the layer.</param>
        /// <param name="batchDims">The number of batch dimensions of the input tensor, the gather begins at the next dimension.</param>
        public GatherND(string name, string input, string indices, int batchDims)
        {
            this.name = name;
            inputs = new[] { input, indices };
            this.batchDims = batchDims;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var dataType = inputTensors[0].dataType;
            var shapeX = inputTensors[0].shape;
            var shapeIndices = inputTensors[1].shape;
            // from https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherND
            Logger.AssertIsTrue(shapeX.hasRank ? shapeX.rank >= batchDims : true, "RankError: incorrect rank, expecting at least {0}, got {1}", batchDims, shapeX.rank);
            Logger.AssertIsTrue(shapeIndices.hasRank ? shapeIndices.rank >= batchDims : true, "RankError: incorrect rank, expecting at least {0}, got {1}", batchDims, shapeIndices.rank);

            if (!shapeX.hasRank || !shapeIndices.hasRank)
                return new PartialTensor(dataType);

            if (!shapeIndices[-1].isValue)
                return new PartialTensor(dataType);

            Logger.AssertIsTrue(batchDims + shapeIndices[-1].value <= shapeX.rank, "GatherND.InputError: last indices dim too large");
            var shapeOut = SymbolicTensorShape.UnknownOfRank(shapeX.rank + shapeIndices.rank - shapeIndices[-1].value - 1 - batchDims);
            for (var i = 0; i < shapeOut.rank; i++)
            {
                if (i < batchDims)
                    shapeOut[i] = SymbolicTensorDim.MaxDefinedDim(shapeX[i], shapeIndices[i]);
                else if (i < shapeIndices.rank - 1)
                    shapeOut[i] = shapeIndices[i];
                else
                    shapeOut[i] = shapeX[i - shapeOut.rank];
            }

            return new PartialTensor(dataType, shapeOut);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensor(ShapeInference.GatherND(inputTensors[0].shape, inputTensors[1].shape, batchDims), inputTensors[0].dataType);
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.GatherND(inputTensors[0], inputTensors[1] as TensorInt, O, batchDims);
            return O;
        }

        internal override string profilerTag => "GatherND";
    }

    /// <summary>
    /// Represents a `NonZero` layer. This returns the indices of the elements of the input tensor that are not zero.
    /// </summary>
    [Serializable]
    public class NonZero : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `NonZero` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public NonZero(string name, string input)
        {
            this.name = name;
            inputs = new[] { input };
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var shape = !inputTensors[0].shape.hasRank ? SymbolicTensorShape.UnknownOfRank(2) : new SymbolicTensorShape(new SymbolicTensorDim(inputTensors[0].shape.rank), SymbolicTensorDim.Unknown);
            return new PartialTensor(DataType.Int, shape);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            if (inputTensors[0] is TensorInt)
            {
                var X = inputTensors[0] as TensorInt;
                ArrayTensorData.Pin(X);
                int nbNonZeroIndices = 0;
                var end = X.shape.length;
                for (int i = 0; i < end; ++i)
                {
                    if (X[i] != 0.0f)
                        nbNonZeroIndices += 1;
                }

                var O = ctx.backend.NewOutputTensorInt(new TensorShape(X.shape.rank, nbNonZeroIndices));
                if (O.shape.HasZeroDims())
                    return O;

                ArrayTensorData.Pin(O, clearOnInit: false);
                int nonZeroIndicesIdx = 0;
                for (var it = new TensorNDIterator(X.shape); it.HasNext(); it.MoveNext())
                {
                    if (X[it.index] != 0.0f)
                    {
                        for (int i = 0; i < X.shape.rank; i++)
                            O[i * nbNonZeroIndices + nonZeroIndicesIdx] = it[i];
                        nonZeroIndicesIdx++;
                    }
                }

                return O;
            }
            else
            {
                var X = inputTensors[0] as TensorFloat;
                ArrayTensorData.Pin(X);
                int nbNonZeroIndices = 0;
                var end = X.shape.length;
                for (int i = 0; i < end; ++i)
                {
                    if (X[i] != 0)
                        nbNonZeroIndices += 1;
                }

                var O = ctx.backend.NewOutputTensorInt(new TensorShape(X.shape.rank, nbNonZeroIndices));
                if (O.shape.HasZeroDims())
                    return O;

                ArrayTensorData.Pin(O, clearOnInit: false);
                int nonZeroIndicesIdx = 0;
                for (var it = new TensorNDIterator(X.shape); it.HasNext(); it.MoveNext())
                {
                    if (X[it.index] != 0)
                    {
                        for (int i = 0; i < X.shape.rank; i++)
                            O[i * nbNonZeroIndices + nonZeroIndicesIdx] = it[i];
                        nonZeroIndicesIdx++;
                    }
                }

                return O;
            }
        }

        internal override string profilerTag => "NonZero";
    }

    /// <summary>
    /// Represents a `ScatterElements` layer. This copies the input tensor and updates values at indexes specified by the `indices` tensor with values specified by the `updates` tensor along a given axis.
    ///
    /// `ScatterElements` updates the values depending on the reduction mode used.
    /// </summary>
    [Serializable]
    public class ScatterElements : Layer
    {
        /// <summary>
        /// The axis on which to perform the scatter.
        /// </summary>
        public int axis;
        /// <summary>
        /// The reduction mode used to update the values as a `ScatterReductionMode`.
        /// </summary>
        public ScatterReductionMode reduction;

        /// <summary>
        /// Initializes and returns an instance of `ScatterElements` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="indices">The name to use for the indices tensor of the layer.</param>
        /// <param name="updates">The name to use for the updates tensor of the layer.</param>
        /// <param name="axis">The axis on which to perform the scatter.</param>
        /// <param name="reduction">The reduction mode used to update the values as a `ScatterReductionMode`.</param>
        public ScatterElements(string name, string input, string indices, string updates, int axis, ScatterReductionMode reduction)
        {
            this.name = name;
            inputs = new[] { input, indices, updates };
            this.axis = axis;
            this.reduction = reduction;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var dataType = inputTensors[0].dataType;
            var shapeX = inputTensors[0].shape;
            var shapeIndices = inputTensors[1].shape;
            var shapeUpdates = inputTensors[2].shape;

            if (!shapeX.hasRank && !shapeIndices.hasRank && !shapeUpdates.hasRank)
                return new PartialTensor(dataType);

            if (!shapeX.hasRank && shapeIndices.hasRank)
                shapeX = SymbolicTensorShape.UnknownOfRank(shapeIndices.rank);

            if (!shapeX.hasRank && shapeUpdates.hasRank)
                shapeX = SymbolicTensorShape.UnknownOfRank(shapeUpdates.rank);

            Logger.AssertIsTrue(shapeX.hasRank ? shapeX.rank >= 1 : true, "RankError: incorrect rank, expecting at least {0}, got {1}", 1, shapeX.rank);

            shapeIndices.DeclareRank(shapeX.rank);
            shapeUpdates.DeclareRank(shapeX.rank);

            // throw error if axis incorrect
            shapeX.Axis(axis);

            // throw error if indices and updates don't match
            for (var i = 0; i < shapeIndices.rank; i++)
            {
                SymbolicTensorDim.MaxDefinedDim(shapeIndices[i], shapeUpdates[i]);
            }

            return new PartialTensor(dataType, shapeX);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensor(inputTensors[0].shape, inputTensors[0].dataType);
            if (O.shape.HasZeroDims())
                return O;
            if (inputTensors[1].shape.HasZeroDims())
                ctx.backend.MemCopy(inputTensors[0], O);
            else
                ctx.backend.ScatterElements(inputTensors[0], inputTensors[1] as TensorInt, inputTensors[2], O, axis, reduction);
            return O;
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, axis: {axis}, reduction: {reduction}";
        }

        internal override string profilerTag => "ScatterElements";
    }

    /// <summary>
    /// Represents a `ScatterND` layer. This copies the input tensor and updates values at indexes specified by the `indices` tensor with values specified by the `updates` tensor.
    ///
    /// `ScatterND` updates the values depending on the reduction mode used.
    /// </summary>
    [Serializable]
    public class ScatterND : Layer
    {
        /// <summary>
        /// The reduction mode used to update the values as a `ScatterReductionMode`.
        /// </summary>
        public ScatterReductionMode reduction;

        /// <summary>
        /// Initializes and returns an instance of `ScatterND` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="indices">The name to use for the indices tensor of the layer.</param>
        /// <param name="updates">The name to use for the updates tensor of the layer.</param>
        /// <param name="reduction">The reduction mode used to update the values as a `ScatterReductionMode`.</param>
        public ScatterND(string name, string input, string indices, string updates, ScatterReductionMode reduction)
        {
            this.name = name;
            inputs = new[] { input, indices, updates };
            this.reduction = reduction;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var dataType = inputTensors[0].dataType;
            var shapeX = inputTensors[0].shape;
            var shapeIndices = inputTensors[1].shape;
            var shapeUpdates = inputTensors[2].shape;

            Logger.AssertIsTrue(shapeIndices.hasRank ? shapeIndices.rank >= 1 : true, "RankError: incorrect rank, expecting at least {0}, got {1}", shapeIndices.rank, 1);

            if (shapeIndices.hasRank && shapeUpdates.hasRank && shapeIndices[-1].isValue)
                shapeX.DeclareRank(shapeUpdates.rank - (shapeIndices.rank - shapeIndices[-1].value - 1));

            Logger.AssertIsTrue(shapeX.hasRank ? shapeX.rank >= 1 : true, "RankError: incorrect rank, expecting at least {0}, got {1}", 1, shapeX.rank);

            return new PartialTensor(dataType, shapeX);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensor(inputTensors[0].shape, inputTensors[0].dataType);
            if (O.shape.HasZeroDims())
                return O;
            if (inputTensors[1].shape.HasZeroDims())
            {
                ctx.backend.MemCopy(inputTensors[0], O);
                return O;
            }
            if (inputTensors[0] is TensorInt)
                ctx.backend.ScatterND(inputTensors[0] as TensorInt, inputTensors[1] as TensorInt, inputTensors[2] as TensorInt, O as TensorInt, reduction);
            else
                ctx.backend.ScatterND(inputTensors[0] as TensorFloat, inputTensors[1] as TensorInt, inputTensors[2] as TensorFloat, O as TensorFloat, reduction);
            return O;
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, reduction: {reduction}";
        }

        internal override string profilerTag => "ScatterND";
    }

    /// <summary>
    /// Represents a `TopK` layer. This calculates the top-K largest or smallest elements of an input tensor along a given axis.
    ///
    /// This layer calculates both the values tensor of the top-K elements and the indices tensor of the top-K elements as outputs.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(1)]
    public class TopK : Layer
    {
        /// <summary>
        /// The axis along which to perform the top-K operation.
        /// </summary>
        public int axis;
        /// <summary>
        /// Whether to calculate the top-K largest elements. If this is `false` the layer calculates the top-K smallest elements.
        /// </summary>
        public bool largest;
        /// <summary>
        /// Whether to return the elements in sorted order in the output tensor.
        /// </summary>
        public bool sorted;

        /// <summary>
        /// Initializes and returns an instance of `TopK` layer.
        /// </summary>
        /// <param name="name">The name to use for the values tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="k">The name to use for the single value 1D tensor containing the number of elements to calculate.</param>
        /// <param name="axis">The axis along which to perform the top-K operation.</param>
        /// <param name="largest">Whether to calculate the top-K largest elements. If this is `false` the layer calculates the top-K smallest elements.</param>
        /// <param name="sorted">Whether to return the elements in sorted order in the output tensor.</param>
        /// <param name="outputNames">A two-element array containing the names to use for the values and indices output tensors of the layer respectively.</param>
        public TopK(string name, string input, string k, int axis, bool largest, bool sorted, string[] outputNames)
        {
            this.name = name;
            this.inputs = new[] { input, k };
            this.axis = axis;
            this.largest = largest;
            this.sorted = sorted;
            this.outputs = outputNames;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var dataType = inputTensors[0].dataType;
            var shapeX = inputTensors[0].shape;
            if (!shapeX.hasRank)
            {
                ctx.AddPartialTensor(outputs[1], new PartialTensor(DataType.Int));
                return new PartialTensor(dataType);
            }

            var shapeK = inputTensors[1].shape;
            shapeK.DeclareRank(1);
            Logger.AssertIsFalse(shapeK[0] != 1, "TopK.InputError: k must be a single value");

            var shapeOut = new SymbolicTensorShape(shapeX);

            var axisX = shapeX.Axis(axis);

            shapeOut[axisX] = (SymbolicTensorDim)inputTensors[1][0];

            ctx.AddPartialTensor(outputs[1], new PartialTensor(DataType.Int, shapeOut));
            return new PartialTensor(dataType, shapeOut);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var k = inputTensors[1].ToReadOnlySpan<int>()[0];
            var outputShape = new TensorShape(inputTensors[0].shape);
            outputShape[axis] = k;

            var values = ctx.backend.NewOutputTensorFloat(outputShape);
            var indices = ctx.backend.NewOutputTensorInt(outputShape);
            if (outputShape.HasZeroDims())
            {
                ctx.vars.Store(outputs[1], indices);
                return values;
            }
            ctx.backend.TopK(inputTensors[0] as TensorFloat, values, indices, k, axis, largest);
            ctx.vars.Store(outputs[1], indices);
            return values;
        }

        internal override string profilerTag => "TopK";
    }
}
