using System;
using UnityEngine;

namespace Unity.Sentis {

/// <summary>
/// Represents a CPU backend ops.
/// </summary>
public partial class CPUBackend : IBackend
{
    bool m_OwnAllocator;
    internal ITensorAllocator m_Allocator;

    /// <inheritdoc/>
    public virtual DeviceType deviceType => DeviceType.CPU;

    /// <summary>
    /// Initializes and returns an instance of `CPUBackend`.
    /// </summary>
    /// <param name="allocator">The allocator to use when allocating tensors.</param>
    public CPUBackend(ITensorAllocator allocator = null)
    {
        if (allocator == null)
        {
            m_OwnAllocator = true;
            m_Allocator = new TensorCachingAllocator();
        }
        else
        {
            m_OwnAllocator = false;
            m_Allocator = allocator;
        }
    }

    /// <inheritdoc/>
    public virtual void ResetAllocator(bool keepCachedMemory = true)
    {
        m_Allocator.Reset(keepCachedMemory);
    }

    /// <summary>
    /// Disposes of the ops and any associated memory.
    /// </summary>
    public virtual void Dispose()
    {
        if (m_OwnAllocator)
            ResetAllocator(keepCachedMemory: false);
    }

    void ConvND(TensorFloat X, TensorFloat K, TensorFloat B, TensorFloat O, int groups, Span<int> stride, Span<int> pad, Span<int> dilation, Layers.FusableActivation fusedActivation)
    {
        var Otmp = (fusedActivation != Layers.FusableActivation.None) ? NewTempTensorFloat(O.shape) : O;

        ArrayTensorData.Pin(X);
        ArrayTensorData.Pin(K);
        ArrayTensorData.Pin(B);
        ArrayTensorData.Pin(Otmp, clearOnInit: false);

        int inputGroupedChannels = X.shape[1] / groups;
        int outputGroupedChannels = Otmp.shape[1] / groups;

        var itK = new TensorNDIterator(K.shape);
        itK = itK.RemoveDim(0);
        itK = itK.RemoveDim(0);

        var itX = new TensorNDIterator(X.shape);
        for (var itO = new TensorNDIterator(Otmp.shape); itO.HasNext(); itO.MoveNext())
        {
            int n = itO[0];
            int k = itO[1];
            float v = B[k];

            itX[0] = n;

            for (var c = 0; c < inputGroupedChannels; ++c)
            {
                itX[1] = (k / outputGroupedChannels) * inputGroupedChannels + c;

                itK.Reset();
                for (; itK.HasNext(); itK.MoveNext())
                {
                    bool outOfBounds = false;
                    for (int i = 0; i < stride.Length; i++)
                    {
                        int dx = itK[i];
                        int ox = itO[2 + i] * stride[i] + dilation[i] * dx - pad[i];

                        if ((ox < 0) || (ox >= X.shape[2 + i]))
                        {
                            outOfBounds = true;
                            break;
                        }

                        itX[2 + i] = ox;
                    }

                    if (outOfBounds)
                        continue;

                    float xv = X[itX.index];
                    float kv = K[k * K.shape[1] * itK.shape.length + c * itK.shape.length +  itK.index];

                    v += xv * kv;
                }
            }
            Otmp[itO.index] = v;
        }

        if (fusedActivation != Layers.FusableActivation.None)
            ApplyFusedActivation(Otmp, O, fusedActivation);
    }

    void ConvTransposeND(TensorFloat X, TensorFloat W, TensorFloat B, TensorFloat O, Span<int> strides, Span<int> pads, Span<int> outputPadding, Layers.FusableActivation fusedActivation)
    {
        var Otmp = (fusedActivation != Layers.FusableActivation.None) ? NewTempTensorFloat(O.shape) : O;

        ArrayTensorData.Pin(X);
        ArrayTensorData.Pin(W);
        ArrayTensorData.Pin(B);
        ArrayTensorData.Pin(O, clearOnInit: false);

        var inputChannels = X.shape[1];

        var itK = new TensorNDIterator(W.shape);
        var itX = new TensorNDIterator(X.shape);
        itX = itX.RemoveDim(0);
        itX = itX.RemoveDim(0);

        for (var itO = new TensorNDIterator(O.shape); itO.HasNext(); itO.MoveNext())
        {
            var n = itO[0];
            var k = itO[1];
            var v = B[k];
            itK[1] = k;

            for (var c = 0; c < inputChannels; ++c)
            {
                itK[0] = c;

                itX.Reset();
                for (; itX.HasNext(); itX.MoveNext())
                {
                    var outOfBounds = false;

                    for (var i = 0; i < strides.Length; i++)
                    {
                        var ox = itX[i];
                        var dx = itO[2 + i] + pads[i] - ox * strides[i];

                        if ((dx < 0) || (dx >= W.shape[2 + i]))
                        {
                            outOfBounds = true;
                            break;
                        }

                        itK[2 + i] = dx;
                    }

                    if (outOfBounds)
                        continue;

                    var xv = X[n * X.shape[1] * itX.shape.length + c * itX.shape.length + itX.index];
                    var kv = W[itK.index];

                    v += xv * kv;
                }
            }
            O[itO.index] = v;
        }

        if (fusedActivation != Layers.FusableActivation.None)
            ApplyFusedActivation(Otmp, O, fusedActivation);
    }

    void ResizeND(TensorFloat X, TensorFloat O, ReadOnlySpan<float> scale, Layers.InterpolationMode interpolationMode, Layers.NearestMode nearestMode = Layers.NearestMode.RoundPreferFloor, Layers.CoordTransformMode coordTransformMode = Layers.CoordTransformMode.HalfPixel)
    {
        for (var i = 0; i < scale.Length; i++)
        {
            var Otmp = i == scale.Length - 1 ? O : NewTempTensorFloat(ShapeInference.Resize(X.shape, i, scale[i]));
            Resize1D(X, Otmp, i, scale[i], interpolationMode, nearestMode, coordTransformMode);
            X = Otmp;
        }
    }

    void Resize1D(TensorFloat X, TensorFloat O, int axis, float scale, Layers.InterpolationMode interpolationMode, Layers.NearestMode nearestMode, Layers.CoordTransformMode coordTransformMode)
    {
        ArrayTensorData.Pin(X, clearOnInit: false);
        ArrayTensorData.Pin(O, clearOnInit: false);

        var itX = new TensorNDIterator(X.shape);

        for (var itO = new TensorNDIterator(O.shape); itO.HasNext(); itO.MoveNext())
        {
            itX.CopyNDIndex(itO);

            OpsUtils.GetScaleAndBias(X.shape[axis], O.shape[axis], scale, coordTransformMode, interpolationMode, nearestMode, out float outputScale, out float outputBias);

            float inputCoord = Math.Max(0.0f, itO[axis] * outputScale + outputBias);

            if (interpolationMode == Layers.InterpolationMode.Linear)
            {
                int indexValue = (int)inputCoord;
                float x_c0 = inputCoord - Mathf.Floor(inputCoord);
                float x_c1 = 1.0f - x_c0;

                itX[axis] = Mathf.Clamp(indexValue, 0, X.shape[axis] - 1);
                float x0 = X[itX.index];
                itX[axis] = Mathf.Clamp(indexValue + 1, 0, X.shape[axis] - 1);
                float x1 = X[itX.index];

                O[itO.index] = x_c0 * x1 + x_c1 * x0;
            }
            else
            {
                int indexValue = 0;
                switch (nearestMode)
                {
                    case Layers.NearestMode.RoundPreferFloor:
                    case Layers.NearestMode.Ceil:
                        indexValue = (int)Mathf.Ceil(inputCoord);
                        break;
                    case Layers.NearestMode.RoundPreferCeil:
                    case Layers.NearestMode.Floor:
                        indexValue = (int)Mathf.Floor(inputCoord);
                        break;
                }

                itX[axis] = Mathf.Clamp(indexValue, 0, X.shape[axis] - 1);
                O[itO.index] = X[itX.index];
            }
        }
    }

    void ApplyLocalPoolingOperator(TensorFloat X, TensorFloat O, int[] pool, int[] stride, int[] pad, Func<float> initOp, Func<float, float, float> accumulateOp, Func<float, int, float> normalizeOp)
    {
        ArrayTensorData.Pin(X);
        ArrayTensorData.Pin(O, clearOnInit: false);

        var itX = new TensorNDIterator(X.shape);
        var itP = new TensorNDIterator(new TensorShape(pool));
        for (var itO = new TensorNDIterator(O.shape); itO.HasNext(); itO.MoveNext())
        {
            itX[0] = itO[0];
            itX[1] = itO[1];

            float acc = initOp();
            int elementCount = 0;

            itP.Reset();
            for (; itP.HasNext(); itP.MoveNext())
            {
                bool outOfBounds = false;
                for (int i = 0; i < pool.Length; i++)
                {
                    int ox = itO[2 + i] * stride[i] + itP[i] - pad[i];

                    if ((ox < 0) || (ox >= X.shape[2 + i]))
                    {
                        outOfBounds = true;
                        break;
                    }

                    itX[2 + i] = ox;
                }

                if (!outOfBounds)
                {
                    acc = accumulateOp(acc, X[itX.index]);
                    elementCount++;
                }
            }

            O[itO.index] = normalizeOp(acc, elementCount);
        }
    }

    void MaxPoolND(TensorFloat X, TensorFloat O, int[] pool, int[] stride, int[] pad)
    {
        Func<float> initOp = () => float.MinValue;
        Func<float, float, float> accumulateOp = (acc, v) => Mathf.Max(acc, v);
        Func<float, int, float> normalizeOp = (acc, elementCount) => acc;
        ApplyLocalPoolingOperator(X, O, pool, stride, pad, initOp, accumulateOp, normalizeOp);
    }

    void AveragePoolND(TensorFloat X, TensorFloat O, int[] pool, int[] stride, int[] pad)
    {
        Func<float> initOp = () => 0.0f;
        Func<float, float, float> accumulateOp = (acc, v) => acc + v;
        Func<float, int, float> normalizeOp = (acc, elementCount) => acc / elementCount;
        ApplyLocalPoolingOperator(X, O, pool, stride, pad, initOp, accumulateOp, normalizeOp);
    }

    /// <inheritdoc/>
    public virtual void LRN(TensorFloat X, TensorFloat O, float alpha, float beta, float bias, int size)
    {
        // https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
        // However divide the sum by size to follow onnx and pytorch implementation
        // ONNX https://github.com/onnx/onnx/blob/master/docs/Operators.md#LRN
        // PYTORCH https://github.com/pytorch/pytorch/blob/1465970a343e61f2f2b104859ca7f5d7e03f5d02/torch/nn/functional.py#L2069
        // Tensorflow don't and follow the paper to the letter https://github.com/tensorflow/tensorflow/blob/e6faa845c51bb69465146d93646947fd2ba53efa/tensorflow/python/kernel_tests/lrn_op_test.py#L53
        // However they bake the division to alpha when exporting to ONNX https://github.com/onnx/tensorflow-onnx/blob/7c37ccb97e0fd478ce093910c4a1411b18e44fd7/tf2onnx/onnx_opset/math.py

        ArrayTensorData.Pin(X);
        ArrayTensorData.Pin(O, clearOnInit: false);

        float sizef = size;

        var itRemap = new TensorNDIterator(O.shape);
        for (var it = new TensorNDIterator(O.shape); it.HasNext(); it.MoveNext())
        {
            int c = it[1];
            float regionCenter = (sizef - 1.0f) / 2.0f;
            int regionStart = Math.Max(0, c - (int)Mathf.Floor(regionCenter));
            int regionEnd = Math.Min(X.shape[1], c + (int)Mathf.Ceil(regionCenter)+1);
            float sumOfSquared = 0.0f;
            for (int ci = regionStart; ci < regionEnd; ++ci)
            {
                itRemap.CopyNDIndex(it);
                itRemap[1] = ci;
                float regionValue = X[itRemap.index];
                sumOfSquared += regionValue * regionValue;
            }

            O[it.index] = X[it.index] / Mathf.Pow(bias + alpha * sumOfSquared / sizef, beta);
        }
    }

    void ScatterElementsReduce(TensorInt X, TensorInt indices, TensorInt updates, TensorInt O, int axis, Layers.ScatterReductionMode reduction)
    {
        MemCopy(X, O);

        ArrayTensorData.Pin(X);
        ArrayTensorData.Pin(indices);
        ArrayTensorData.Pin(updates);
        ArrayTensorData.Pin(O);

        var itO = new TensorNDIterator(O.shape);
        for (var itIndices = new TensorNDIterator(indices.shape); itIndices.HasNext(); itIndices.MoveNext())
        {
            itO = itIndices;

            var index = indices[itIndices.index];
            index = index < 0 ? X.shape[axis] + index : index;

            itO[axis] = index;

            if (reduction == Layers.ScatterReductionMode.None)
                O[itO.index] = updates[itIndices.index];
            else if (reduction == Layers.ScatterReductionMode.Add)
                O[itO.index] += updates[itIndices.index];
            else if (reduction == Layers.ScatterReductionMode.Mul)
                O[itO.index] *= updates[itIndices.index];
        }
    }

    /// <inheritdoc/>
    public virtual Tensor ShallowCopy(Tensor X, AllocScope allocScope)
    {
        if (X.allocator != null)
            return X.ShallowCopy();

        var O = NewTensor(X.shape, X.dataType, allocScope);
        if (O.shape.HasZeroDims())
            return O;
        MemCopy(X, O);
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor ShallowReshape(Tensor X, TensorShape shape, AllocScope allocScope)
    {
        Logger.AssertAreEqual(X.shape.length, shape.length, "Reshape.LengthError: in/out tensorshape must have the same # of elements : ({0}, {1})", X.shape.length, shape.length);
        // if already managed by allocator, can do a shallow copy
        if (X.allocator != null)
            return X.ShallowReshape(shape);

        var O = NewTensor(shape, X.dataType, allocScope);
        if (O.shape.HasZeroDims())
            return O;
        Reshape(X, O);
        return O;
    }
}
} // namespace Unity.Sentis
