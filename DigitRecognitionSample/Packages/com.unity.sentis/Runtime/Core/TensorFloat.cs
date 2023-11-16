using System;
using Unity.Collections;

namespace Unity.Sentis {

/// <summary>
/// Represents data in a multidimensional array-like structure of floats.
/// </summary>
public class TensorFloat : Tensor
{
    /// <inheritdoc/>
    public override DataType dataType { get { return DataType.Float; } }

    /// <inheritdoc/>
    internal TensorFloat(TensorShape shape, ITensorData data = null, ITensorAllocator allocator = null) : base(shape, data, allocator) { }

    /// <summary>
    /// Initializes and returns a tensor with the specified `shape` and a float[] array of `srcData` data.
    /// </summary>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="srcData">The data elements of the tensor.</param>
    public TensorFloat(TensorShape shape, float[] srcData) : this(shape, srcData, 0) { }

    /// <summary>
    /// Initializes and returns a tensor with specified `shape` and a float[] array of `srcData` data. Sentis reads `srcData` from `dataStartIndex`.
    ///
    /// `srcData.Length` - `dataStartIndex` must be bigger than or equal to `shape.length`.
    /// </summary>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="srcData">The data elements of the tensor.</param>
    /// <param name="dataStartIndex">The index of the first tensor element in the srcData array.</param>
    public TensorFloat(TensorShape shape, float[] srcData, int dataStartIndex = 0) : base(shape)
    {
        Logger.AssertIsTrue((srcData.Length - dataStartIndex) >= shape.length, "RangeError: array length {0} is too small compared to shape length {1}", srcData.Length, shape);

        tensorOnDevice = new ArrayTensorData(shape);
        NativeTensorArray.Copy(srcData, (tensorOnDevice as ArrayTensorData).array, shape.length, dataStartIndex);

        m_TensorAllocator = null;
    }

    /// <summary>
    /// Initializes and returns a tensor with specified `shape` and a native float array of `srcData` data. Sentis reads `srcData` from `dataStartIndex`.
    ///
    /// `srcData.Length` - `dataStartIndex` must be bigger than or equal to `shape.length`.
    /// </summary>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="srcData">The data elements of the tensor.</param>
    /// <param name="dataStartIndex">The index of the first tensor element in the srcData native array.</param>
    public TensorFloat(TensorShape shape, NativeArray<float> srcData, int dataStartIndex = 0) : base(shape)
    {
        Logger.AssertIsTrue((srcData.Length - dataStartIndex) >= shape.length, "RangeError: array length {0} is too small compared to shape length {1}", srcData.Length, shape);

        tensorOnDevice = new ArrayTensorData(shape);
        NativeTensorArray.Copy(srcData, (tensorOnDevice as ArrayTensorData).array, shape.length, dataStartIndex);

        m_TensorAllocator = null;
    }

    /// <summary>
    /// Initializes and returns a scalar tensor with the value of `srcData`.
    /// </summary>
    /// <param name="srcData">The data element of the tensor.</param>
    public TensorFloat(float srcData) : this(new TensorShape(), new[] { srcData }) { }

    /// <summary>
    /// Initializes and returns a tensor with the specified `shape` and filled with `0`.
    /// </summary>
    /// <param name="shape">The shape of the tensor.</param>
    /// <returns>The instantiated zero tensor.</returns>
    public static TensorFloat Zeros(TensorShape shape)
    {
        var tensorOnDevice = new ArrayTensorData(shape, clearOnInit: true);
        return new TensorFloat(shape, tensorOnDevice);
    }

    /// <inheritdoc/>
    public override Tensor ShallowReshape(TensorShape newShape)
    {
        TensorFloat copy;
        if (m_TensorAllocator != null)
            copy = m_TensorAllocator.Alloc(newShape, DataType.Float, m_TensorOnDevice, AllocScope.LayerOutput) as TensorFloat;
        else
            copy = new TensorFloat(newShape, m_TensorOnDevice, null);

        return copy;
    }

    /// <inheritdoc/>
    public override Tensor DeepCopy()
    {
        var copy = new TensorFloat(shape);
        if (shape.HasZeroDims())
            return copy;
        copy.AttachToDevice(m_TensorOnDevice.Clone());
        return copy;
    }

    /// <inheritdoc/>
    public override void UploadToDevice(ITensorData destination)
    {
        var data = m_TensorOnDevice.Download<float>(shape.length);
        destination.Upload(data, shape.length);
        data.Dispose();
        PinToDevice(destination, disposeUnpinned: true);
    }

    /// <summary>
    /// Returns the tensor element at offset `(d7, d6, d5, d4, d3, d2, d1, d0)`, which is position `d7 * stride6 + d6 * stride5 + d5 * stride4 + d4 * stride3 + d3 * stride2 + d2 * stride1 + d1 * stride0 + d0`.
    /// </summary>
    /// <param name="d7">Axis 7.</param>
    /// <param name="d6">Axis 6.</param>
    /// <param name="d5">Axis 5.</param>
    /// <param name="d4">Axis 4.</param>
    /// <param name="d3">Axis 3.</param>
    /// <param name="d2">Axis 2.</param>
    /// <param name="d1">Axis 1.</param>
    /// <param name="d0">Axis 0.</param>
    public float this[int d7, int d6, int d5, int d4, int d3, int d2, int d1, int d0]
    {
        get { return this[shape.RavelIndex(d7, d6, d5, d4, d3, d2, d1, d0)]; }
        set { this[shape.RavelIndex(d7, d6, d5, d4, d3, d2, d1, d0)] = value;}
    }

    /// <summary>
    /// Returns the tensor element at offset `(d6, d5, d4, d3, d2, d1, d0)`, which is position `d6 * stride5 + d5 * stride4 + d4 * stride3 + d3 * stride2 + d2 * stride1 + d1 * stride0 + d0`.
    /// </summary>
    /// <param name="d6">Axis 6.</param>
    /// <param name="d5">Axis 5.</param>
    /// <param name="d4">Axis 4.</param>
    /// <param name="d3">Axis 3.</param>
    /// <param name="d2">Axis 2.</param>
    /// <param name="d1">Axis 1.</param>
    /// <param name="d0">Axis 0.</param>
    public float this[int d6, int d5, int d4, int d3, int d2, int d1, int d0]
    {
        get { return this[shape.RavelIndex(d6, d5, d4, d3, d2, d1, d0)]; }
        set { this[shape.RavelIndex(d6, d5, d4, d3, d2, d1, d0)] = value;}
    }
    /// <summary>
    /// Returns the tensor element at offset `(d5, d4, d3, d2, d1, d0)`, which is position `d5 * stride4 + d4 * stride3 + d3 * stride2 + d2 * stride1 + d1 * stride0 + d0`.
    /// </summary>
    /// <param name="d5">Axis 5.</param>
    /// <param name="d4">Axis 4.</param>
    /// <param name="d3">Axis 3.</param>
    /// <param name="d2">Axis 2.</param>
    /// <param name="d1">Axis 1.</param>
    /// <param name="d0">Axis 0.</param>
    public float this[int d5, int d4, int d3, int d2, int d1, int d0]
    {
        get { return this[shape.RavelIndex(d5, d4, d3, d2, d1, d0)]; }
        set { this[shape.RavelIndex(d5, d4, d3, d2, d1, d0)] = value;}
    }
    /// <summary>
    /// Returns the tensor element at offset `(d4, d3, d2, d1, d0)`, which is position `d4 * stride3 + d3 * stride2 + d2 * stride1 + d1 * stride0 + d0`.
    /// </summary>
    /// <param name="d4">Axis 4.</param>
    /// <param name="d3">Axis 3.</param>
    /// <param name="d2">Axis 2.</param>
    /// <param name="d1">Axis 1.</param>
    /// <param name="d0">Axis 0.</param>
    public float this[int d4, int d3, int d2, int d1, int d0]
    {
        get { return this[shape.RavelIndex(d4, d3, d2, d1, d0)]; }
        set { this[shape.RavelIndex(d4, d3, d2, d1, d0)] = value;}
    }
    /// <summary>
    /// Returns the tensor element at offset `(d3, d2, d1, d0)`, which is position `d3 * stride2 + d2 * stride1 + d1 * stride0 + d0` in this tensor.
    /// </summary>
    /// <param name="d3">Axis 3.</param>
    /// <param name="d2">Axis 2.</param>
    /// <param name="d1">Axis 1.</param>
    /// <param name="d0">Axis 0.</param>
    public float this[int d3, int d2, int d1, int d0]
    {
        get { return this[shape.RavelIndex(d3, d2, d1, d0)]; }
        set { this[shape.RavelIndex(d3, d2, d1, d0)] = value;}
    }
    /// <summary>
    /// Returns the tensor element at offset `(d2, d1, d0)`, which is position `d2 * stride1 + d1 * stride0 + d0`.
    /// </summary>
    /// <param name="d2">Axis 2.</param>
    /// <param name="d1">Axis 1.</param>
    /// <param name="d0">Axis 0.</param>
    public float this[int d2, int d1, int d0]
    {
        get { return this[shape.RavelIndex(d2, d1, d0)]; }
        set { this[shape.RavelIndex(d2, d1, d0)] = value;}
    }
    /// <summary>
    /// Returns the tensor element at offset `(d1, d0)`, which is position `d1 * stride0 + d0`.
    /// </summary>
    /// <param name="d1">Axis 1.</param>
    /// <param name="d0">Axis 0.</param>
    public float this[int d1, int d0]
    {
        get { return this[shape.RavelIndex(d1, d0)]; }
        set { this[shape.RavelIndex(d1, d0)] = value;}
    }

    /// <summary>
    /// Returns the tensor element at offset `d0`.
    /// </summary>
    /// <param name="d0">Axis 0.</param>
    public float this[int d0]
    {
        get { return base.Get<float>(d0); }
        set { base.Set<float>(d0, value); }
    }

    /// <summary>
    /// Returns a copy of linear memory representation of the data in this tensor.
    ///
    /// the returned array is a deepcopy of the tensor, the caller of this methods is now responsible for it.
    /// If you modify the contents of the returned array, it will not modify the underlying tensor
    /// </summary>
    /// <returns>Float array copy of tensor data.</returns>
    public float[] ToReadOnlyArray()
    {
        return base.ToReadOnlyArray<float>();
    }

    /// <summary>
    /// Returns a ReadOnlySpan on the linear memory representation of the data in this tensor.
    /// </summary>
    /// <returns>Span of tensor data.</returns>
    public ReadOnlySpan<float> ToReadOnlySpan()
    {
        return base.ToReadOnlySpan<float>();
    }
}

} // namespace Unity.Sentis
