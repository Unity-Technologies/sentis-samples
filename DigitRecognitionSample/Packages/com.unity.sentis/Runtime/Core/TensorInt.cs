using System;
using Unity.Collections;

namespace Unity.Sentis {

/// <summary>
/// Represents data in a multidimensional array-like structure of ints.
/// </summary>
public class TensorInt : Tensor
{
    /// <inheritdoc/>
    public override DataType dataType => DataType.Int;

    /// <inheritdoc/>
    internal TensorInt(TensorShape shape, ITensorData data = null, ITensorAllocator allocator = null) : base(shape, data, allocator) { }

    /// <summary>
    /// Initializes and returns a tensor with the specified `shape` and an int[] array of `srcData` data.
    /// </summary>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="srcData">The data elements of the tensor.</param>
    public TensorInt(TensorShape shape, int[] srcData) : this(shape, srcData, 0) { }

    /// <summary>
    /// Initializes and returns a tensor with specified `shape` and an int[] array of `srcData` data. Sentis reads `srcData` from `dataStartIndex`.
    ///
    /// `srcData.Length` - `dataStartIndex` must be bigger than or equal to `shape.length`.
    /// </summary>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="srcData">The data elements of the tensor.</param>
    /// <param name="dataStartIndex">The index of the first tensor element in the srcData array.</param>
    public TensorInt(TensorShape shape, int[] srcData, int dataStartIndex = 0) : base(shape)
    {
        Logger.AssertIsTrue((srcData.Length - dataStartIndex) >= shape.length, "RangeError: array length {0} is too small compared to shape length {1}", srcData.Length, shape);

        tensorOnDevice = new ArrayTensorData(shape);
        NativeTensorArray.Copy(srcData, (tensorOnDevice as ArrayTensorData).array, shape.length, dataStartIndex);

        m_TensorAllocator = null;
    }

    /// <summary>
    /// Initializes and returns a tensor with specified `shape` and a native int array of `srcData` data. Sentis reads `srcData` from `dataStartIndex`.
    ///
    /// `srcData.Length` - `dataStartIndex` must be bigger than or equal to `shape.length`.
    /// </summary>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="srcData">The data elements of the tensor.</param>
    /// <param name="dataStartIndex">The index of the first tensor element in the srcData native array.</param>
    public TensorInt(TensorShape shape, NativeArray<int> srcData, int dataStartIndex = 0) : base(shape)
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
    public TensorInt(int srcData) : this(new TensorShape(), new[] { srcData }) { }

    /// <summary>
    /// Initializes and returns a tensor with the specified `shape` and filled with `0`.
    /// </summary>
    /// <param name="shape">The shape of the tensor.</param>
    /// <returns>The instantiated zero tensor.</returns>
    public static TensorInt Zeros(TensorShape shape)
    {
        var tensorOnDevice = new ArrayTensorData(shape, clearOnInit: true);
        return new TensorInt(shape, tensorOnDevice);
    }

    /// <inheritdoc/>
    public override Tensor ShallowReshape(TensorShape newShape)
    {
        TensorInt copy;
        if (m_TensorAllocator != null)
            copy = m_TensorAllocator.Alloc(newShape, DataType.Int, m_TensorOnDevice, AllocScope.LayerOutput) as TensorInt;
        else
            copy = new TensorInt(newShape, m_TensorOnDevice, null);

        return copy;
    }

    /// <inheritdoc/>
    public override Tensor DeepCopy()
    {
        var copy = new TensorInt(shape);
        if (shape.HasZeroDims())
            return copy;
        copy.AttachToDevice(m_TensorOnDevice.Clone());
        return copy;
    }

    /// <inheritdoc/>
    public override void UploadToDevice(ITensorData destination)
    {
        var data = m_TensorOnDevice.Download<int>(shape.length);
        destination.Upload(data, shape.length);
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
    public int this[int d7, int d6, int d5, int d4, int d3, int d2, int d1, int d0]
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
    public int this[int d6, int d5, int d4, int d3, int d2, int d1, int d0]
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
    public int this[int d5, int d4, int d3, int d2, int d1, int d0]
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
    public int this[int d4, int d3, int d2, int d1, int d0]
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
    public int this[int d3, int d2, int d1, int d0]
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
    public int this[int d2, int d1, int d0]
    {
        get { return this[shape.RavelIndex(d2, d1, d0)]; }
        set { this[shape.RavelIndex(d2, d1, d0)] = value;}
    }
    /// <summary>
    /// Returns the tensor element at offset `(d1, d0)`, which is position `d1 * stride0 + d0`.
    /// </summary>
    /// <param name="d1">Axis 1.</param>
    /// <param name="d0">Axis 0.</param>
    public int this[int d1, int d0]
    {
        get { return this[shape.RavelIndex(d1, d0)]; }
        set { this[shape.RavelIndex(d1, d0)] = value;}
    }

    /// <summary>
    /// Returns the tensor element at offset `d0`.
    /// </summary>
    /// <param name="d0">Axis 0.</param>
    public int this[int d0]
    {
        get { return base.Get<int>(d0); }
        set { base.Set<int>(d0, value); }
    }

    /// <summary>
    /// Returns a copy of linear memory representation of the data in this tensor.
    ///
    /// If the tensor is the result of computation on a different backend, the method creates a blocking read.
    ///
    /// the returned array is a deepcopy of the tensor, the caller of this methods is now responsible for it.
    /// If you modify the contents of the returned array, it will not modify the underlying tensor
    /// </summary>
    /// <returns>Int array copy of tensor data.</returns>
    public int[] ToReadOnlyArray()
    {
        return base.ToReadOnlyArray<int>();
    }

    /// <summary>
    /// Returns a ReadOnlySpan on the linear memory representation of the data in this tensor.
    /// </summary>
    /// <returns>Span of tensor data.</returns>
    public ReadOnlySpan<int> ToReadOnlySpan()
    {
        return base.ToReadOnlySpan<int>();
    }
}

} // namespace Unity.Sentis
