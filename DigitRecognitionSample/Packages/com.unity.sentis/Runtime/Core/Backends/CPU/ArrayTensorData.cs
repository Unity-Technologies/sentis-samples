using System;
using Unity.Collections;

namespace Unity.Sentis {

/// <summary>
/// An interface that provides methods for converting custom tensor data to `ArrayTensorData`.
/// </summary>
public interface IConvertibleToArrayTensorData
{
    /// <summary>
    /// Implement this method to convert to `ArrayTensorData`.
    /// </summary>
    /// <param name="shape">The shape of the tensor using the tensor data.</param>
    /// <returns>Converted `ArrayTensorData`.</returns>
    ArrayTensorData ConvertToArrayTensorData(TensorShape shape);
}

/// <summary>
/// Represents internal `Tensor` data backed by a managed array.
/// </summary>
public class ArrayTensorData : ITensorData, IConvertibleToBurstTensorData, IConvertibleToComputeTensorData, IReadableTensorData
{
    NativeTensorArray m_Array;
    /// <summary>
    /// The `NativeTensorArray` managed array containing the `Tensor` data.
    /// </summary>
    public NativeTensorArray array => m_Array;

    TensorShape m_Shape;

    /// <summary>
    /// The shape of the tensor using this data as a `TensorShape`.
    /// </summary>
    public TensorShape shape => m_Shape;

    /// <inheritdoc/>
    public int maxCapacity => m_Array.Length;

    /// <inheritdoc/>
    public DeviceType deviceType => DeviceType.CPU;

    /// <summary>
    /// Initializes and returns an instance of `ArrayTensorData`, and allocates storage for a tensor with the shape of `shape`.
    /// </summary>
    /// <param name="shape">The shape of the tensor data to allocate.</param>
    /// <param name="clearOnInit">Whether to zero the data on allocation. The default value is `true`.</param>
    public ArrayTensorData(TensorShape shape, bool clearOnInit = true)
    {
        m_Shape = shape;
        m_Array = new NativeTensorArray(m_Shape.length, clearOnInit);
    }

    /// <summary>
    /// Initializes and returns an instance of `ArrayTensorData` with given storage and offset.
    /// </summary>
    /// <param name="shape">The shape of the tensor data.</param>
    /// <param name="data">The allocated data to use as backing data.</param>
    /// <param name="offset">The integer offset from the start of the backing array. The default value is 0.</param>
    /// <param name="clearOnInit">Whether to zero the data on instantiation. The default value is `true`.</param>
    public ArrayTensorData(TensorShape shape, NativeTensorArray data, int offset = 0, bool clearOnInit = true)
    {
        m_Shape = shape;
        m_Array = new NativeTensorArray(m_Shape.length, clearOnInit);
        NativeTensorArray.Copy(data, offset, m_Array, 0, m_Shape.length);
    }

    /// <summary>
    /// Initializes and returns an instance of `ArrayTensorData` with a given array of data.
    /// </summary>
    /// <param name="shape">The shape of the tensor data.</param>
    /// <param name="data">The data to use as backing data.</param>
    /// <param name="clearOnInit">Whether to zero the data on instantiation. The default value is `true`.</param>
    public ArrayTensorData(TensorShape shape, Array data, bool clearOnInit = true)
    {
        m_Shape = shape;
        m_Array = new NativeTensorArrayFromManagedArray(data);
    }

    /// <inheritdoc/>
    public ITensorData Clone()
    {
        return new ArrayTensorData(m_Shape, m_Array);
    }

    /// <summary>
    /// Disposes of the `ArrayTensorData` and any associated memory.
    /// </summary>
    public void Dispose()
    {
        m_Array = null;
    }

    /// <inheritdoc/>
    public void Reserve(int count)
    {
        if (count > m_Array.Length)
            m_Array = new NativeTensorArray(count);
    }

    /// <inheritdoc/>
    public void Upload<T>(NativeArray<T> data, int srCount, int srcOffset = 0) where T : unmanaged
    {
        var numItemToCopy = srCount;
        var numItemAvailableInData = data.Length - srcOffset;

        Logger.AssertIsTrue(srcOffset >= 0, "ArrayTensorData.Upload.ValueError: negative start index {0}, not supported", srcOffset);
        Logger.AssertIsTrue(numItemToCopy <= numItemAvailableInData, "ArrayTensorData.Upload.ValueError: cannot copy {0} items from data of size {1}", numItemToCopy, numItemAvailableInData);

        Reserve(numItemToCopy);
        NativeTensorArray.Copy(data, srcOffset, m_Array, 0, numItemToCopy);
    }

    /// <inheritdoc/>
    public bool IsAsyncReadbackRequestDone()
    {
        return true;
    }

    /// <inheritdoc/>
    public void AsyncReadbackRequest(Action<bool> callback = null)
    {
        callback?.Invoke(true);
    }

    /// <inheritdoc/>
    public void CompleteAllPendingOperations() { }

    /// <inheritdoc/>
    public NativeArray<T> Download<T>(int dstCount, int srcOffset = 0) where T : unmanaged
    {
        var downloadCount = dstCount;
        Logger.AssertIsTrue(m_Array.Length >= downloadCount, "ArrayTensorData.Download.ValueError: cannot download {0} items from tensor of size {1}", downloadCount, m_Array.Length);

        var dest = new NativeArray<T>(downloadCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
        NativeTensorArray.Copy(m_Array, srcOffset, dest, 0, downloadCount);

        return dest;
    }

    /// <inheritdoc/>
    public T Get<T>(int index) where T : unmanaged
    {
        return m_Array.Get<T>(index);
    }

    /// <inheritdoc/>
    public void Set<T>(int index, T value) where T : unmanaged
    {
        m_Array.Set<T>(index, value);
    }

    /// <inheritdoc/>
    public ReadOnlySpan<T> ToReadOnlySpan<T>(int dstCount, int srcOffset = 0) where T : unmanaged
    {
        return m_Array.AsReadOnlySpan<T>(dstCount, srcOffset);
    }

    /// <inheritdoc/>
    public NativeArray<T>.ReadOnly GetReadOnlyNativeArrayHandle<T>(int dstCount, int srcOffset = 0) where T : unmanaged
    {
        return m_Array.GetReadOnlyNativeArrayHandle<T>(dstCount, srcOffset);
    }

    /// <inheritdoc/>
    public T[] ToArray<T>(int dstCount, int srcOffset = 0) where T : unmanaged
    {
        return m_Array.ToArray<T>(dstCount, srcOffset);
    }

    /// <inheritdoc/>
    public ComputeTensorData ConvertToComputeTensorData(TensorShape shape)
    {
        return new ComputeTensorData(shape, array);
    }

    /// <inheritdoc/>
    public BurstTensorData ConvertToBurstTensorData(TensorShape shape)
    {
        return new BurstTensorData(shape, array);
    }

    /// <summary>
    /// Returns a summary of the storage used by the tensor array, as a string.
    /// </summary>
    /// <returns>The string summary of the `ArrayTensorData`.</returns>
    public override string ToString()
    {
        return string.Format("(CPU array [{0}])", m_Array?.Length);
    }

    /// <summary>
    /// Moves the tensor into memory on the CPU backend device.
    /// </summary>
    /// <param name="X">The `Tensor` to move to the CPU.</param>
    /// <param name="clearOnInit">Whether to initialize the backend data. The default value is `true`.</param>
    /// <returns>The pinned `ArrayTensorData`.</returns>
    public static ArrayTensorData Pin(Tensor X, bool clearOnInit = true)
    {
        var onDevice = X.tensorOnDevice;
        if (onDevice == null)
        {
            X.AttachToDevice(new ArrayTensorData(X.shape, clearOnInit));
            return X.tensorOnDevice as ArrayTensorData;
        }

        if (onDevice is ArrayTensorData)
            return onDevice as ArrayTensorData;

        if (onDevice is IConvertibleToArrayTensorData asConvertible)
            X.AttachToDevice(asConvertible.ConvertToArrayTensorData(X.shape));
        else
            X.UploadToDevice(new ArrayTensorData(X.shape, clearOnInit: false)); // device is not compatible, create new array and upload

        return X.tensorOnDevice as ArrayTensorData;
    }
}

/// <summary>
/// Represents internal `Tensor` data backed by a managed array and shared between multiple tensors.
/// </summary>
public class SharedArrayTensorData : ITensorData, IConvertibleToBurstTensorData, IConvertibleToComputeTensorData, IConvertibleToArrayTensorData, IReadableTensorData
{
    internal NativeTensorArray m_Array;
    internal int m_Offset;
    internal int m_Count;
    TensorShape m_Shape;

    /// <summary>
    /// The shape of the tensor using this data as a `TensorShape`.
    /// </summary>
    public TensorShape shape => m_Shape;

    /// <inheritdoc/>
    public virtual int maxCapacity => m_Count;

    /// <inheritdoc/>
    public DeviceType deviceType => DeviceType.CPU;

    /// <summary>
    /// The data storage array
    /// </summary>
    public NativeTensorArray array => m_Array;

    /// <summary>
    /// Offset in the storage array.
    /// </summary>
    public int offset => m_Offset;

    /// <summary>
    /// The number of data elements.
    /// </summary>
    public int count => m_Count;

    /// <summary>
    /// Initializes and returns an instance of `SharedArrayTensorData` with a given array of data.
    /// </summary>
    /// <param name="shape">The shape of the tensor data.</param>
    /// <param name="data">The data to use as backing data.</param>
    public SharedArrayTensorData(TensorShape shape, Array data)
        : this(shape, new NativeTensorArrayFromManagedArray(data), 0) { }

    /// <summary>
    /// Initializes and returns an instance of `SharedArrayTensorData` with a given `NativeTensorArray` of data.
    /// </summary>
    /// <param name="shape">The shape of the tensor data.</param>
    /// <param name="data">The allocated data to use as backing data.</param>
    /// <param name="offset">The integer offset to use for the data.</param>
    public SharedArrayTensorData(TensorShape shape, NativeTensorArray data, int offset = 0)
    {
        m_Count = shape.length;
        m_Shape = shape;
        m_Array = data;
        m_Offset = offset;
        Logger.AssertIsTrue(m_Offset >= 0, "SharedArrayTensorData.ValueError: negative offset {0} not supported", m_Offset);
        Logger.AssertIsTrue(m_Count >= 0, "SharedArrayTensorData.ValueError: negative count {0} not supported", m_Count);
        Logger.AssertIsTrue(m_Offset + m_Count <= m_Array.Length, "SharedArrayTensorData.ValueError: offset + count {0} is bigger than input buffer size {1}, copy will result in a out of bound memory access", m_Offset + m_Count, m_Array.Length);
    }

    /// <inheritdoc/>
    public ITensorData Clone()
    {
        return new SharedArrayTensorData(m_Shape, m_Array, m_Offset);
    }

    /// <summary>
    /// Finalizes the `SharedArrayTensorData`.
    /// </summary>
    ~SharedArrayTensorData()
    {
        Dispose();
    }

    /// <summary>
    /// Disposes of the `SharedArrayTensorData` and any associated memory.
    /// </summary>
    public void Dispose() { }

    /// <inheritdoc/>
    public void Reserve(int count)
    {
        // currently always readonly
        throw new InvalidOperationException("SharedArrayTensorData is readonly!");
    }

    /// <inheritdoc/>
    public void Upload<T>(NativeArray<T> data, int srcCount, int srcOffset = 0) where T : unmanaged
    {
        // currently always readonly
        throw new InvalidOperationException("SharedArrayTensorData is readonly!");
    }

    /// <inheritdoc/>
    public bool IsAsyncReadbackRequestDone()
    {
        return true;
    }

    /// <inheritdoc/>
    public void AsyncReadbackRequest(Action<bool> callback = null)
    {
        callback?.Invoke(true);
    }

    /// <inheritdoc/>
    public void CompleteAllPendingOperations() { }

    /// <inheritdoc/>
    public NativeArray<T> Download<T>(int dstCount, int srcOffset = 0) where T : unmanaged
    {
        var downloadCount = dstCount;
        Logger.AssertIsTrue(m_Count >= downloadCount, "SharedArrayTensorData.Download.ValueError: cannot download {0} items from tensor of size {1}", downloadCount, m_Count);

        var dest = new NativeArray<T>(downloadCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
        NativeTensorArray.Copy(m_Array, srcOffset + m_Offset, dest, 0, downloadCount);
        return dest;
    }

    /// <inheritdoc/>
    public T Get<T>(int index) where T : unmanaged
    {
        return m_Array.Get<T>(m_Offset + index);
    }

    /// <inheritdoc/>
    public void Set<T>(int index, T value) where T : unmanaged
    {
        m_Array.Set<T>(m_Offset + index, value);
    }

    /// <inheritdoc/>
    public ReadOnlySpan<T> ToReadOnlySpan<T>(int dstCount, int srcOffset = 0) where T : unmanaged
    {
        return m_Array.AsReadOnlySpan<T>(dstCount, m_Offset + srcOffset);
    }

    /// <inheritdoc/>
    public NativeArray<T>.ReadOnly GetReadOnlyNativeArrayHandle<T>(int dstCount, int srcOffset = 0) where T : unmanaged
    {
        return m_Array.GetReadOnlyNativeArrayHandle<T>(dstCount, m_Offset + srcOffset);
    }

    /// <inheritdoc/>
    public T[] ToArray<T>(int dstCount, int srcOffset = 0) where T : unmanaged
    {
        return m_Array.ToArray<T>(dstCount, m_Offset + srcOffset);
    }

    /// <inheritdoc/>
    public ComputeTensorData ConvertToComputeTensorData(TensorShape shape)
    {
        return new ComputeTensorData(shape, array, offset);
    }

    /// <inheritdoc/>
    public BurstTensorData ConvertToBurstTensorData(TensorShape shape)
    {
        return new BurstTensorData(shape, array, offset);
    }

    /// <inheritdoc/>
    public ArrayTensorData ConvertToArrayTensorData(TensorShape shape)
    {
        return new ArrayTensorData(shape, array, offset, clearOnInit: false);
    }

    /// <summary>
    /// Returns a string that represents the `SharedArrayTensorData`.
    /// </summary>
    /// <returns>The string summary of the `SharedArrayTensorData`.</returns>
    public override string ToString()
    {
        return string.Format("(CPU shared[{0}], offset: {1} count: {2})",
            m_Array.Length, m_Offset, m_Count);
    }
}
} // namespace Unity.Sentis
