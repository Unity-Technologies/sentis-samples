using System;
using System.Threading;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine.Assertions;

namespace Unity.Sentis {

/// <summary>
/// An interface that provides methods for converting custom tensor data to `BurstTensorData`.
/// </summary>
public interface IConvertibleToBurstTensorData
{
    /// <summary>
    /// Implement this method to convert to `BurstTensorData`.
    /// </summary>
    /// <param name="shape">The shape of the tensor using the tensor data.</param>
    /// <returns>Converted `BurstTensorData`.</returns>
    BurstTensorData ConvertToBurstTensorData(TensorShape shape);
}

/// <summary>
/// An interface that provides Job system dependency fences for the memory resource.
/// </summary>
public interface IDependableMemoryResource
{
    /// <summary>
    /// A read fence job handle. You can use `fence` as a `dependsOn` argument when you schedule a job that reads data. The job will start when the tensor data is ready for read access.
    /// </summary>
    Unity.Jobs.JobHandle fence { get; set; }
    /// <summary>
    /// A write fence job handle. You can use `reuse` as a `dependsOn` argument when you schedule a job that reads data. The job will start when the tensor data is ready for write access.
    /// </summary>
    Unity.Jobs.JobHandle reuse { get; set; }
    /// <summary>
    /// The raw memory pointer for the resource.
    /// </summary>
    unsafe void* rawPtr { get; }
}

/// <summary>
/// Represents Burst-specific internal data storage for a `Tensor`.
/// </summary>
public class BurstTensorData : ITensorData, IDependableMemoryResource, IConvertibleToComputeTensorData, IConvertibleToArrayTensorData, IReadableTensorData
{
    JobHandle m_ReadFence;
    JobHandle m_WriteFence;
    bool m_SafeToDispose = true;
    NativeTensorArray m_Array;
    int m_Offset;
    int m_Count;
    TensorShape m_Shape;

    /// <summary>
    /// The shape of the tensor using this data as a `TensorShape`.
    /// </summary>
    public TensorShape shape => m_Shape;
    /// <inheritdoc/>
    public virtual DeviceType deviceType => DeviceType.CPU;
    /// <inheritdoc/>
    public int maxCapacity => m_Count;
    /// <summary>
    /// The `NativeTensorArray` managed array containing the `Tensor` data.
    /// </summary>
    public NativeTensorArray array => m_Array;
    /// <summary>
    /// The integer offset for the backing data.
    /// </summary>
    public int offset => m_Offset;
    /// <summary>
    /// The length of the tensor using this data.
    /// </summary>
    public int count => m_Count;

    /// <inheritdoc/>
    public JobHandle fence { get { return m_ReadFence; }  set { m_ReadFence = value; m_WriteFence = value; m_SafeToDispose = false; } }
    /// <inheritdoc/>
    public JobHandle reuse { get { return m_WriteFence; } set { m_WriteFence = JobHandle.CombineDependencies(value, m_WriteFence); m_SafeToDispose = false; } }

    /// <inheritdoc/>
    public unsafe void* rawPtr => m_Array.AddressAt<float>(m_Offset);

    /// <summary>
    /// Initializes and returns an instance of `BurstTensorData`, and allocates storage for a tensor with the shape of `shape`.
    /// </summary>
    /// <param name="shape">The shape of the tensor data to allocate.</param>
    /// <param name="clearOnInit">Whether to zero the data on allocation. The default value is `true`.</param>
    public BurstTensorData(TensorShape shape, bool clearOnInit = true)
    {
        m_Count = shape.length;
        m_Shape = shape;
        m_Array = new NativeTensorArray(m_Count, clearOnInit);
        m_Offset = 0;
    }

    /// <summary>
    /// Initializes and returns an instance of `BurstTensorData` from an `ArrayTensorData`.
    /// </summary>
    /// <param name="sharedArray">The `ArrayTensorData` to convert.</param>
    public BurstTensorData(ArrayTensorData sharedArray)
        : this(sharedArray.shape, sharedArray.array) { }

    /// <summary>
    /// Initializes and returns an instance of `BurstTensorData` from a `SharedArrayTensorData`.
    /// </summary>
    /// <param name="sharedArray">The `SharedArrayTensorData` to convert.</param>
    public BurstTensorData(SharedArrayTensorData sharedArray)
        : this(sharedArray.shape, sharedArray.array, sharedArray.offset) { }

    /// <summary>
    /// Initializes and returns an instance of `BurstTensorData` from a `TensorShape` and an `Array`.
    /// </summary>
    /// <param name="shape">The shape of the tensor data.</param>
    /// <param name="data">The values of the tensor data as an `Array`.</param>
    public BurstTensorData(TensorShape shape, Array data)
        : this(shape, new NativeTensorArrayFromManagedArray(data), 0) { }

    /// <summary>
    /// Initializes and returns an instance of `BurstTensorData` from a `NativeTensorArray` and an offset.
    /// </summary>
    /// <param name="shape">The shape of the tensor data.</param>
    /// <param name="data">The values of the tensor data as a `NativeTensorArray`.</param>
    /// <param name="offset">The integer offset for the backing data.</param>
    public BurstTensorData(TensorShape shape, NativeTensorArray data, int offset = 0)
    {
        m_Count = shape.length;
        m_Shape = shape;
        m_Array = data;
        m_Offset = offset;
        Logger.AssertIsTrue(m_Offset >= 0, "BurstTensorData.ValueError: negative offset {0} not supported", m_Offset);
        Logger.AssertIsTrue(m_Count >= 0, "BurstTensorData.ValueError: negative count {0} not supported", m_Count);
        Logger.AssertIsTrue(m_Offset + m_Count <= m_Array.Length, "BurstTensorData.ValueError: offset + count {0} is bigger than input buffer size {1}, copy will result in a out of bound memory access", m_Offset + m_Count, m_Array.Length);
    }

    /// <inheritdoc/>
    public ITensorData Clone()
    {
        return new BurstTensorData(m_Shape, m_Array, m_Offset);
    }

    /// <summary>
    /// Finalizes the `BurstTensorData`.
    /// </summary>
    ~BurstTensorData()
    {
        if (!m_SafeToDispose)
            D.LogWarning($"Found unreferenced, but undisposed BurstTensorData that potentially participates in an unfinished job and might lead to hazardous memory overwrites");
    }

    /// <summary>
    /// Disposes of the `BurstTensorData` and any associated memory.
    /// </summary>
    public void Dispose()
    {
        // It isn't safe to Complete jobs from a finalizer thread, so
        if (Thread.CurrentThread == CPUBackend.MainThread)
            CompleteAllPendingOperations();
    }

    /// <inheritdoc/>
    public void CompleteAllPendingOperations()
    {
        fence.Complete();
        reuse.Complete();
        m_SafeToDispose = true;
    }

    /// <summary>
    /// Reserves storage for `count` elements.
    /// </summary>
    /// <param name="count">The number of elements to reserve.</param>
    public void Reserve(int count)
    {
        if (count > maxCapacity)
        {
            // going to reallocate memory in base.Reserve()
            // thus need to finish current work
            CompleteAllPendingOperations();
        }
        if (count > maxCapacity)
        {
            m_Array = new NativeTensorArray(count);
            m_Offset = 0;
            m_Count = m_Array.Length;
        }
    }

    /// <summary>
    /// Uploads data to internal storage.
    /// </summary>
    /// <param name="data">The data to upload as a native array.</param>
    /// <param name="srcCount">The number of elements to upload.</param>
    /// <param name="srcOffset">The index of the first element in the native array.</param>
    /// <typeparam name="T">The data type of the elements.</typeparam>
    public void Upload<T>(NativeArray<T> data, int srcCount, int srcOffset = 0) where T : unmanaged
    {
        CompleteAllPendingOperations();

        var numItemToCopy = srcCount;
        var numItemAvailableInData = data.Length - srcOffset;
        Assert.IsTrue(srcOffset >= 0);
        Assert.IsTrue(numItemToCopy <= numItemAvailableInData);

        Reserve(numItemToCopy);
        NativeTensorArray.Copy(data, srcOffset, m_Array, m_Offset, numItemToCopy);
    }

    /// <summary>
    /// Returns data from internal storage.
    /// </summary>
    /// <param name="dstCount">The number of elements to download.</param>
    /// <param name="srcOffset">The index of the first element in the data.</param>
    /// <typeparam name="T">The data type of the elements.</typeparam>
    /// <returns>The downloaded data as a native array.</returns>
    public NativeArray<T> Download<T>(int dstCount, int srcOffset = 0) where T : unmanaged
    {
        // Download() as optimization gives direct access to the internal buffer
        // thus need to prepare internal buffer for potential writes
        CompleteAllPendingOperations();

        var downloadCount = dstCount;
        Logger.AssertIsTrue(m_Count >= downloadCount, "SharedArrayTensorData.Download.ValueError: cannot download {0} items from tensor of size {1}", downloadCount, m_Count);

        var dest = new NativeArray<T>(downloadCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
        NativeTensorArray.Copy(m_Array, srcOffset + m_Offset, dest, 0, downloadCount);
        return dest;
    }

    /// <inheritdoc/>
    public T Get<T>(int index) where T : unmanaged
    {
        CompleteAllPendingOperations();
        return m_Array.Get<T>(m_Offset + index);
    }

    /// <inheritdoc/>
    public void Set<T>(int index, T value) where T : unmanaged
    {
        CompleteAllPendingOperations();
        m_Array.Set<T>(m_Offset + index, value);
    }

    /// <inheritdoc/>
    public ReadOnlySpan<T> ToReadOnlySpan<T>(int dstCount, int srcOffset = 0) where T : unmanaged
    {
        CompleteAllPendingOperations();
        return m_Array.AsReadOnlySpan<T>(dstCount, m_Offset + srcOffset);
    }

    /// <inheritdoc/>
    public NativeArray<T>.ReadOnly GetReadOnlyNativeArrayHandle<T>(int dstCount, int srcOffset = 0) where T : unmanaged
    {
        CompleteAllPendingOperations();
        return m_Array.GetReadOnlyNativeArrayHandle<T>(dstCount, m_Offset + srcOffset);
    }

    /// <inheritdoc/>
    public T[] ToArray<T>(int dstCount, int srcOffset = 0) where T : unmanaged
    {
        CompleteAllPendingOperations();
        return m_Array.ToArray<T>(dstCount, m_Offset + srcOffset);
    }

    /// <inheritdoc/>
    public ComputeTensorData ConvertToComputeTensorData(TensorShape shape)
    {
        CompleteAllPendingOperations();
        return new ComputeTensorData(shape, array, offset);
    }

    /// <inheritdoc/>
    public ArrayTensorData ConvertToArrayTensorData(TensorShape shape)
    {
        CompleteAllPendingOperations();
        return new ArrayTensorData(shape, array, offset, clearOnInit: false);
    }

    /// <inheritdoc/>
    public bool IsAsyncReadbackRequestDone()
    {
        return fence.IsCompleted;
    }

    /// <inheritdoc/>
    public void AsyncReadbackRequest(Action<bool> callback = null)
    {
        fence.Complete();
        callback?.Invoke(true);
    }

    /// <summary>
    /// Returns a string that represents the `BurstTensorData`.
    /// </summary>
    /// <returns>The string summary of the `BurstTensorData`.</returns>
    public override string ToString()
    {
        return string.Format("(CPU burst: [{0}], offset: {1} uploaded: {2})", m_Array?.Length, m_Offset, m_Count);
    }

    /// <summary>
    /// Moves a tensor into memory on the CPU backend device.
    /// </summary>
    /// <param name="X">The `Tensor` to move to the CPU.</param>
    /// <param name="clearOnInit">Whether to initialize the backend data. The default value is `true`.</param>
    /// <returns>The pinned `BurstTensorData`.</returns>
    public static BurstTensorData Pin(Tensor X, bool clearOnInit = true)
    {
        var onDevice = X.tensorOnDevice;
        if (onDevice == null)
        {
            X.AttachToDevice(new BurstTensorData(X.shape, clearOnInit));
            return X.tensorOnDevice as BurstTensorData;
        }

        if (onDevice is BurstTensorData)
            return onDevice as BurstTensorData;

        if (onDevice is IConvertibleToBurstTensorData asConvertible)
            X.AttachToDevice(asConvertible.ConvertToBurstTensorData(X.shape));
        else
            X.UploadToDevice(new BurstTensorData(X.shape, clearOnInit: false)); // device is not compatible, create new array and upload

        return X.tensorOnDevice as BurstTensorData;
    }
}
} // namespace Unity.Sentis
