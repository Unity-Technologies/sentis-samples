using UnityEngine.Assertions;
using System;
using Unity.Collections;

namespace Unity.Sentis {

/// <summary>
/// Represents data in a multidimensional array-like structure.
/// </summary>
public abstract class Tensor : IDisposable
{
    private protected ITensorData m_TensorOnDevice;
    private protected ITensorAllocator m_TensorAllocator;
    private protected TensorShape m_Shape;
    private protected bool m_Disposed = false;

    /// <summary>
    /// The data type of the elements of the tensor.
    /// </summary>
    public abstract DataType dataType { get; }

    /// <summary>
    /// The shape of the tensor, as a `TensorShape`.
    /// </summary>
    public TensorShape shape
    {
        get => m_Shape;
        protected set => m_Shape = value;
    }

    /// <summary>
    /// The device-specific internal representation of the tensor data.
    /// </summary>
    public ITensorData tensorOnDevice
    {
        get => m_TensorOnDevice;
        protected set => m_TensorOnDevice = value;
    }

    /// <summary>
    /// The allocator for the tensor. Refer to <see cref="ITensorAllocator"/>.
    /// </summary>
    public ITensorAllocator allocator => m_TensorAllocator;

    internal bool disposed => m_Disposed;

    /// <summary>
    /// Instantiates and returns a Tensor with the specified `shape`, an `ITensorData` `data` and an ITensorAllocator `allocator`.
    /// </summary>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="data">The optional tensor data.</param>
    /// <param name="allocator">The allocator that was used to instantiate the tensor.</param>
    protected internal Tensor(TensorShape shape, ITensorData data = null, ITensorAllocator allocator = null)
    {
        m_Shape = shape;
        tensorOnDevice = data;
        m_TensorAllocator = allocator;
    }

    /// <summary>
    /// Dispose of the tensor and any associated memory.
    /// </summary>
    ~Tensor()
    {
        Dispose();
    }

    private protected void PinToDevice(ITensorData onDevice, bool disposeUnpinned = true)
    {
        Logger.AssertIsTrue(onDevice?.maxCapacity >= shape.length || onDevice == null, "Tensor.PinToDevice: not enough capacity on device to pin tensor or device null");

        if (m_TensorAllocator != null)
            m_TensorAllocator.MoveToDevice(this, onDevice, m_TensorOnDevice, disposeUnpinned);
        else if (disposeUnpinned)
            m_TensorOnDevice?.Dispose();

        tensorOnDevice = onDevice;
    }

    /// <summary>
    /// Associates a tensor with the block of data on a device. Sentis downloads from `source` on first access.
    ///
    /// Make sure `source` contains initialized and valid data that represents tensor values.
    /// </summary>
    /// <param name="source">The data on device to associate to the tensor.</param>
    public void AttachToDevice(ITensorData source)
    {
        if (m_TensorOnDevice == source)
            return;

        PinToDevice(source, disposeUnpinned: true);
    }

    /// <summary>
    /// Uploads the tensor data to the destination data location on device.
    /// </summary>
    /// <param name="destination">The data on device to upload the tensor data to.</param>
    public abstract void UploadToDevice(ITensorData destination);

    /// <summary>
    /// Synchronizes the tensor data with the data on the device, then remove the tensor from the device.
    /// </summary>
    /// <param name="disposeDeviceData">Whether to free the space on device after detaching.</param>
    /// <returns>The detached tensor data.</returns>
    public ITensorData DetachFromDevice(bool disposeDeviceData = true)
    {
        ITensorData unpinned = (disposeDeviceData) ? null : m_TensorOnDevice;
        PinToDevice(null, disposeDeviceData);
        return unpinned;
    }

    /// <summary>
    /// Blocking call to make tensor data read write.
    ///
    /// Issues a blocking download of the internal data. And converts tensorData to ArrayTensorData
    /// </summary>
    public void MakeReadable()
    {
        m_TensorOnDevice?.CompleteAllPendingOperations();
        if (tensorOnDevice is IReadableTensorData)
            return;
        ArrayTensorData.Pin(this, clearOnInit: false);
    }

    /// <summary>
    /// Checks if asynchronous readback request it done.
    ///
    /// Returns true if async readback is successful.
    /// </summary>
    /// <returns>Whether the async readback request is successful.</returns>
    public bool IsAsyncReadbackRequestDone()
    {
        if (m_TensorOnDevice == null)
            return false;

        return m_TensorOnDevice.IsAsyncReadbackRequestDone();
    }

    /// <summary>
    /// Schedules asynchronous download of the internal data.
    /// </summary>
    /// <param name="callback">Callback invoked when async readback is finished. Return value indicates if async readback is successful.</param>
    public void AsyncReadbackRequest(Action<bool> callback = null)
    {
        if (m_TensorOnDevice == null)
        {
            callback?.Invoke(false);
            return;
        }

        m_TensorOnDevice.AsyncReadbackRequest(callback);
    }

    /// <summary>
    /// Completes all scheduled tensor operations on device.
    /// </summary>
    public void CompleteAllPendingOperations()
    {
        m_TensorOnDevice.CompleteAllPendingOperations();
    }

    /// <summary>
    /// Returns a shallow copy of the `Tensor` with a new shape. The copy shares data storage with the original tensor.
    ///
    /// `newShape.length` must be equal to `this.shape.length`.
    /// </summary>
    /// <param name="newShape">The shape of the returned tensor.</param>
    /// <returns>The tensor with the new shape sharing the same data storage as the original tensor.</returns>
    public abstract Tensor ShallowReshape(TensorShape newShape);

    /// <summary>
    /// Returns a shallow copy of the current `Tensor`. The copy shares data storage with original tensor.
    /// </summary>
    /// <returns>The copied tensor sharing the same data storage as the original tensor.</returns>
    public Tensor ShallowCopy()
    {
        return ShallowReshape(shape);
    }

    /// <summary>
    /// Returns a deep copy of the current Tensor.
    /// </summary>
    /// <returns>The copied tensor.</returns>
    public abstract Tensor DeepCopy();

    /// <summary>
    /// Removes system references to the tensor. The caller assumes ownership.
    /// </summary>
    public void TakeOwnership()
    {
        m_TensorAllocator?.WaiveOwnership(this);
        m_TensorAllocator = null;
    }

    /// Called from ITensorAllocator, puts Tensor in the ready for reuse state.
    internal ITensorData Invalidate()
    {
        ITensorData unpinned = m_TensorOnDevice;
        PinToDevice(null, false);
        Assert.AreEqual(m_TensorOnDevice, null, "Tensor.Invalidate: tensorOnDevice not null");
        tensorOnDevice = null;
        m_TensorAllocator = null;
        return unpinned;
    }

    internal void Init(TensorShape shape, ITensorData buffer, ITensorAllocator allocator)
    {
        this.shape = shape;
        tensorOnDevice = buffer;
        m_TensorAllocator = allocator;
        m_Disposed = false;
    }

    /// <summary>
    /// Disposes of the tensor and any associated memory.
    /// </summary>
    public virtual void Dispose()
    {
        if (m_TensorAllocator != null)
        {
            m_TensorAllocator.Release(this, true);
        }
        else if (m_TensorOnDevice != null)
        {
            m_TensorOnDevice.Dispose();
        }

        tensorOnDevice = null;
        m_TensorAllocator = null;
        m_Disposed = true;
    }

    /// <summary>
    /// Returns a string that represents the `Tensor`.
    /// </summary>
    /// <returns>String representation of tensor.</returns>
    public override string ToString()
    {
        return $"Tensor{dataType}{shape}";
    }

    internal T[] ToReadOnlyArray<T>() where T : unmanaged
    {
        if (m_TensorOnDevice is IReadableTensorData rwData)
            return rwData.ToArray<T>(shape.length);
        else
            throw new InvalidOperationException("Tensor data cannot be read from, use .MakeReadable() to allow reading from tensor.");
    }

    internal NativeArray<T>.ReadOnly ToReadOnlyNativeArray<T>() where T : unmanaged
    {
        if (m_TensorOnDevice is IReadableTensorData rwData)
            return rwData.GetReadOnlyNativeArrayHandle<T>(shape.length);
        else
            throw new InvalidOperationException("Tensor data cannot be read from, use .MakeReadable() to allow reading from tensor.");
    }

    internal ReadOnlySpan<T> ToReadOnlySpan<T>() where T : unmanaged
    {
        if (m_TensorOnDevice is IReadableTensorData rwData)
            return rwData.ToReadOnlySpan<T>(shape.length);
        else
            throw new InvalidOperationException("Tensor data cannot be read from, use .MakeReadable() to allow reading from tensor.");
    }

    internal T Get<T>(int d0) where T : unmanaged
    {
        if (m_TensorOnDevice is IReadableTensorData rwData)
            return rwData.Get<T>(d0);
        else
            throw new InvalidOperationException("Tensor data cannot be read from, use .MakeReadable() to allow reading from tensor.");
    }

    internal void Set<T>(int d0, T value) where T : unmanaged
    {
        if (m_TensorOnDevice is IReadableTensorData rwData)
            rwData.Set<T>(d0, value);
        else
            throw new InvalidOperationException("Tensor data cannot be written to, use .MakeReadable() to allow writing to tensor.");
    }
}

} // namespace Unity.Sentis
