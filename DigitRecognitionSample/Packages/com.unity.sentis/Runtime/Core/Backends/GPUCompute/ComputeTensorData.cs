using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Profiling;
using UnityEngine.Rendering;
using System;
using System.Threading;
using Unity.Collections;

namespace Unity.Sentis {
/// <summary>
/// An interface that provides methods for converting custom tensor data to `ComputeTensorData`.
/// </summary>
public interface IConvertibleToComputeTensorData
{
    /// <summary>
    /// Implement this method to convert to `ComputeTensorData`.
    /// </summary>
    /// <param name="shape">The shape of the tensor using the tensor data.</param>
    /// <returns>Converted `ComputeTensorData`.</returns>
    ComputeTensorData ConvertToComputeTensorData(TensorShape shape);
}

/// <summary>
/// Represents data storage for a `Tensor` as a compute buffer, for GPUCompute backend.
/// </summary>
public class ComputeTensorData : ITensorData
{
    bool m_DisposeBufferAfterUse;
    ComputeBuffer m_Buffer;
    TensorShape m_Shape;

    /// <inheritdoc/>
    public int maxCapacity => m_Shape.length;

    /// <inheritdoc/>
    public DeviceType deviceType => DeviceType.GPU;

    /// <summary>
    /// The shape of the tensor using this data as a `TensorShape`.
    /// </summary>
    public TensorShape shape => m_Shape;

    /// <summary>
    /// The data storage as a compute buffer.
    /// </summary>
    public ComputeBuffer buffer => m_Buffer;

    /// <summary>
    /// Initializes and returns an instance of `ComputeTensorData`, and allocates storage for a tensor with the shape of `shape`.
    /// </summary>
    /// <param name="shape">The shape of the tensor data to allocate.</param>
    /// <param name="clearOnInit">Whether to zero the data on allocation. The default value is `true`.</param>
    public ComputeTensorData(TensorShape shape, bool clearOnInit = true)
    {
        // Minimum size of 1 to handle 0-dim tensors.
        m_Buffer = new ComputeBuffer(Math.Max(1, shape.length), sizeof(float));

        // @TODO: consider zero initialization only for "debug" mode
        if (clearOnInit)
        {
            var empty = new NativeArray<float>(shape.length, Allocator.Temp, NativeArrayOptions.ClearMemory);
            m_Buffer.SetData(empty);
            empty.Dispose();
        }

        m_Shape = shape;

        m_DisposeBufferAfterUse = true;
    }

    /// <inheritdoc/>
    public ITensorData Clone()
    {
        var copy = new ComputeTensorData(m_Shape);

        int length = m_Buffer.count;

        var fn = ComputeFuncSingleton.Instance.Get("MemCopy");
        fn.SetTensorAsBuffer(ShaderPropertyID.k_ID_Xptr, this);
        fn.SetTensorAsBuffer(ShaderPropertyID.k_ID_Optr, copy);
        fn.SetInt(ShaderPropertyID.k_ID_offsetX, 0);
        fn.SetInt(ShaderPropertyID.k_ID_offsetO, 0);
        fn.SetInt(ShaderPropertyID.k_ID_count, length);
        fn.Dispatch(ComputeHelper.IDivC(length, 4), 1, 1);

        return copy;
    }

    /// <summary>
    /// Initializes and returns an instance of `ComputeTensorData` with given data and offset.
    /// </summary>
    /// <param name="shape">The shape of the tensor data.</param>
    /// <param name="array">The allocated data to use as backing data.</param>
    /// <param name="offset">The integer offset from the start of the backing array. The default value is 0.</param>
    public ComputeTensorData(TensorShape shape, NativeTensorArray array, int offset = 0)
    {
        // Minimum size of 1 to handle 0-dim tensors.
        m_Buffer = new ComputeBuffer(Math.Max(1, shape.length), sizeof(float));
        if (shape.length != 0)
            m_Buffer.SetData(array.GetNativeArrayHandle<float>(), offset, 0, shape.length);

        m_Shape = shape;

        m_DisposeBufferAfterUse = true;
    }

    /// <summary>
    /// Finalizes the `ComputeTensorData`.
    /// </summary>
    ~ComputeTensorData()
    {
        if (m_Buffer == null)
            return;
        if (!m_DisposeBufferAfterUse)
            return;

        D.LogWarning($"Found unreferenced, but undisposed ComputeTensorData which might lead to GPU resource leak");
    }

    /// <summary>
    /// Disposes of the `ComputeTensorData` and any associated memory.
    /// </summary>
    public void Dispose()
    {
        // It isn't safe to Release RT from a finalizer thread
        if (Thread.CurrentThread == CPUBackend.MainThread)
        {
            if (m_DisposeBufferAfterUse)
            {
                m_Buffer.Dispose();
                m_Buffer = null;
            }

            m_DisposeBufferAfterUse = false;
        }
    }

    /// <inheritdoc/>
    public void Reserve(int count)
    {
        if (count > maxCapacity)
        {
            m_Buffer.Dispose();
            m_Buffer = new ComputeBuffer(count, sizeof(float));
        }
    }

    /// <inheritdoc/>
    public void Upload<T>(NativeArray<T> data, int srcCount, int srcOffset = 0) where T : unmanaged
    {
        var numItemToCopy = srcCount;
        var numItemAvailableInData = data.Length - srcOffset;

        Assert.IsTrue(srcOffset >= 0);
        Assert.IsTrue(numItemToCopy <= numItemAvailableInData);
        m_Buffer.SetData(data, srcOffset, 0, numItemToCopy);

        m_AsyncDownloadRequested = false;
    }

    bool m_AsyncDownloadRequested = false;
    AsyncGPUReadbackRequest m_AsyncDownloadRequest;

    /// <inheritdoc/>
    public bool IsAsyncReadbackRequestDone()
    {
        if (m_AsyncDownloadRequested)
        {
            if (m_AsyncDownloadRequest.hasError)
                m_AsyncDownloadRequested = false;
            else
                m_AsyncDownloadRequest.Update();
        }

        return m_AsyncDownloadRequest.done;
    }

    /// <inheritdoc/>
    public void AsyncReadbackRequest(Action<bool> callback = null)
    {
        if (!SystemInfo.supportsAsyncGPUReadback)
        {
            callback?.Invoke(false);
            return;
        }

        Action<AsyncGPUReadbackRequest> task = request =>
        {
            callback?.Invoke(!request.hasError);
        };
        m_AsyncDownloadRequest = AsyncGPUReadback.Request(m_Buffer, m_Buffer.count * sizeof(float), 0 * sizeof(float), task);
        m_AsyncDownloadRequested = true;
    }

    /// <inheritdoc/>
    public NativeArray<T> Download<T>(int dstCount, int srcOffset = 0) where T : unmanaged
    {
        var count = dstCount;

        Assert.IsTrue(maxCapacity >= count);
        count = Math.Min(maxCapacity, count);

        if (count == 0)
            return new NativeArray<T>(0, Allocator.Temp);

        Profiler.BeginSample("Sentis.ComputeTensorData.DownloadDataFromGPU");

        if (m_AsyncDownloadRequested)
        {
            m_AsyncDownloadRequested = false;
            if (!m_AsyncDownloadRequest.done)
                m_AsyncDownloadRequest.WaitForCompletion();

            var reqData = m_AsyncDownloadRequest.GetData<T>();
            Profiler.EndSample();
            return reqData;
        }

        if (!SystemInfo.supportsAsyncGPUReadback)
        {
            var dataArray = new T[count];
            m_Buffer.GetData(dataArray, 0, srcOffset, count);
            return new NativeArray<T>(dataArray, Allocator.Temp);
        }

        m_AsyncDownloadRequest = AsyncGPUReadback.Request(m_Buffer, dstCount * sizeof(float), srcOffset * sizeof(float));
        m_AsyncDownloadRequest.WaitForCompletion();

        var data = m_AsyncDownloadRequest.GetData<T>();

        Profiler.EndSample();

        return data;
    }

    /// <inheritdoc/>
    public void CompleteAllPendingOperations()
    {
        if (m_AsyncDownloadRequested)
        {
            if (!m_AsyncDownloadRequest.done)
                m_AsyncDownloadRequest.WaitForCompletion();
            return;
        }

        m_AsyncDownloadRequest = AsyncGPUReadback.Request(m_Buffer, m_Buffer.count * sizeof(float), 0 * sizeof(float));
        m_AsyncDownloadRequest.WaitForCompletion();
        m_AsyncDownloadRequested = false;
    }

    /// <summary>
    /// Returns a string that represents the `ComputeTensorData`.
    /// </summary>
    /// <returns>The string summary of the `ComputeTensorData`.</returns>
    public override string ToString()
    {
        return string.Format("GPU<ComputeTensorData>:{0} buffer: {1}", m_Shape, m_Buffer);
    }

    /// <summary>
    /// Moves the tensor into GPU memory on the GPUCompute backend device.
    /// </summary>
    /// <param name="X">The tensor to move to the compute backend.</param>
    /// <param name="clearOnInit">Whether to zero the data on pinning. The default value is `true`.</param>
    /// <returns>The pinned `ComputeTensorData`.</returns>
    public static ComputeTensorData Pin(Tensor X, bool clearOnInit = true)
    {
        var onDevice = X.tensorOnDevice;
        if (onDevice == null)
        {
            X.AttachToDevice(new ComputeTensorData(X.shape, clearOnInit));
            return X.tensorOnDevice as ComputeTensorData;
        }

        if (onDevice is ComputeTensorData)
            return onDevice as ComputeTensorData;

        if (onDevice is IConvertibleToComputeTensorData asConvertible)
            X.AttachToDevice(asConvertible.ConvertToComputeTensorData(X.shape));
        else
            X.UploadToDevice(new ComputeTensorData(X.shape, clearOnInit: false)); // device is not compatible, create new array and upload

        return X.tensorOnDevice as ComputeTensorData;
    }
}
} // namespace Unity.Sentis
