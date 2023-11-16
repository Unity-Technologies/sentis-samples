using System;
using Unity.Collections;
using UnityEngine;

namespace Unity.Sentis {
/// <summary>
/// An interface that represents a device-dependent representation of the data in a tensor.
/// </summary>
public interface ITensorData : IDisposable
{
    /// <summary>
    /// Reserves memory for `count` elements.
    /// </summary>
    /// <param name="count">The number of elements to reserve in memory.</param>
    void Reserve(int count);

    /// <summary>
    /// Uploads a contiguous block of tensor data to internal storage.
    /// </summary>
    /// <param name="data">The data to upload.</param>
    /// <param name="srcCount">The number of elements to upload.</param>
    /// <param name="srcOffset">The index of the first data element to upload.</param>
    /// <typeparam name="T">The type of data to upload.</typeparam>
    void Upload<T>(NativeArray<T> data, int srcCount, int srcOffset = 0) where T : unmanaged;

    /// <summary>
    /// Checks if asynchronous readback request is done.
    /// </summary>
    /// <returns>Whether async readback is successful.</returns>
    bool IsAsyncReadbackRequestDone();

    /// <summary>
    /// Schedules asynchronous readback of the internal data.
    /// </summary>
    /// <param name="callback">Callback invoked when async readback is finished. Return value indicates if async readback is successful.</param>
    void AsyncReadbackRequest(Action<bool> callback = null);

    /// <summary>
    /// Blocking call to make sure that internal data is correctly written to and available for CPU read back.
    /// </summary>
    void CompleteAllPendingOperations();

    /// <summary>
    /// Returns a contiguous block of data from internal storage.
    /// </summary>
    /// <param name="dstCount">The number of elements to download.</param>
    /// <param name="srcOffset">The index of the first element in storage to download.</param>
    /// <typeparam name="T">The data type of the elements.</typeparam>
    /// <returns>A native array of downloaded elements.</returns>
    NativeArray<T> Download<T>(int dstCount, int srcOffset = 0) where T : unmanaged;

    /// <summary>
    /// Returns a deep copy of the internal storage.
    /// </summary>
    /// <returns>Cloned internal storage.</returns>
    ITensorData Clone();

    /// <summary>
    /// The maximum count of the stored data elements.
    /// </summary>
    int maxCapacity { get; }

    /// <summary>
    /// On what device backend are the data elements stored.
    /// </summary>
    DeviceType deviceType { get; }
}

/// <summary>
/// An interface that represents tensor data that can be read to and written from on CPU.
/// </summary>
public interface IReadableTensorData
{
    /// <summary>
    /// Returns a data element.
    /// </summary>
    /// <param name="index">The index of the element.</param>
    /// <typeparam name="T">The data type of the element.</typeparam>
    /// <returns>Data element.</returns>
    T Get<T>(int index) where T : unmanaged;

    /// <summary>
    /// Sets `value` data element at `index`.
    /// </summary>
    /// <param name="index">The index of the element to set.</param>
    /// <param name="value">The value to set for the element.</param>
    /// <typeparam name="T">The data type of the element.</typeparam>
    void Set<T>(int index, T value) where T : unmanaged;

    /// <summary>
    /// Returns a ReadOnlySpan on the linear memory data.
    /// </summary>
    /// <param name="dstCount">The number of elements to span.</param>
    /// <param name="srcOffset">The index of the first element in the data.</param>
    /// <typeparam name="T">The data type of the elements.</typeparam>
    /// <returns>Span of elements.</returns>
    ReadOnlySpan<T> ToReadOnlySpan<T>(int dstCount, int srcOffset = 0) where T : unmanaged;

    /// <summary>
    /// Returns a ReadOnlyNativeArray handle on the linear memory data.
    /// </summary>
    /// <param name="dstCount">The number of elements in the array.</param>
    /// <param name="srcOffset">The index of the first element in the data.</param>
    /// <typeparam name="T">The data type of the elements.</typeparam>
    /// <returns>NativeArray of elements.</returns>
    NativeArray<T>.ReadOnly GetReadOnlyNativeArrayHandle<T>(int dstCount, int srcOffset = 0) where T : unmanaged;

    /// <summary>
    /// Returns an array that is a copy of the linear memory data.
    /// </summary>
    /// <param name="dstCount">The number of elements in the array.</param>
    /// <param name="srcOffset">The index of the first element in the data.</param>
    /// <typeparam name="T">The data type of the elements.</typeparam>
    /// <returns>Array of elements.</returns>
    T[] ToArray<T>(int dstCount, int srcOffset = 0) where T : unmanaged;
}

} // namespace Unity.Sentis
