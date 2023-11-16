using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine.Networking;

namespace Unity.Sentis
{
    class AsyncWebReadbackRequest
    {
        public UnityWebRequestAsyncOperation asyncOperation;
        public int inferenceHeaderLength;
        public bool downloadDone = false;
        public int refCount = 0;
    }

    /// <summary>
    /// Represents `Tensor` data as returned from a `Cloud` layer. Waits on network completion
    /// </summary>
    class CloudTensorData : ITensorData
    {
        private int capacity;
        private int offset;

        private AsyncWebReadbackRequest m_AsyncDownloadRequest;

        public CloudTensorData(int capacity, int offset, AsyncWebReadbackRequest request)
        {
            this.capacity = capacity;
            this.offset = offset;
            m_AsyncDownloadRequest = request;
            m_AsyncDownloadRequest.refCount++;
        }

        /// <summary>
        /// Reserves memory for `count` elements.
        /// </summary>
        public void Reserve(int count)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Uploads the tensor data to internal storage.
        /// </summary>
        public void Upload<T>(NativeArray<T> data, int srcCount, int srcOffset = 0) where T : unmanaged
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Checks if asynchronous readback request it done.
        ///
        /// Returns true if async readback is successful.
        /// </summary>
        public bool IsAsyncReadbackRequestDone()
        {
            return m_AsyncDownloadRequest.downloadDone;
        }

        public void AsyncReadbackRequest(Action<bool> callback = null)
        {
            Action<UnityWebRequest> task = request =>
            {
                callback?.Invoke(!request.isDone);
            };
            task(m_AsyncDownloadRequest.asyncOperation.webRequest);
        }

        /// <summary>
        /// Blocking call to make sure that internal data is correctly written to and available for CPU read back.
        /// </summary>
        public void CompleteAllPendingOperations()
        {
            if (m_AsyncDownloadRequest.downloadDone)
            {
                return;
            }
            while (!m_AsyncDownloadRequest.asyncOperation.isDone)
            {

            }

            var request = m_AsyncDownloadRequest.asyncOperation.webRequest;
            Logger.AssertIsTrue(request.GetResponseHeaders().ContainsKey(Cloud.kContentLengthHeader), "CloudTensorData.UnsuportedArgument");
            m_AsyncDownloadRequest.inferenceHeaderLength = Int32.Parse(request.GetResponseHeaders()[Cloud.kContentLengthHeader]);

            m_AsyncDownloadRequest.downloadDone = true;
        }

        public NativeArray<T> Download<T>(int dstCount, int srcOffset = 0) where T: unmanaged
        {
            if (!m_AsyncDownloadRequest.downloadDone)
            {
                CompleteAllPendingOperations();
            }

            var ret = new NativeArray<T>(dstCount, Allocator.Temp);

            unsafe
            {
                float* dstPtr = (float*) ret.GetUnsafePtr();

                byte* srcPtr = (byte*)m_AsyncDownloadRequest.asyncOperation.webRequest.downloadHandler.nativeData.GetUnsafeReadOnlyPtr<byte>();

                UnsafeUtility.MemCpy(dstPtr, srcPtr + offset + m_AsyncDownloadRequest.inferenceHeaderLength, sizeof(float) * dstCount);
            }

            return ret;
        }

        /// <summary>
        /// Returns a deep copy of the internal storage.
        /// </summary>
        public ITensorData Clone()
        {
            return new CloudTensorData(capacity, offset, m_AsyncDownloadRequest);
        }

        public void Dispose()
        {
            if (m_AsyncDownloadRequest.refCount > 0)
                m_AsyncDownloadRequest.refCount--;

            if (m_AsyncDownloadRequest.refCount != 0)
                return;

            m_AsyncDownloadRequest.asyncOperation.webRequest.Dispose();
            m_AsyncDownloadRequest.asyncOperation = null;
        }

        /// <inheritdoc/>
        public int maxCapacity => capacity;

        /// <inheritdoc/>
        public DeviceType deviceType => DeviceType.CPU; // TODO change to WEB when we add this again
    }
}
