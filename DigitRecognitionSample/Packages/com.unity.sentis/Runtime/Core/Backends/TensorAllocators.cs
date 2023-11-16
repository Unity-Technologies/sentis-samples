using System;
using System.Collections.Generic;
using UnityEngine.Assertions;
using UnityEngine.Profiling;

namespace Unity.Sentis {

static class AllocatorUtils
{
    internal static Tensor NewTensor(DataType dataType, TensorShape shape, ITensorData buffer, ITensorAllocator allocator)
    {
        switch (dataType)
        {
            case DataType.Float:
                return new TensorFloat(shape, buffer, allocator);
            case DataType.Int:
                return new TensorInt(shape, buffer, allocator);
            default:
                throw new NotImplementedException($"DataType {dataType} not supported");
        }
    }
}

// @TODO: reduce code duplication between TensorCachingByShapeAllocator and TensorCachingAllocator
class TensorCachingByShapeAllocator : ITensorAllocator
{
    struct Entry
    {
        public TensorShape shape;
        public ITensorData buffer;
        public CacheKey ToKey() { return new CacheKey { shape = shape, deviceType = buffer.deviceType }; }
    }

    struct CacheKey
    {
        public TensorShape shape;
        public DeviceType deviceType;
    }

    // multi-value Dictionary<CacheKey, Entry*> implemented via
    // pair of m_FreeTensorByShape and m_FreeTensors
    Dictionary<CacheKey, LinkedListNode<Entry>> m_FreeBufferByShape = new Dictionary<CacheKey, LinkedListNode<Entry>>();
    LinkedList<Entry> m_FreeBuffers = new LinkedList<Entry>();
    Dictionary<Tensor, ITensorData> m_BusyTensors = new Dictionary<Tensor, ITensorData>();
    Dictionary<ITensorData, int> m_SharedBuffers = new Dictionary<ITensorData, int>();

    ~TensorCachingByShapeAllocator()
    {
        Dispose();
    }

    protected void AddRef(ITensorData buffer)
    {
        if (buffer == null)
            return;

        var sharedBufferCount = 0;
        m_SharedBuffers.TryGetValue(buffer, out sharedBufferCount);
        m_SharedBuffers[buffer] = sharedBufferCount + 1;
    }

    protected void DecRef(ITensorData buffer, Action<ITensorData> onLastRef = null)
    {
        if (buffer == null)
            return;

        Assert.IsTrue(m_SharedBuffers.ContainsKey(buffer));
        Assert.IsTrue(m_SharedBuffers[buffer] > 0);
        if (--m_SharedBuffers[buffer] > 0)
            return;

        m_SharedBuffers.Remove(buffer);

        if (onLastRef != null)
            onLastRef(buffer);
    }

    protected void AdoptFreeBuffer(TensorShape shape, ITensorData buffer)
    {
        // code below automatically covers handles edge-case (2)
        // by adopting tensor's with the new ITensorData into m_FreeTensors/m_FreeTensorByShape
        var newEntry = new Entry { shape = shape, buffer = buffer };
        var key = newEntry.ToKey();
        LinkedListNode<Entry> node;
        if (m_FreeBufferByShape.TryGetValue(key, out node))
        {
            m_FreeBuffers.AddAfter(node, newEntry);
        }
        else
        {
            var newNode = m_FreeBuffers.AddLast(newEntry);
            m_FreeBufferByShape.Add(key, newNode);
        }
    }

    /// <inheritdoc/>
    public virtual Tensor Alloc(TensorShape shape, DataType dataType, DeviceType deviceType, AllocScope scope)
    {
        Profiler.BeginSample("Sentis.ShapeAllocator.Alloc");

        var key = new CacheKey { shape = shape, deviceType = deviceType };
        LinkedListNode<Entry> node;
        if (m_FreeBufferByShape.TryGetValue(key, out node))
        {
            Assert.AreEqual(node.Value.shape, shape);

            // advance dictionary to the next Tensor with the same shape, if available
            if (node.Next != null && node.Next.Value.shape == shape && node.Next.Value.buffer.deviceType == deviceType)
                m_FreeBufferByShape[key] = node.Next;
            else
                m_FreeBufferByShape.Remove(key);

            var buffer = node.Value.buffer;

            buffer?.Reserve(shape.length);
            Tensor tensor = AllocatorUtils.NewTensor(dataType, shape, buffer, this);

            m_FreeBuffers.Remove(node);
            m_BusyTensors.Add(tensor, buffer);
            AddRef(buffer);

            Assert.AreEqual(tensor.shape, shape);
            Profiler.EndSample();
            return tensor;
        }

        Tensor newTensor = AllocatorUtils.NewTensor(dataType, shape, null, this);

        m_BusyTensors.Add(newTensor, newTensor.tensorOnDevice);
        AddRef(newTensor.tensorOnDevice);

        Profiler.EndSample();
        return newTensor;
    }

    /// <inheritdoc/>
    public virtual Tensor Alloc(TensorShape shape, DataType dataType, ITensorData buffer, AllocScope scope)
    {
        // TODO be careful with this and Pixelshader op since buffer with the same length might not work
        Profiler.BeginSample("Sentis.ShapeAllocator.Alloc");

        Tensor tensor = AllocatorUtils.NewTensor(dataType, shape, buffer, this); // @TODO: reuse Tensor instances

        m_BusyTensors.Add(tensor, buffer);
        AddRef(buffer);

        Profiler.EndSample();
        return tensor;
    }

    /// <inheritdoc/>
    public virtual void PostLayerCleanup() { }

    /// <inheritdoc/>
    public virtual void Release(Tensor tensor, bool calledFromTensorDispose)
    {
        Profiler.BeginSample("Sentis.ShapeAllocator.Release");
        Assert.AreEqual(tensor.allocator, this);

        var detachedBuffer = tensor.Invalidate(); // calls MoveToDevice(newBuffer=null)

        if (!m_BusyTensors.ContainsKey(tensor))
        {
            if (detachedBuffer == null)
                return;

            foreach (var freeEntry in m_FreeBuffers)
                if (freeEntry.buffer == detachedBuffer)
                    return;

            // some operations can create new Tensor and reassign ITensorData to it
            foreach (var busyEntry in m_BusyTensors)
                if (busyEntry.Value == detachedBuffer)
                    return; // we have at least another instance ITensorData in m_BusyTensors, nothing to realease
        }

        Assert.IsTrue(m_BusyTensors.ContainsKey(tensor));
        m_BusyTensors.Remove(tensor);
        Profiler.EndSample();
    }

    /// <inheritdoc/>
    public virtual void MoveToDevice(Tensor X, ITensorData newBuffer, ITensorData oldBuffer, bool disposeDetachedBufferHint)
    {
        if (newBuffer == oldBuffer)
            return;

        Assert.AreEqual(X.allocator, this);
        Assert.IsTrue(m_BusyTensors.ContainsKey(X));
        m_BusyTensors[X] = newBuffer;

        AddRef(newBuffer);
        DecRef(oldBuffer,
            (freeBuffer) => {
                if (disposeDetachedBufferHint)
                    freeBuffer.Dispose();
                else
                    AdoptFreeBuffer(X.shape, freeBuffer);
            });
    }

    readonly List<Tensor> m_TensorToReleaseTempList = new List<Tensor>();

    /// <inheritdoc/>
    public virtual void Reset(bool keepCachedMemory)
    {
        Profiler.BeginSample("Sentis.ShapeAllocator.Reset");

        if (!keepCachedMemory)
            Dispose();

        //avoid GC when m_BusyTensors.Keys would be converted to list.
        m_TensorToReleaseTempList.Clear();
        m_TensorToReleaseTempList.AddRange(m_BusyTensors.Keys);

        foreach (var tensor in m_TensorToReleaseTempList)
            Release(tensor, false);

        Assert.AreEqual(m_BusyTensors.Count, 0);
        Assert.AreEqual(m_SharedBuffers.Count, 0);

        Profiler.EndSample();
    }

    /// <inheritdoc/>
    public virtual void WaiveOwnership(Tensor X)
    {
        Assert.AreEqual(X.allocator, this);
        Assert.IsTrue(m_BusyTensors.ContainsKey(X));
        m_BusyTensors.Remove(X);

        var buffer = X.tensorOnDevice;
        if (buffer == null)
            return;

        Profiler.BeginSample("Sentis.ShapeAllocator.WaiveOwnership");

        int sharedCount = 0;
        m_SharedBuffers.TryGetValue(buffer, out sharedCount);
        if (sharedCount > 1)
        {
            var patchBusyTensors = new List<Tensor>();
            foreach (var busyEntry in m_BusyTensors)
                if (busyEntry.Value == buffer)
                    patchBusyTensors.Add(busyEntry.Key);

            Assert.AreEqual(sharedCount - 1, patchBusyTensors.Count);

            foreach (var busyTensor in patchBusyTensors)
            {
                Assert.AreEqual(m_BusyTensors[busyTensor], buffer);

                var oldBuffer = busyTensor.DetachFromDevice(false);
                var newBuffer = busyTensor.tensorOnDevice;
                Assert.IsTrue(oldBuffer == buffer);
                Assert.IsTrue(newBuffer != buffer);
                m_BusyTensors[busyTensor] = newBuffer;
                AddRef(newBuffer);
            }
        }

        // Assert no references to tensor are left owned by allocator
        Assert.IsTrue(m_SharedBuffers[buffer] == 1);
        m_SharedBuffers.Remove(buffer);
        foreach (var freeEntry in m_FreeBuffers)
        {
            Assert.IsTrue(freeEntry.buffer != buffer);
        }

        foreach (var busyEntry in m_BusyTensors)
        {
            Assert.IsTrue(busyEntry.Key != X);
            Assert.IsTrue(busyEntry.Value != buffer);
        }

        Profiler.EndSample();
    }

    /// <summary>
    /// Disposes of the allocator and any associated memory.
    /// </summary>
    public virtual void Dispose()
    {
        m_FreeBufferByShape.Clear();

        //avoid GC when m_BusyTensors.Keys would be converted to list.
        m_TensorToReleaseTempList.Clear();
        m_TensorToReleaseTempList.AddRange(m_BusyTensors.Keys);
        foreach (var tensor in m_TensorToReleaseTempList)
            Release(tensor, false);

        foreach (var entry in m_FreeBuffers)
            entry.buffer?.Dispose();

        m_BusyTensors.Clear();
        m_FreeBuffers.Clear();
        m_SharedBuffers.Clear();
    }
}

class TempTensorCachingAllocator : TensorCachingAllocator
{
    /// <inheritdoc/>
    public override void PostLayerCleanup()
    {
        base.PostLayerCleanup();

        m_TensorToReleaseTempList.Clear();
        m_TensorToReleaseTempList.AddRange(m_BusyTensors.Keys);

        //Release tensorData to allocator + keep tensor instance
        foreach(var tensor in m_TensorToReleaseTempList)
            Release(tensor, true);
    }
}

/// <summary>
/// Represents a caching `Tensor` allocator.
/// </summary>
public class TensorCachingAllocator : ITensorAllocator
{
    struct Entry
    {
        public int size;
        public ITensorData tensorData;
        public DeviceType deviceType;
        public bool free;

        public int maxCapacity => tensorData.maxCapacity;
    }

    // Sorted by size array of ITensorData
    List<Entry> m_AllocatedBuffers = new List<Entry>();
    protected Dictionary<Tensor, ITensorData> m_BusyTensors = new Dictionary<Tensor, ITensorData>();
    Dictionary<ITensorData, int> m_SharedBuffers = new Dictionary<ITensorData, int>();

    Action<ITensorData> m_DisposeAllocatedBufferDelegate;
    Action<ITensorData> m_AdoptFreeBufferDelegate;

    // Stores only hollow tensor objects, tensor data is stored by m_AllocatedBuffers
    Stack<TensorFloat> m_AllocatedTensorsFloat = new Stack<TensorFloat>();
    Stack<TensorInt> m_AllocatedTensorsInt = new Stack<TensorInt>();
    int m_NumAllocatedBufferSinceCleanup = 0;

    /// <summary>
    /// Initializes and returns an instance of `TensorCachingAllocator`.
    /// </summary>
    public TensorCachingAllocator()
    {
        m_DisposeAllocatedBufferDelegate = DisposeAllocatedBuffer;
        m_AdoptFreeBufferDelegate = AdoptFreeBuffer;
    }

    /// <summary>
    /// Finalizes the `TensorCachingAllocator`.
    /// </summary>
    ~TensorCachingAllocator()
    {
        Dispose();
    }

    Tensor AllocTensorInternal(TensorShape shape, ITensorData buffer, DataType dataType)
    {
        Tensor res = null;

        switch (dataType)
        {
            case DataType.Float:
            {
                lock (m_AllocatedTensorsFloat)
                {
                    if (m_AllocatedTensorsFloat.Count > 0)
                    {
                        res = m_AllocatedTensorsFloat.Pop();
                        res.Init(shape, buffer, this);
                    }
                    else
                        res = new TensorFloat(shape, buffer, this);
                }

                break;
            }
            case DataType.Int:
            {
                lock (m_AllocatedTensorsInt)
                {
                    if (m_AllocatedTensorsInt.Count > 0)
                    {
                        res = m_AllocatedTensorsInt.Pop();
                        res.Init(shape, buffer, this);
                    }
                    else
                        res = new TensorInt(shape, buffer, this);
                }

                break;
            }
            default:
                throw new NotImplementedException($"DataType {dataType} not supported");
        }

        return res;
    }

    int FindEntryIndexForSize(int minimumSize)
    {
        var allocatedBufferCount = m_AllocatedBuffers.Count;

        if (allocatedBufferCount == 0 || m_AllocatedBuffers[^1].size < minimumSize)
            return -1;

        if (m_AllocatedBuffers[0].size >= minimumSize)
            return 0;

        // l will be the last entry which size < minimumSize
        // r will be the first entry which size >= minimumSize
        int l = 0, r = allocatedBufferCount - 1;

        while (r - l > 1)
        {
            var m = (l + r) / 2;
            if (m_AllocatedBuffers[m].size >= minimumSize)
                r = m;
            else
                l = m;
        }

        return r;
    }

    void AddRef(ITensorData buffer)
    {
        if (buffer == null)
            return;

        var sharedBufferCount = 0;
        m_SharedBuffers.TryGetValue(buffer, out sharedBufferCount);
        m_SharedBuffers[buffer] = sharedBufferCount + 1;
    }

    void DecRef(ITensorData buffer, Action<ITensorData> onLastRef = null)
    {
        if (buffer == null)
            return;

        Assert.IsTrue(m_SharedBuffers.ContainsKey(buffer));
        Assert.IsTrue(m_SharedBuffers[buffer] > 0);
        if (--m_SharedBuffers[buffer] > 0)
            return;

        m_SharedBuffers.Remove(buffer);

        if (onLastRef != null)
            onLastRef(buffer);
    }

    void AdoptFreeBuffer(ITensorData buffer)
    {
        var bufferSize = buffer.maxCapacity;
        var newEntry = new Entry { size = bufferSize, tensorData = buffer, deviceType = buffer.deviceType, free = true };

        var index = FindEntryIndexForSize(bufferSize);

        if (index == -1)
        {
            m_AllocatedBuffers.Add(newEntry);
            return;
        }

        var allocatedBufferCount = m_AllocatedBuffers.Count;

        for (; index < allocatedBufferCount; index++)
        {
            var entry = m_AllocatedBuffers[index];

            if (entry.size != bufferSize)
                break;

            if (ReferenceEquals(buffer, entry.tensorData))
            {
                Assert.IsFalse(entry.free);
                entry.free = true;
                m_AllocatedBuffers[index] = entry;
                return;
            }
        }

        // m is now out of the list, or points to the first entry with entry.size > buffer.size
        if (index == allocatedBufferCount)
            m_AllocatedBuffers.Add(newEntry);
        else
            m_AllocatedBuffers.Insert(index, newEntry);
    }

    void DisposeAllocatedBuffer(ITensorData buffer)
    {
        for (int i = m_AllocatedBuffers.Count - 1; i >= 0; i--)
            if (m_AllocatedBuffers[i].tensorData == buffer)
                m_AllocatedBuffers.RemoveAt(i);
        buffer.Dispose();
    }

    /// <inheritdoc/>
    public virtual Tensor Alloc(TensorShape shape, DataType dataType, DeviceType deviceType, AllocScope scope)
    {
        Tensor AllocNew(TensorShape s, DataType dt)
        {
            ++m_NumAllocatedBufferSinceCleanup;

            var newTensor = AllocTensorInternal(s, null, dt);
            m_BusyTensors.Add(newTensor, newTensor.tensorOnDevice);
            AddRef(newTensor.tensorOnDevice);
            return newTensor;
        }

        var index = FindEntryIndexForSize(shape.length);
        if(index == -1)
            return AllocNew(shape, dataType);

        for(var allocatedBufferCount = m_AllocatedBuffers.Count; index < allocatedBufferCount; index++)
        {
            var entry = m_AllocatedBuffers[index];
            if (entry.deviceType != deviceType || !entry.free)
                continue;

            entry.free = false;
            m_AllocatedBuffers[index] = entry;

            var buffer = entry.tensorData;
            buffer?.Reserve(shape.length);

            var tensor = AllocTensorInternal(shape, buffer, dataType);

            m_BusyTensors.Add(tensor, tensor.tensorOnDevice);
            AddRef(tensor.tensorOnDevice);
            return tensor;
        }

        return AllocNew(shape, dataType);
    }

    /// <inheritdoc/>
    public virtual Tensor Alloc(TensorShape shape, DataType dataType, ITensorData buffer, AllocScope scope)
    {
        var tensor = AllocTensorInternal(shape, buffer, dataType);
        m_BusyTensors.Add(tensor, tensor.tensorOnDevice);
        AddRef(tensor.tensorOnDevice);
        return tensor;
    }

    /// <inheritdoc/>
    public virtual void PostLayerCleanup()
    {
        //This allocator does not have support for allocation scope,
        //all tensors live until Reset() is called.

        //however allocation of new buffer are tracked for debug warning purpose
        //reset here to help catch context of those allocation (potential leaks)
        m_NumAllocatedBufferSinceCleanup = 0;
    }

    /// <inheritdoc/>
    public virtual void Release(Tensor tensor, bool calledFromTensorDispose)
    {
        Profiler.BeginSample("Sentis.SizeAllocator.Release");
        Assert.AreEqual(tensor.allocator, this);

        var detachedBuffer = tensor.Invalidate(); // calls MoveToDevice(newBuffer=null,disposeDetachedBufferHint=false)

        if (calledFromTensorDispose)
        {
            switch (tensor.dataType)
            {
                case DataType.Float:
                {
                    lock (m_AllocatedTensorsFloat)
                    {
                        m_AllocatedTensorsFloat.Push(tensor as TensorFloat);
                    }

                    break;
                }
                case DataType.Int:
                {
                    lock (m_AllocatedTensorsInt)
                    {
                        m_AllocatedTensorsInt.Push(tensor as TensorInt);
                    }

                    break;
                }
                default:
                    throw new NotImplementedException($"DataType {tensor.dataType} not supported");
            }
        }

        if (!m_BusyTensors.ContainsKey(tensor))
        {
            if (detachedBuffer == null)
                return;

            foreach (var entry in m_AllocatedBuffers)
                if (entry.tensorData == detachedBuffer && entry.free)
                    return;

            // some operations can create new Tensor and reassign ITensorData to it
            foreach (var busyEntry in m_BusyTensors)
                if (busyEntry.Value == detachedBuffer)
                    return; // we have original ITensorData in m_BusyTensors, nothing to release
        }

        Assert.IsTrue(m_BusyTensors.ContainsKey(tensor));
        m_BusyTensors.Remove(tensor);

        Profiler.EndSample();
    }

    /// <inheritdoc/>
    public virtual void MoveToDevice(Tensor X, ITensorData newBuffer, ITensorData oldBuffer, bool disposeDetachedBufferHint)
    {
        if (newBuffer == oldBuffer)
            return;

        Assert.AreEqual(X.allocator, this);
        Assert.IsTrue(m_BusyTensors.ContainsKey(X));
        m_BusyTensors[X] = newBuffer;

        AddRef(newBuffer);

        if (disposeDetachedBufferHint)
            DecRef(oldBuffer, m_DisposeAllocatedBufferDelegate);
        else
            DecRef(oldBuffer, m_AdoptFreeBufferDelegate);
    }

    protected readonly List<Tensor> m_TensorToReleaseTempList = new List<Tensor>();

    /// <inheritdoc/>
    public virtual void Reset(bool keepCachedMemory)
    {
        Profiler.BeginSample("Sentis.SizeAllocator.Reset");

        if (!keepCachedMemory)
            Dispose();

        //avoid GC when m_BusyTensors.Keys would be converted to list.
        m_TensorToReleaseTempList.Clear();
        m_TensorToReleaseTempList.AddRange(m_BusyTensors.Keys);

        foreach(var tensor in m_TensorToReleaseTempList)
            Release(tensor, false);

        Assert.AreEqual(m_BusyTensors.Count, 0);
        Assert.AreEqual(m_SharedBuffers.Count, 0);

        foreach(var buf in m_AllocatedBuffers)
            Assert.IsTrue(buf.free);

        Profiler.EndSample();
    }

    /// <inheritdoc/>
    public virtual void WaiveOwnership(Tensor X)
    {
        Assert.AreEqual(X.allocator, this);
        Assert.IsTrue(m_BusyTensors.ContainsKey(X));
        m_BusyTensors.Remove(X);

        var buffer = X.tensorOnDevice;
        if (buffer == null)
            return;

        Profiler.BeginSample("Sentis.SizeAllocator.WaiveOwnership");

        int sharedCount = 0;
        m_SharedBuffers.TryGetValue(buffer, out sharedCount);
        if (sharedCount > 1)
        {
            var patchBusyTensors = new List<Tensor>();
            foreach (var busyEntry in m_BusyTensors)
                if (busyEntry.Value == buffer)
                    patchBusyTensors.Add(busyEntry.Key);

            Assert.AreEqual(sharedCount - 1, patchBusyTensors.Count);

            foreach (var busyTensor in patchBusyTensors)
            {
                Assert.AreEqual(m_BusyTensors[busyTensor], buffer);

                var oldBuffer = busyTensor.DetachFromDevice(false);
                var newBuffer = busyTensor.tensorOnDevice;
                Assert.IsTrue(oldBuffer == buffer);
                Assert.IsTrue(newBuffer != buffer);
                m_BusyTensors[busyTensor] = newBuffer;
                AddRef(newBuffer);
            }
        }

        // Assert no references to tensor are left owned by allocator
        Assert.IsTrue(m_SharedBuffers[buffer] == 1);
        m_SharedBuffers.Remove(buffer);

        int countInAllocatedBuffers = 0;
        for (int i = 0; i < m_AllocatedBuffers.Count; i++)
        {
            Entry entry = m_AllocatedBuffers[i];
            if (entry.tensorData == buffer)
            {
                Assert.IsFalse(entry.free);
                m_AllocatedBuffers.RemoveAt(i);
                countInAllocatedBuffers++;
            }
        }

        // This entry should have only been in the allocated buffers once at most
        Assert.IsTrue(countInAllocatedBuffers <= 1);

        foreach(var busyEntry in m_BusyTensors)
        {
            Assert.IsTrue(busyEntry.Key != X);
            Assert.IsTrue(busyEntry.Value != buffer);
        }

        Profiler.EndSample();
    }

    /// <summary>
    /// Disposes of the `TensorCachingAllocator` and any associated memory.
    /// </summary>
    public virtual void Dispose()
    {
        //avoid GC when m_BusyTensors.Keys would be converted to list.
        m_TensorToReleaseTempList.Clear();
        m_TensorToReleaseTempList.AddRange(m_BusyTensors.Keys);
        foreach(var tensor in m_TensorToReleaseTempList)
            Release(tensor, false);

        foreach (var entry in m_AllocatedBuffers)
            entry.tensorData?.Dispose();

        m_BusyTensors.Clear();
        m_AllocatedBuffers.Clear();
        m_AllocatedTensorsFloat.Clear();
        m_AllocatedTensorsInt.Clear();
        m_SharedBuffers.Clear();
    }
}

} // namespace Unity.Sentis
