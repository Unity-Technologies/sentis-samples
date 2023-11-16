using System;
using System.Runtime.ConstrainedExecution;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace Unity.Sentis
{

///see https://referencesource.microsoft.com/#mscorlib/system/runtime/interopservices/safehandle.cs
class NativeMemorySafeHandle : SafeHandle
{
    readonly Allocator m_AllocatorLabel;

    [ReliabilityContract(Consistency.WillNotCorruptState, Cer.MayFail)]
    public unsafe NativeMemorySafeHandle(long size, int alignment, bool clearOnInit, Allocator allocator) : base(IntPtr.Zero, true)
    {
        m_AllocatorLabel = allocator;
        if (size <= 0)
            return;

        SetHandle((IntPtr)UnsafeUtility.Malloc(size, alignment, allocator));
        if (clearOnInit)
            UnsafeUtility.MemClear((void*)handle, size);
    }

    public override bool IsInvalid {
        get { return handle == IntPtr.Zero; }
    }

    [ReliabilityContract(Consistency.WillNotCorruptState, Cer.Success)]
    protected override unsafe bool ReleaseHandle()
    {
        UnsafeUtility.Free((void*)handle, m_AllocatorLabel);
        return true;
    }
}

class PinnedMemorySafeHandle : SafeHandle
{
    private readonly GCHandle m_GCHandle;

    [ReliabilityContract(Consistency.WillNotCorruptState, Cer.MayFail)]
    public PinnedMemorySafeHandle(object managedObject) : base(IntPtr.Zero, true)
    {
        m_GCHandle = GCHandle.Alloc(managedObject, GCHandleType.Pinned);
        IntPtr pinnedPtr = m_GCHandle.AddrOfPinnedObject();
        SetHandle(pinnedPtr);
    }

    public override bool IsInvalid {
        get { return handle == IntPtr.Zero; }
    }

    [ReliabilityContract(Consistency.WillNotCorruptState, Cer.Success)]
    protected override bool ReleaseHandle()
    {
        m_GCHandle.Free();
        return true;
    }
}

/// <summary>
/// Options for the data type of a `Tensor`.
/// </summary>
public enum DataType
{
    /// <summary>
    /// Use 32-bit floating point data.
    /// </summary>
    Float,
    /// <summary>
    /// Use 32-bit signed integer data.
    /// </summary>
    Int
}

/// <summary>
/// Represents an area of managed memory that's exposed as if it's native memory.
/// </summary>
public class NativeTensorArrayFromManagedArray : NativeTensorArray
{
    readonly int m_PinnedMemoryByteOffset;

    /// <summary>
    /// Initializes and returns an instance of `NativeTensorArrayFromManagedArray` from an `Array` and an integer offset.
    /// </summary>
    /// <param name="srcData">The data for the `Tensor` as an `Array`.</param>
    /// <param name="srcOffset">The integer offset to use for the backing data.</param>
    public NativeTensorArrayFromManagedArray(Array srcData, int srcOffset = 0)
        : this(srcData, srcOffset, sizeof(float), srcData.Length - srcOffset) { }

    /// <summary>
    /// Initializes and returns an instance of `NativeTensorArrayFromManagedArray` from an `Array` and a integer offset and count.
    /// </summary>
    /// <param name="srcData">The data for the `Tensor` as an `Array`.</param>
    /// <param name="srcOffset">The integer offset to use for the backing data.</param>
    /// <param name="numDestElement">The integer count to use for the backing data.</param>
    public NativeTensorArrayFromManagedArray(byte[] srcData, int srcOffset, int numDestElement)
        : this(srcData, srcOffset, sizeof(byte), numDestElement) { }

    /// <summary>
    /// Initializes and returns an instance of `NativeTensorArrayFromManagedArray` from an `Array` and a integer offset, size and count.
    /// </summary>
    /// <param name="srcData">The data for the `Tensor` as an `Array`.</param>
    /// <param name="srcElementOffset">The integer offset to use for the backing data.</param>
    /// <param name="srcElementSize">The integer size to use for the backing data in bytes.</param>
    /// <param name="numDestElement">The integer count to use for the backing data.</param>
    unsafe NativeTensorArrayFromManagedArray(Array srcData, int srcElementOffset, int srcElementSize, int numDestElement)
        : base(new PinnedMemorySafeHandle(srcData), numDestElement)
    {
        m_PinnedMemoryByteOffset = srcElementSize * srcElementOffset;

        //Safety checks
        int requiredAlignment = DataItemSize();
        int srcLengthInByte = (srcData.Length - srcElementOffset) * srcElementSize;
        int dstLengthInByte = numDestElement * DataItemSize();
        IntPtr pinnedPtrWithOffset = (IntPtr)base.RawPtr + m_PinnedMemoryByteOffset;
        if (srcElementOffset > srcData.Length)
            throw new ArgumentOutOfRangeException(nameof(srcElementOffset), "SrcElementOffset must be <= srcData.Length");
        if (dstLengthInByte > srcLengthInByte)
            throw new ArgumentOutOfRangeException(nameof(numDestElement), "NumDestElement too big for srcData and srcElementOffset");

        if (pinnedPtrWithOffset.ToInt64() % requiredAlignment != 0)
            throw new InvalidOperationException($"The NativeTensorArrayFromManagedArray source ptr (including offset) need to be aligned on {requiredAlignment} bytes.");

        var neededSrcPaddedLengthInByte = numDestElement * DataItemSize();
        if (srcLengthInByte < neededSrcPaddedLengthInByte)
            throw new InvalidOperationException($"The NativeTensorArrayFromManagedArray source ptr (including offset) is to small to account for extra padding.");
    }

    /// <inheritdoc/>
    public override unsafe void* RawPtr => (byte*)base.RawPtr + m_PinnedMemoryByteOffset;
}

/// <summary>
/// Represents an area of native memory that's exposed to managed code.
/// </summary>
public class NativeTensorArray : IDisposable
{
    private protected readonly SafeHandle m_SafeHandle;
    readonly Allocator m_Allocator;
    readonly int m_Length;

    /// <summary>
    /// Gets the size in bytes of an individual element.
    /// </summary>
    /// <returns>The size in bytes of an element.</returns>
    protected int DataItemSize()
    {
        return sizeof(float);
    }

    void CheckElementAccess(long index)
    {
        //Disabled by default for performance reasons.
        #if ENABLE_SENTIS_DEBUG
        if (Disposed)
            throw new InvalidOperationException("The NativeTensorArray was disposed.");
        if (index <0 || index >= m_Length)
            throw new IndexOutOfRangeException($"Accessing NativeTensorArray of length {m_Length} at index {index}.");
        #endif
    }

    /// <summary>
    /// Initializes and returns an instance of `NativeTensorArray` with a preexisting handle.
    /// </summary>
    /// <param name="safeHandle">The safe handle to the data.</param>
    /// <param name="dataLength">The integer number of elements.</param>
    protected NativeTensorArray(SafeHandle safeHandle, int dataLength)
    {
        m_Length = dataLength;
        m_SafeHandle = safeHandle;
        m_Allocator = Allocator.Persistent;
    }

    /// <summary>
    /// Initializes and returns an instance of `NativeTensorArray` with a given length.
    /// </summary>
    /// <param name="length">The integer number of elements to allocate.</param>
    /// <param name="clearOnInit">Whether to zero the data after allocating.</param>
    /// <param name="allocator">The allocation type to use as an `Allocator`.</param>
    public NativeTensorArray(int length, bool clearOnInit = true, Allocator allocator = Allocator.Persistent)
    {
        if (!UnsafeUtility.IsValidAllocator(allocator))
            throw new InvalidOperationException("The NativeTensorArray should use a valid allocator.");
        if (length < 0)
            throw new ArgumentOutOfRangeException(nameof (length), "Length must be >= 0");

        m_Length = length;
        m_SafeHandle = new NativeMemorySafeHandle(m_Length * DataItemSize(), DataItemSize(), clearOnInit, allocator);
        m_Allocator = allocator;
    }

    /// <summary>
    /// Clears the allocated memory to zero.
    /// </summary>
    public unsafe void ZeroMemory()
    {
        var numByteToClear = m_Length * DataItemSize();
        UnsafeUtility.MemClear(RawPtr, numByteToClear);
    }

    /// <summary>
    /// Disposes of the array and any associated memory.
    /// </summary>
    public virtual void Dispose()
    {
        m_SafeHandle.Dispose();
    }

    /// <summary>
    /// The size in bytes of an element of the stored type.
    /// </summary>
    public int SizeOfType => DataItemSize();

    /// <summary>
    /// The number of allocated elements.
    /// </summary>
    public int Length => m_Length;
    /// <summary>
    /// The number of allocated elements as a 64-bit integer.
    /// </summary>
    public long LongLength => m_Length;

    /// <summary>
    /// The raw pointer of the backing data.
    /// </summary>
    public virtual unsafe void* RawPtr
    {
        get
        {
            if (Disposed)
                throw new InvalidOperationException("The NativeTensorArray was disposed.");
            return (void*)m_SafeHandle.DangerousGetHandle();
        }
    }

    /// <summary>
    /// Whether the backing data is disposed.
    /// </summary>
    public bool Disposed => m_SafeHandle.IsClosed;

    /// <summary>
    /// Returns the raw pointer of the backing data at a given index.
    /// </summary>
    /// <param name="index">The index of the element.</param>
    /// <typeparam name="T">The type of the element.</typeparam>
    /// <returns>The raw pointer to the element in the data.</returns>
    public unsafe T* AddressAt<T>(long index) where T : unmanaged
    {
        return ((T*)RawPtr) + index;
    }

    /// <summary>
    /// Returns the value of the backing data at a given index.
    /// </summary>
    /// <param name="index">The index of the element.</param>
    /// <typeparam name="T">The type of the element.</typeparam>
    /// <returns>The value of the element in the data.</returns>
    public unsafe T Get<T>(int index) where T : unmanaged
    {
        CheckElementAccess(index);
        return UnsafeUtility.ReadArrayElement<T>(RawPtr, index);
    }

    /// <summary>
    /// Sets the value of the backing data at a given index.
    /// </summary>
    /// <param name="index">The index of the element.</param>
    /// <param name="value">The value to set at the index.</param>
    /// <typeparam name="T">The type of the element.</typeparam>
    public unsafe void Set<T>(int index, T value) where T : unmanaged
    {
        CheckElementAccess(index);
        UnsafeUtility.WriteArrayElement<T>(RawPtr, index, value);
    }

    /// <summary>
    /// Returns the data converted to a `NativeArray`.
    /// </summary>
    /// <typeparam name="T">The type of the data.</typeparam>
    /// <returns>The converted native array from data.</returns>
    public NativeArray<T> GetNativeArrayHandle<T>() where T : unmanaged
    {
        unsafe
        {
            NativeArray<T> nativeArray = NativeArrayUnsafeUtility.ConvertExistingDataToNativeArray<T>(RawPtr, m_Length, m_Allocator);
            #if ENABLE_UNITY_COLLECTIONS_CHECKS
            NativeArrayUnsafeUtility.SetAtomicSafetyHandle(ref nativeArray, AtomicSafetyHandle.Create());
            #endif
            return nativeArray;
        }
    }

    /// <summary>
    /// Returns the data as a `NativeArray` constrained to read only operations.
    /// </summary>
    /// <param name="dstCount">The number of elements.</param>
    /// <param name="srcOffset">The index of the first element.</param>
    /// <typeparam name="T">The type of the data.</typeparam>
    /// <returns>The read only native array of the data.</returns>
    public NativeArray<T>.ReadOnly GetReadOnlyNativeArrayHandle<T>(int dstCount, int srcOffset = 0) where T : unmanaged
    {
        unsafe
        {
            NativeArray<T> nativeArray = NativeArrayUnsafeUtility.ConvertExistingDataToNativeArray<T>((byte*)RawPtr + srcOffset * sizeof(T), dstCount, m_Allocator);
            #if ENABLE_UNITY_COLLECTIONS_CHECKS
            NativeArrayUnsafeUtility.SetAtomicSafetyHandle(ref nativeArray, AtomicSafetyHandle.Create());
            #endif
            return nativeArray.AsReadOnly();
        }
    }

    /// <summary>
    /// Returns the data as a `ReadOnlySpan`.
    /// </summary>
    /// <param name="dstCount">The number of elements.</param>
    /// <param name="srcOffset">The index of the first element.</param>
    /// <typeparam name="T">The type of the data.</typeparam>
    /// <returns>The span of the data.</returns>
    public ReadOnlySpan<T> AsReadOnlySpan<T>(int dstCount, int srcOffset = 0) where T : unmanaged
    {
        unsafe
        {
            #if ENABLE_UNITY_COLLECTIONS_CHECKS
            AtomicSafetyHandle.CheckReadAndThrow(AtomicSafetyHandle.Create());
            #endif

            return new ReadOnlySpan<T>((byte*)RawPtr + srcOffset * sizeof(T), dstCount);
        }
    }

    /// <summary>
    /// Returns the data as an array.
    /// </summary>
    /// <param name="dstCount">The number of elements.</param>
    /// <param name="srcOffset">The index of the first element.</param>
    /// <typeparam name="T">The type of the data.</typeparam>
    /// <returns>The copied array of the data.</returns>
    public T[] ToArray<T>(int dstCount, int srcOffset = 0) where T : unmanaged
    {
        var array = new T[dstCount];
        Copy(this, srcOffset, array, 0, dstCount);
        return array;
    }

    /// <summary>
    /// Copies all of the data to a `NativeTensorArray` starting from a given offset.
    /// </summary>
    /// <param name="dst">The `NativeTensorArray` to copy to.</param>
    /// <param name="dstOffset">The index of the first element in the destination to copy to.</param>
    public void CopyTo(NativeTensorArray dst, int dstOffset)
    {
        Copy(this, 0, dst, dstOffset, Length);
    }

    /// <summary>
    /// Copies the data from a source `NativeTensorArray` to a destination `NativeTensorArray` up to a given length.
    /// </summary>
    /// <param name="sourceArray">The array to copy from.</param>
    /// <param name="destinationArray">The array to copy to.</param>
    /// <param name="length">The number of elements to copy.</param>
    /// <param name="srcIndex">The index of the first element to copy from.</param>
    public static void Copy(NativeTensorArray sourceArray, NativeTensorArray destinationArray, int length = -1, int srcIndex = 0)
    {
        Copy(sourceArray, srcIndex, destinationArray, 0, length);
    }

    /// <summary>
    /// Copies the data from a source array to a destination `NativeTensorArray` up to a given length.
    /// </summary>
    /// <param name="sourceArray">The array to copy from.</param>
    /// <param name="destinationArray">The array to copy to.</param>
    /// <param name="length">The number of elements to copy.</param>
    /// <param name="srcIndex">The index of the first element to copy from.</param>
    /// <typeparam name="T">The data type of the elements.</typeparam>
    public static void Copy<T>(T[] sourceArray, NativeTensorArray destinationArray, int length = -1, int srcIndex = 0) where T : unmanaged
    {
        Copy(sourceArray, srcIndex, destinationArray, 0, length);
    }

    /// <summary>
    /// Copies the data from a source array to a destination `NativeTensorArray` up to a given length.
    /// </summary>
    /// <param name="sourceArray">The array to copy from.</param>
    /// <param name="destinationArray">The array to copy to.</param>
    /// <param name="length">The number of elements to copy.</param>
    /// <param name="srcIndex">The index of the first element to copy from.</param>
    /// <typeparam name="T">The data type of the elements.</typeparam>
    public static void Copy<T>(NativeArray<T> sourceArray, NativeTensorArray destinationArray, int length = -1, int srcIndex = 0) where T : unmanaged
    {
        Copy(sourceArray, srcIndex, destinationArray, 0, length);
    }

    /// <summary>
    /// Copies the data from a source array to a destination array up to a given length.
    /// </summary>
    /// <param name="sourceArray">The array to copy from.</param>
    /// <param name="destinationArray">The array to copy to.</param>
    /// <param name="length">The number of elements to copy.</param>
    /// <param name="srcIndex">The index of the first element to copy from.</param>
    /// <typeparam name="T">The data type of the elements.</typeparam>
    public static void Copy<T>(NativeTensorArray sourceArray, T[] destinationArray, int length = -1, int srcIndex = 0) where T : unmanaged
    {
        Copy(sourceArray, srcIndex, destinationArray, 0, length);
    }

    /// <summary>
    /// Copies the data from a source `NativeTensorArray` to a destination `NativeTensorArray` up to a given length starting from given indexes.
    /// </summary>
    /// <param name="sourceArray">The array to copy from.</param>
    /// <param name="sourceIndex">The index of the first element to copy from.</param>
    /// <param name="destinationArray">The array to copy to.</param>
    /// <param name="destinationIndex">The index of the first element to copy to.</param>
    /// <param name="length">The number of elements.</param>
    /// <exception cref="ArgumentException">Thrown if the given indexes and length are out of bounds of the source or destination array.</exception>
    public static unsafe void Copy(NativeTensorArray sourceArray, int sourceIndex, NativeTensorArray destinationArray, int destinationIndex, int length)
    {
        if (length < 0)
            length = sourceArray.Length;
        if (length == 0)
            return;
        if (sourceIndex + length > sourceArray.Length)
            throw new ArgumentException($"Cannot copy {length} element from sourceIndex {sourceIndex} and Sentis array of length {sourceArray.Length}.");
        if (destinationIndex + length > destinationArray.Length)
            throw new ArgumentException($"Cannot copy {length} element to sourceIndex {destinationIndex} and Sentis array of length {destinationArray.Length}.");

        int itemSize = sourceArray.DataItemSize();
        void* srcPtr = (byte*)sourceArray.RawPtr + sourceIndex * itemSize;
        void* dstPtr = (byte*)destinationArray.RawPtr + destinationIndex * itemSize;
        UnsafeUtility.MemCpy(dstPtr, srcPtr, length * itemSize);
    }

    /// <summary>
    /// Copies the data from a source `NativeTensorArray` to a destination array up to a given length starting from given indexes.
    /// </summary>
    /// <param name="sourceArray">The array to copy from.</param>
    /// <param name="sourceIndex">The index of the first element to copy from.</param>
    /// <param name="destinationArray">The array to copy to.</param>
    /// <param name="destinationIndex">The index of the first element to copy to.</param>
    /// <param name="length">The number of elements.</param>
    /// <typeparam name="T">The data type of the elements.</typeparam>
    /// <exception cref="ArgumentException">Thrown if the given indexes and length are out of bounds of the source or destination array.</exception>
    public static unsafe void Copy<T>(NativeTensorArray sourceArray, int sourceIndex, T[] destinationArray, int destinationIndex, int length) where T : unmanaged
    {
        if (length < 0)
            length = sourceArray.Length;
        if (length == 0)
            return;
        if (sourceIndex + length > sourceArray.Length)
            throw new ArgumentException($"Cannot copy {length} element from sourceIndex {sourceIndex} and Sentis array of length {sourceArray.Length}.");
        if (destinationIndex + length > destinationArray.Length)
            throw new ArgumentException($"Cannot copy {length} element to sourceIndex {destinationIndex} and array of length {destinationArray.Length}.");

        fixed (void* dstPtr = &destinationArray[destinationIndex])
        {
            int itemSize = sourceArray.DataItemSize();
            void* srcPtr = (byte*)sourceArray.RawPtr + sourceIndex * itemSize;
            UnsafeUtility.MemCpy(dstPtr, srcPtr, length * itemSize);
        }
    }

    /// <summary>
    /// Copies the data from a source `NativeTensorArray` to a destination array up to a given length starting from given indexes.
    /// </summary>
    /// <param name="sourceArray">The array to copy from.</param>
    /// <param name="sourceIndex">The index of the first element to copy from.</param>
    /// <param name="destinationArray">The array to copy to.</param>
    /// <param name="destinationIndex">The index of the first element to copy to.</param>
    /// <param name="length">The number of elements.</param>
    /// <typeparam name="T">The data type of the elements.</typeparam>
    /// <exception cref="ArgumentException">Thrown if the given indexes and length are out of bounds of the source or destination array.</exception>
    public static unsafe void Copy<T>(NativeTensorArray sourceArray, int sourceIndex, NativeArray<T> destinationArray, int destinationIndex, int length) where T : unmanaged
    {
        if (length < 0)
            length = sourceArray.Length;
        if (length == 0)
            return;
        if (sourceIndex + length > sourceArray.Length)
            throw new ArgumentException($"Cannot copy {length} element from sourceIndex {sourceIndex} and Sentis array of length {sourceArray.Length}.");
        if (destinationIndex + length > destinationArray.Length)
            throw new ArgumentException($"Cannot copy {length} element to sourceIndex {destinationIndex} and array of length {destinationArray.Length}.");

        int itemSize = sourceArray.DataItemSize();
        void* srcPtr = (byte*)sourceArray.RawPtr + sourceIndex * itemSize;
        void* dstPtr = (byte*)destinationArray.GetUnsafeReadOnlyPtr<T>() + destinationIndex * itemSize;

        UnsafeUtility.MemCpy(dstPtr, srcPtr, length * itemSize);
    }

    /// <summary>
    /// Copies the data from a source array to a destination `NativeTensorArray` up to a given length starting from given indexes.
    /// </summary>
    /// <param name="sourceArray">The array to copy from.</param>
    /// <param name="sourceIndex">The index of the first element to copy from.</param>
    /// <param name="destinationArray">The array to copy to.</param>
    /// <param name="destinationIndex">The index of the first element to copy to.</param>
    /// <param name="length">The number of elements.</param>
    /// <typeparam name="T">The data type of the elements.</typeparam>
    /// <exception cref="ArgumentException">Thrown if the given indexes and length are out of bounds of the source or destination array.</exception>
    public static unsafe void Copy<T>(T[] sourceArray, int sourceIndex, NativeTensorArray destinationArray, int destinationIndex, int length) where T : unmanaged
    {
        if (length < 0)
            length = sourceArray.Length;
        if (length == 0)
            return;
        if (sourceIndex + length > sourceArray.Length)
            throw new ArgumentException($"Cannot copy {length} element from sourceIndex {sourceIndex} and Sentis array of length {sourceArray.Length}.");
        if (destinationIndex + length > destinationArray.Length)
            throw new ArgumentException($"Cannot copy {length} element to sourceIndex {destinationIndex} and Sentis array of length {destinationArray.Length}.");

        fixed (void* srcPtr = &sourceArray[sourceIndex])
        {
            int itemSize = destinationArray.DataItemSize();
            void* dstPtr = (byte*)destinationArray.RawPtr + destinationIndex * itemSize;
            UnsafeUtility.MemCpy(dstPtr, srcPtr, length * itemSize);
        }
    }

    /// <summary>
    /// Copies the data from a source `NativeArray` to a destination array up to a given length starting from given indexes.
    /// </summary>
    /// <param name="sourceArray">The array to copy from.</param>
    /// <param name="sourceIndex">The index of the first element to copy from.</param>
    /// <param name="destinationArray">The array to copy to.</param>
    /// <param name="destinationIndex">The index of the first element to copy to.</param>
    /// <param name="length">The number of elements.</param>
    /// <typeparam name="T">The data type of the elements.</typeparam>
    /// <exception cref="ArgumentException">Thrown if the given indexes and length are out of bounds of the source or destination array.</exception>
    public static unsafe void Copy<T>(NativeArray<T> sourceArray, int sourceIndex, NativeTensorArray destinationArray, int destinationIndex, int length) where T : unmanaged
    {
        if (length < 0)
            length = sourceArray.Length;
        if (length == 0)
            return;
        if (sourceIndex + length > sourceArray.Length)
            throw new ArgumentException($"Cannot copy {length} element from sourceIndex {sourceIndex} and Sentis array of length {sourceArray.Length}.");
        if (destinationIndex + length > destinationArray.Length)
            throw new ArgumentException($"Cannot copy {length} element to sourceIndex {destinationIndex} and Sentis array of length {destinationArray.Length}.");

        int itemSize = destinationArray.DataItemSize();
        void* srcPtr = (byte*)sourceArray.GetUnsafeReadOnlyPtr<T>() + sourceIndex * itemSize;
        void* dstPtr = (byte*)destinationArray.RawPtr + destinationIndex * itemSize;
        UnsafeUtility.MemCpy(dstPtr, srcPtr, length * itemSize);
    }

    /// <summary>
    /// Copies the data from a source `NativeArray` to a destination array up to a given length starting from given indexes.
    /// </summary>
    /// <param name="sourceArray">The array to copy from.</param>
    /// <param name="sourceIndex">The index of the first element to copy from.</param>
    /// <param name="destinationArray">The array to copy to.</param>
    /// <param name="destinationIndex">The index of the first element to copy to.</param>
    /// <param name="length">The number of elements.</param>
    /// <typeparam name="T">The data type of the elements.</typeparam>
    /// <exception cref="ArgumentException">Thrown if the given indexes and length are out of bounds of the source or destination array.</exception>
    public static unsafe void Copy<T>(NativeArray<T>.ReadOnly sourceArray, int sourceIndex, NativeTensorArray destinationArray, int destinationIndex, int length) where T : unmanaged
    {
        if (length < 0)
            length = sourceArray.Length;
        if (length == 0)
            return;
        if (sourceIndex + length > sourceArray.Length)
            throw new ArgumentException($"Cannot copy {length} element from sourceIndex {sourceIndex} and Sentis array of length {sourceArray.Length}.");
        if (destinationIndex + length > destinationArray.Length)
            throw new ArgumentException($"Cannot copy {length} element to sourceIndex {destinationIndex} and Sentis array of length {destinationArray.Length}.");

        int itemSize = destinationArray.DataItemSize();
        void* srcPtr = (byte*)sourceArray.GetUnsafeReadOnlyPtr<T>() + sourceIndex * itemSize;
        void* dstPtr = (byte*)destinationArray.RawPtr + destinationIndex * itemSize;
        UnsafeUtility.MemCpy(dstPtr, srcPtr, length * itemSize);
    }

    /// <summary>
    /// Copies the data from a source `NativeTensorArray` to a destination byte array up to a given length starting from given offsets.
    /// </summary>
    /// <param name="sourceArray">The array to copy from.</param>
    /// <param name="sourceOffset">The index of the first element to copy from.</param>
    /// <param name="destinationArray">The array to copy to.</param>
    /// <param name="destinationByteOffset">The offset in bytes to copy to in the destination array.</param>
    /// <param name="lengthInBytes">The number of bytes to copy.</param>
    /// <exception cref="ArgumentException">Thrown if the given indexes and length are out of bounds of the source or destination array.</exception>
    public static unsafe void BlockCopy(NativeTensorArray sourceArray, int sourceOffset, byte[] destinationArray, int destinationByteOffset, int lengthInBytes)
    {
        int itemSize = sourceArray.SizeOfType;
        int srcLengthBytes = sourceArray.Length * itemSize;
        int srcOffsetBytes = sourceOffset * itemSize;

        if (lengthInBytes == 0)
            return;
        if (lengthInBytes < 0)
            lengthInBytes = srcLengthBytes;

        if (srcOffsetBytes + lengthInBytes > srcLengthBytes)
            throw new ArgumentException($"Cannot copy {lengthInBytes} bytes from sourceByteOffset {srcOffsetBytes} and NativeTensorArray of {srcLengthBytes} num bytes.");
        if (destinationByteOffset + lengthInBytes > destinationArray.Length)
            throw new ArgumentException($"Cannot copy {lengthInBytes} bytes to destinationByteOffset {destinationByteOffset} and byte[] array of {destinationArray.Length} num bytes.");

        fixed (void* dstPtr = &destinationArray[destinationByteOffset])
        {
            void* srcPtr = (byte*)sourceArray.RawPtr + srcOffsetBytes;
            UnsafeUtility.MemCpy(dstPtr, srcPtr, lengthInBytes);
        }
    }

    /// <summary>
    /// Copies the data from a source byte array to a destination `NativeTensorArray` up to a given length starting from given offsets.
    /// </summary>
    /// <param name="sourceArray">The array to copy from.</param>
    /// <param name="sourceByteOffset">The offset in bytes to copy from.</param>
    /// <param name="destinationArray">The array to copy to.</param>
    /// <param name="destinationByteOffset">The offset in bytes to copy to in the destination array.</param>
    /// <param name="lengthInBytes">The number of bytes to copy.</param>
    /// <exception cref="ArgumentException">Thrown if the given indexes and length are out of bounds of the source or destination array.</exception>
    public static unsafe void BlockCopy(byte[] sourceArray, int sourceByteOffset, NativeTensorArray destinationArray, int destinationByteOffset, int lengthInBytes)
    {
        if (lengthInBytes == 0)
            return;
        if (lengthInBytes < 0)
            lengthInBytes = sourceArray.Length;

        if (sourceByteOffset + lengthInBytes > sourceArray.Length)
            throw new ArgumentException($"Cannot copy {lengthInBytes} bytes from sourceByteOffset {sourceByteOffset} and byte[] array of {sourceArray.Length} num bytes.");
        var fullDestPaddedSizeInByte = destinationArray.Length * destinationArray.DataItemSize();
        if (destinationByteOffset + lengthInBytes > fullDestPaddedSizeInByte)
            throw new ArgumentException($"Cannot copy {lengthInBytes} bytes to destinationByteOffset {destinationByteOffset} and byte[] array of {destinationArray.Length} num bytes.");

        void* dstPtr = (byte*)destinationArray.RawPtr + destinationByteOffset;
        fixed (void* srcPtr = &sourceArray[sourceByteOffset])
        {
            UnsafeUtility.MemCpy(dstPtr, srcPtr, lengthInBytes);
        }
    }
}

} // namespace Unity.Sentis
