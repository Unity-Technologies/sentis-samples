using UnityEngine.Assertions;
using System;
using System.Text;
using UnityEngine;

namespace Unity.Sentis {

/// <summary>
/// Represents a struct used to iterate over a `TensorShape`.
/// </summary>
public unsafe struct TensorNDIterator
{
    TensorShape m_Shape;
    /// <summary>
    /// The shape that is iterated over.
    /// </summary>
    public TensorShape shape => m_Shape;

    /// <summary>
    /// The 1D flattened index.
    /// </summary>
    public int index;

    int m_D7;
    int m_D6;
    int m_D5;
    int m_D4;
    int m_D3;
    int m_D2;
    int m_D1;
    int m_D0;

    /// <summary>
    /// Returns a copy of another `TensorNDIterator`.
    /// </summary>
    /// <param name="other">The iterator to copy.</param>
    public TensorNDIterator(TensorNDIterator other)
    {
        m_Shape = other.shape;
        index = other.index;
        m_D0 = other.m_D0;
        m_D1 = other.m_D1;
        m_D2 = other.m_D2;
        m_D3 = other.m_D3;
        m_D4 = other.m_D4;
        m_D5 = other.m_D5;
        m_D6 = other.m_D6;
        m_D7 = other.m_D7;
    }

    /// <summary>
    /// Initializes and returns an instance of `TensorNDIterator` with a given shape.
    /// </summary>
    /// <param name="shape">The shape of the iterator.</param>
    public TensorNDIterator(TensorShape shape)
    {
        m_Shape = shape;
        index = 0;
        m_D0 = 0; m_D1 = 0; m_D2 = 0; m_D3 = 0; m_D4 = 0; m_D5 = 0; m_D6 = 0; m_D7 = 0;
    }

    /// <summary>
    /// Initializes and returns an instance of `TensorNDIterator` with a given shape, and uses a given index in the flattened 1D version of the shape.
    /// </summary>
    /// <param name="shape">The shape of the iterator.</param>
    /// <param name="index">The index in the flattened shape.</param>
    public TensorNDIterator(TensorShape shape, int index)
    {
        Logger.AssertIsTrue(index < shape.length, "TensorNDIterator.ValueError: shape meant for iteration length {0} is smaller than given flattened 1D index {1}", shape.length, index);
        m_Shape = shape;
        this.index = index;
        m_D0 = 0; m_D1 = 0; m_D2 = 0; m_D3 = 0; m_D4 = 0; m_D5 = 0; m_D6 = 0; m_D7 = 0;

        UnravelNDIterator(index);
    }

    /// <summary>
    /// Copies the dimension indices from another iterator. If the shapes of the iterators are not the same the final flattened index may be different.
    /// </summary>
    /// <param name="other">The iterator to copy.</param>
    public void CopyNDIndex(TensorNDIterator other)
    {
        m_D0 = other.m_D0;
        m_D1 = other.m_D1;
        m_D2 = other.m_D2;
        m_D3 = other.m_D3;
        m_D4 = other.m_D4;
        m_D5 = other.m_D5;
        m_D6 = other.m_D6;
        m_D7 = other.m_D7;
        RavelShapeToIndex();
    }

    /// <summary>
    /// Creates a new iterator by skipping an axis of this iterator.
    /// </summary>
    /// <param name="axis">The axis to skip.</param>
    /// <returns>The created iterator.</returns>
    public TensorNDIterator RemoveDim(int axis)
    {
        Logger.AssertIsTrue(shape.rank > 0, "TensorNDIterator.ValueError: iterator is iterating on rank 0 shape, cannot remove additional dimensions");
        axis = shape.Axis(axis);

        TensorNDIterator removed = new TensorNDIterator();
        removed.m_Shape = TensorShape.Ones(shape.rank-1);

        int idx = 0;
        int trailing = 1;

        int* dst = &removed.m_D7;
        fixed (int* src = &this.m_D7)
        {
            int ii = (removed.shape.rank - 1);
            for (int i = (shape.rank-1); i >= 0; i--)
            {
                if (i == axis)
                    continue;

                removed.m_Shape[ii] = shape[i];
                dst[(TensorShape.maxRank - removed.shape.rank) + ii] = src[(TensorShape.maxRank - shape.rank) + i];

                idx += trailing * dst[(TensorShape.maxRank - removed.shape.rank) + ii];
                trailing *= removed.shape[ii];

                ii--;
            }
        }

        removed.index = idx;

        return removed;
    }

    /// <summary>
    /// Creates a new iterator by broadcasting this iterator on a given shape following the broadcast rule.
    /// </summary>
    /// <param name="shapeToBroadcast">The shape to broadcast with.</param>
    /// <returns>The broadcast iterator.</returns>
    public TensorNDIterator Broadcast(TensorShape shapeToBroadcast)
    {
        TensorNDIterator broadcast = new TensorNDIterator(shapeToBroadcast);

        int idx = 0;
        int trailingDim = 1;

        int* dst = &broadcast.m_D7;
        fixed (int* src = &this.m_D7)
        {
            for (int i = (shapeToBroadcast.rank-1); i >= 0; i--)
            {
                int srcA = shape[(shape.rank - 1) - ((shapeToBroadcast.rank - 1) - i)];
                int srcB = shapeToBroadcast[i];
                Logger.AssertIsTrue((srcA == srcB) || (srcA == 1) || (srcB == 1) || (srcA == 0 && srcB != 0) || (srcB == 0 && srcA != 0), "TensorNDIterator.ValueError: operands could not be broadcast together with shapes {0}, {1}", shape, shapeToBroadcast);
                dst[(TensorShape.maxRank - shapeToBroadcast.rank) + i] = src[(TensorShape.maxRank - shapeToBroadcast.rank) + i] % shapeToBroadcast[i];
                idx += trailingDim * dst[(TensorShape.maxRank - shapeToBroadcast.rank) + i];
                trailingDim *= shapeToBroadcast[i];
            }
        }

        broadcast.index = idx;

        return broadcast;
    }

    /// <summary>
    /// Creates a new iterator by transposing this iterator using the given permutations.
    /// </summary>
    /// <param name="permutations">The permutation array.</param>
    /// <returns>The transposed iterator.</returns>
    public TensorNDIterator Transpose(int[] permutations)
    {
        Logger.AssertAreEqual(shape.rank, permutations.Length, "TensorNDIterator.ValueError: shape ranks and permutations length do not match {0}, {1}", shape, permutations.Length);
        TensorNDIterator transpose = new TensorNDIterator(shape.Transpose(permutations));

        int idx = 0;
        int trailingDim = 1;

        int* dst = &transpose.m_D7;
        fixed (int* src = &this.m_D7)
        {
            for (int i = (shape.rank-1); i >= 0; i--)
            {
                int ti = permutations[i];
                dst[(TensorShape.maxRank - shape.rank) + i] = src[(TensorShape.maxRank - shape.rank) + ti];
                idx += trailingDim * dst[(TensorShape.maxRank - shape.rank) + i];
                trailingDim *= shape[ti];
            }
        }

        transpose.index = idx;

        return transpose;
    }

    /// <summary>
    /// Creates a new iterator by transposing this iterator reversing the axes.
    /// </summary>
    /// <returns>The transposed iterator.</returns>
    public TensorNDIterator Transpose()
    {
        TensorNDIterator transpose = new TensorNDIterator(shape.Transpose());

        int idx = 0;
        int trailingDim = 1;

        int* dst = &transpose.m_D7;
        fixed (int* src = &this.m_D7)
        {
            for (int i = (shape.rank-1); i >= 0; i--)
            {
                int ti = (shape.rank - 1) - i;
                dst[(TensorShape.maxRank - m_Shape.rank) + i] = src[(TensorShape.maxRank - shape.rank) + ti];
                idx += trailingDim * dst[(TensorShape.maxRank - shape.rank) + i];
                trailingDim *= shape[ti];
            }
        }

        transpose.index = idx;

        return transpose;
    }

    void UnravelNDIterator(int index)
    {
        Logger.AssertIsTrue(index < shape.length, "TensorNDIterator.ValueError: shape length {0} is smaller than given flattened 1D index {1}", shape.length, index);
        this.index = index;

        fixed (int* src = &this.m_D7)
        {
            int trailing = index;
            for (int i = (shape.rank-1); i >= 0; i--)
            {
                src[(TensorShape.maxRank - shape.rank) + i] = trailing % shape[i];
                trailing /= shape[i];
            }
        }
    }

    void RavelShapeToIndex()
    {
        this.index = 0;
        int trailing = 1;
        for (int i = (shape.rank-1); i >= 0; i--)
        {
            this.index += trailing * this[i];
            trailing *= shape[i];
        }
    }

    /// <summary>
    /// Increments the flattened index by one.
    /// </summary>
    public void MoveNext()
    {
        if (!HasNext())
            return;

        ++index;

        // carry-over chain
        fixed (int* src = &this.m_D7)
        {
            src[(TensorShape.maxRank - 1)] += 1;

            if (shape.rank == 0)
                return;

            for (int i = 0; i < TensorShape.maxRank; i++)
            {
                if (i >= shape.rank)
                    return;
                if (src[(TensorShape.maxRank - 1) - i] < shape.UnsafeGet((TensorShape.maxRank - 1) - i))
                    return;
                src[(TensorShape.maxRank - 1) - i] = 0;
                src[(TensorShape.maxRank - 1) - (i + 1)] += 1;
            }
        }
    }

    /// <summary>
    /// Increments the index at a given axis by one.
    /// </summary>
    /// <param name="axis">The axis along which to increment.</param>
    public void MoveNextAxis(int axis)
    {
        if (!HasNext())
            return;
        if (!HasNext(axis))
            return;
        axis = shape.Axis(axis);

        int trailing = 1;
        for (int i = (shape.rank-1); i > axis; i--)
        {
            trailing *= shape[i];
        }

        this.index += trailing;

        fixed (int* src = &this.m_D7)
        {
            src[(TensorShape.maxRank - 1) - ((shape.rank - 1) - axis)] += 1;
        }
    }

    /// <summary>
    /// Whether the iterator is yet to reach the end of the shape.
    /// </summary>
    /// <returns>Whether the iterator can increment.</returns>
    public bool HasNext()
    {
        return index < shape.length;
    }

    /// <summary>
    /// Whether the iterator is yet to reach the end of the shape on a given axis.
    /// </summary>
    /// <param name="axis">The axis along which to check.</param>
    /// <returns>Whether the iterator can increment along the axis.</returns>
    public bool HasNext(int axis)
    {
        return HasNext() && (this[axis] < shape[axis]);
    }

    /// <summary>
    /// Resets the iterator to the start of the shape.
    /// </summary>
    public void Reset()
    {
        index = 0;
        m_D0 = 0; m_D1 = 0; m_D2 = 0; m_D3 = 0; m_D4 = 0; m_D5 = 0; m_D6 = 0; m_D7 = 0;
    }

    /// <summary>
    /// Gets or sets the iterator at a given axis.
    /// </summary>
    /// <param name="axis">The axis at which to get or set the index.</param>
    public int this[int axis]
    {
        get
        {
            axis = shape.Axis(axis);
            Assert.IsTrue(shape.rank > axis);
            fixed (int* src = &m_D7)
            {
                return src[(TensorShape.maxRank - shape.rank) + axis];
            }
        }

        set
        {
            axis = shape.Axis(axis);
            Assert.IsTrue(shape.rank > axis);
            fixed (int* src = &m_D7)
            {
                src[(TensorShape.maxRank - shape.rank) + axis] = value;
            }
            RavelShapeToIndex();
        }
    }

    /// <summary>
    /// Returns a string that represents the `TensorNDIterator`.
    /// </summary>
    /// <returns>The string representation of the iterator.</returns>
    public override string ToString()
    {
        StringBuilder sb = new StringBuilder();
        sb.Append("(");
        for (int i = 0; i < shape.rank; i++)
        {
            if (i != 0)
                sb.Append(", ");
            sb.Append(this[i]);
        }
        sb.Append(") <");
        sb.Append(shape);
        sb.Append(">");
        return sb.ToString();
    }
}

} // namespace Unity.Sentis
