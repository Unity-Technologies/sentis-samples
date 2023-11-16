using UnityEngine.Assertions;
using System;
using System.Text;
using UnityEngine;
using Unity.Sentis;

using System.Runtime.CompilerServices;

[assembly: InternalsVisibleTo("Unity.Sentis.Tests")]
[assembly: InternalsVisibleTo("Unity.Sentis.EditorTests")]
[assembly: InternalsVisibleTo("Unity.Sentis.CPUBackend")]

namespace Unity.Sentis {

/// <summary>
/// Represents the shape of a tensor.
/// </summary>
[Serializable]
public unsafe struct TensorShape
{
    /// <summary>
    /// The maximum rank a `TensorShape` can have.
    /// </summary>
    public const int maxRank = 8;

#pragma warning disable CS0649

    // disable warning: Field 'field' is never assigned to, and will always have its default value 'value'
    // fields are accessed using unsafe code (ptr)
    // TensorShapes are 0 padded from right to left to match TensorDim slices
    // Ex:
    // (5)       -> 0,0,0,5
    // (3,5)     -> 0,0,3,5
    // (7,3,5)   -> 0,7,3,5
    // (6,7,3,5) -> 6,7,3,5
    // ...
    int m_D7;
    int m_D6;
    int m_D5;
    int m_D4;
    int m_D3;
    int m_D2;
    int m_D1;
    int m_D0;
#pragma warning restore CS0649

    int m_Rank;

    /// <summary>
    /// The rank of a `TensorShape`. For example, a tensor of shape (5) has a rank of 1. A tensor of shape (7, 3, 5) has a rank of 3.
    /// </summary>
    public int rank => m_Rank;

    int m_Length;

    /// <summary>
    /// The number of elements represented by the `TensorShape`. For example a shape of (1, 2, 3, 4) represents 24 elements: `1 * 2 * 3 * 4`.
    /// </summary>
    public int length => rank == 0 ? 1 : m_Length;

    /// <summary>
    /// Initializes and returns an instance of `TensorShape` with a rank of 8: (d7, d6, d5, d4, d3, d2, d1, d0).
    ///
    /// For example (2, 3, 4, 5, 6, 7, 8, 9).
    /// </summary>
    /// <param name="d7">Length of axis 7.</param>
    /// <param name="d6">Length of axis 6.</param>
    /// <param name="d5">Length of axis 5.</param>
    /// <param name="d4">Length of axis 4.</param>
    /// <param name="d3">Length of axis 3.</param>
    /// <param name="d2">Length of axis 2.</param>
    /// <param name="d1">Length of axis 1.</param>
    /// <param name="d0">Length of axis 0.</param>
    public TensorShape(int d7, int d6, int d5, int d4, int d3, int d2, int d1, int d0)
    {
        m_D7 = d7 >= 0 ? d7 : 0;
        m_D6 = d6 >= 0 ? d6 : 0;
        m_D5 = d5 >= 0 ? d5 : 0;
        m_D4 = d4 >= 0 ? d4 : 0;
        m_D3 = d3 >= 0 ? d3 : 0;
        m_D2 = d2 >= 0 ? d2 : 0;
        m_D1 = d1 >= 0 ? d1 : 0;
        m_D0 = d0 >= 0 ? d0 : 0;

        m_Rank = 8;
        m_Length = d7 * d6 * d5 * d4 * d3 * d2 * d1 * d0;
    }

    /// <summary>
    /// Initializes and returns an instance of `TensorShape` with a rank of 7: (d6, d5, d4, d3, d2, d1, d0).
    ///
    /// For example (3, 4, 5, 6, 7, 8, 9).
    /// </summary>
    /// <param name="d6">Length of axis 6.</param>
    /// <param name="d5">Length of axis 5.</param>
    /// <param name="d4">Length of axis 4.</param>
    /// <param name="d3">Length of axis 3.</param>
    /// <param name="d2">Length of axis 2.</param>
    /// <param name="d1">Length of axis 1.</param>
    /// <param name="d0">Length of axis 0.</param>
    public TensorShape(int d6, int d5, int d4, int d3, int d2, int d1, int d0)
    {
        m_D7 = 0;
        m_D6 = d6 >= 0 ? d6 : 0;
        m_D5 = d5 >= 0 ? d5 : 0;
        m_D4 = d4 >= 0 ? d4 : 0;
        m_D3 = d3 >= 0 ? d3 : 0;
        m_D2 = d2 >= 0 ? d2 : 0;
        m_D1 = d1 >= 0 ? d1 : 0;
        m_D0 = d0 >= 0 ? d0 : 0;

        m_Rank = 7;
        m_Length = d6 * d5 * d4 * d3 * d2 * d1 * d0;
    }

    /// <summary>
    /// Initializes and returns an instance of `TensorShape` with a rank of 6: (d5, d4, d3, d2, d1, d0).
    ///
    /// For example (4, 5, 6, 7, 8, 9).
    /// </summary>
    /// <param name="d5">Length of axis 5.</param>
    /// <param name="d4">Length of axis 4.</param>
    /// <param name="d3">Length of axis 3.</param>
    /// <param name="d2">Length of axis 2.</param>
    /// <param name="d1">Length of axis 1.</param>
    /// <param name="d0">Length of axis 0.</param>
    public TensorShape(int d5, int d4, int d3, int d2, int d1, int d0)
    {
        m_D7 = 0;
        m_D6 = 0;
        m_D5 = d5 >= 0 ? d5 : 0;
        m_D4 = d4 >= 0 ? d4 : 0;
        m_D3 = d3 >= 0 ? d3 : 0;
        m_D2 = d2 >= 0 ? d2 : 0;
        m_D1 = d1 >= 0 ? d1 : 0;
        m_D0 = d0 >= 0 ? d0 : 0;

        m_Rank = 6;
        m_Length = d5 * d4 * d3 * d2 * d1 * d0;
    }

    /// <summary>
    /// Initializes and returns an instance of `TensorShape` with a rank of 5: (d4, d3, d2, d1, d0).
    ///
    /// For example (5, 6, 7, 8, 9).
    /// </summary>
    /// <param name="d4">Length of axis 4.</param>
    /// <param name="d3">Length of axis 3.</param>
    /// <param name="d2">Length of axis 2.</param>
    /// <param name="d1">Length of axis 1.</param>
    /// <param name="d0">Length of axis 0.</param>
    public TensorShape(int d4, int d3, int d2, int d1, int d0)
    {
        m_D7 = 0;
        m_D6 = 0;
        m_D5 = 0;
        m_D4 = d4 >= 0 ? d4 : 0;
        m_D3 = d3 >= 0 ? d3 : 0;
        m_D2 = d2 >= 0 ? d2 : 0;
        m_D1 = d1 >= 0 ? d1 : 0;
        m_D0 = d0 >= 0 ? d0 : 0;

        m_Rank = 5;
        m_Length = d4 * d3 * d2 * d1 * d0;
    }

    /// <summary>
    /// Initializes and returns an instance of `TensorShape` with a rank of 4: (d3, d2, d1, d0).
    ///
    /// For example (6, 7, 8, 9).
    /// </summary>
    /// <param name="d3">Length of axis 3.</param>
    /// <param name="d2">Length of axis 2.</param>
    /// <param name="d1">Length of axis 1.</param>
    /// <param name="d0">Length of axis 0.</param>
    public TensorShape(int d3, int d2, int d1, int d0)
    {
        m_D7 = 0;
        m_D6 = 0;
        m_D5 = 0;
        m_D4 = 0;
        m_D3 = d3 >= 0 ? d3 : 0;
        m_D2 = d2 >= 0 ? d2 : 0;
        m_D1 = d1 >= 0 ? d1 : 0;
        m_D0 = d0 >= 0 ? d0 : 0;

        m_Rank = 4;
        m_Length = d3 * d2 * d1 * d0;
    }

    /// <summary>
    /// Initializes and returns an instance of `TensorShape` with a rank of 3: (d2, d1, d0).
    ///
    /// For example (7, 8, 9).
    /// </summary>
    /// <param name="d2">Length of axis 2.</param>
    /// <param name="d1">Length of axis 1.</param>
    /// <param name="d0">Length of axis 0.</param>
    public TensorShape(int d2, int d1, int d0)
    {
        m_D7 = 0;
        m_D6 = 0;
        m_D5 = 0;
        m_D4 = 0;
        m_D3 = 0;
        m_D2 = d2 >= 0 ? d2 : 0;
        m_D1 = d1 >= 0 ? d1 : 0;
        m_D0 = d0 >= 0 ? d0 : 0;

        m_Rank = 3;
        m_Length = d2 * d1 * d0;
    }

    /// <summary>
    /// Initializes and returns an instance of `TensorShape` with a rank of 2: (d1, d0).
    ///
    /// For example (8, 9).
    /// </summary>
    /// <param name="d1">Length of axis 1.</param>
    /// <param name="d0">Length of axis 0.</param>
    public TensorShape(int d1, int d0)
    {
        m_D7 = 0;
        m_D6 = 0;
        m_D5 = 0;
        m_D4 = 0;
        m_D3 = 0;
        m_D2 = 0;
        m_D1 = d1 >= 0 ? d1 : 0;
        m_D0 = d0 >= 0 ? d0 : 0;

        m_Rank = 2;
        m_Length = d1 * d0;
    }

    /// <summary>
    /// Initializes and returns an instance of `TensorShape` with a rank of 1: (d0).
    ///
    /// For example (9).
    /// </summary>
    /// <param name="d0">Length of axis 0.</param>
    public TensorShape(int d0)
    {
        m_D7 = 0;
        m_D6 = 0;
        m_D5 = 0;
        m_D4 = 0;
        m_D3 = 0;
        m_D2 = 0;
        m_D1 = 0;
        m_D0 = d0 >= 0 ? d0 : 0;

        m_Rank = 1;
        m_Length = d0;
    }

    /// <summary>
    /// Initializes and returns an instance of `TensorShape` with a given shape. For example: `TensorShape(new [] {3, 4, 5, 6})` returns a tensor with a shape of (3, 4, 5, 6).
    /// </summary>
    /// <param name="shape">The shape as a span.</param>
    public TensorShape(ReadOnlySpan<int> shape)
        : this()
    {
        Logger.AssertIsTrue(shape.Length <= maxRank, "ValueError: TensorShape are capped to rank=8, cannot create tensorshape of rank {0}", shape.Length);

        m_Rank = shape.Length;

        if (shape.Length == 0)
            return;

        fixed (int* dst = &m_D7, src = &shape[0])
        {
            Buffer.MemoryCopy(src, dst + (maxRank - rank), shape.Length * sizeof(int), shape.Length * sizeof(int));
        }

        m_Length = 1;
        for (int i = 0; i < rank; i++)
        {
            m_Length *= shape[i];
        }
    }

    /// <summary>
    /// Returns a copy of another `TensorShape`.
    /// </summary>
    /// <param name="shape">The shape to copy.</param>
    public TensorShape(TensorShape shape)
    {
        m_Rank = shape.rank;
        m_Length = shape.length;

        m_D7 = shape.m_D7;
        m_D6 = shape.m_D6;
        m_D5 = shape.m_D5;
        m_D4 = shape.m_D4;
        m_D3 = shape.m_D3;
        m_D2 = shape.m_D2;
        m_D1 = shape.m_D1;
        m_D0 = shape.m_D0;
    }

    /// <summary>
    /// Accesses internal shape layout:
    /// shape (3,4,5,6), rank 4,  (internally) = ( 0, 0, 0, 0, 3, 4, 5, 6), rank 8
    ///                                        = (d7,d6,d5,d4,d3,d2,d1,d0)
    /// axis: 0 => d7
    ///       1 => d6
    ///       2 => d5
    ///       3 => d4
    ///       4 => d3
    ///       5 => d2
    ///       6 => d1
    ///       7 => d0
    /// </summary>
    internal int UnsafeGet(int axis)
    {
        fixed (int* shape = &m_D7)
        {
            return shape[axis];
        }
    }

    /// <summary>
    /// Update TensorShape Length
    /// shape (3,4,5,6) => length = 3*4*5*6
    /// </summary>
    void RecomputeLength()
    {
        m_Length = 1;
        fixed (int* srcA = &m_D7)
        {
            for (int i = 0; i < rank; i++)
            {
                m_Length *= srcA[maxRank - 1 - i];
            }
        }
    }

    /// <summary>
    /// Gets or sets the tensor shape at a given axis. A negative axis counts backwards from the inner dimension.
    /// </summary>
    /// <param name="axis">The axis to get or set.</param>
    public int this[int axis]
    {
        get
        {
            axis = Axis(axis);

            fixed (int* shape = &m_D7)
            {
                return shape[(maxRank - rank) + axis];
            }
        }

        set
        {
            axis = Axis(axis);

            fixed (int* shape = &m_D7)
            {
                shape[(maxRank - rank) + axis] = value;
            }

            RecomputeLength();
        }
    }

    /// <summary>
    /// Calculates whether any axes are length 0. In this case the length is also 0.
    /// </summary>
    /// <returns>Whether the shape has any axes that are length 0.</returns>
    public bool HasZeroDims()
    {
        return length == 0 && rank > 0;
    }

    /// <summary>
    /// Returns a string that represents the `TensorShape`.
    /// </summary>
    /// <returns>String representation of shape.</returns>
    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.Append("(");
        for (int i = 0; i < rank; i++)
        {
            if (i != 0)
                sb.Append(", ");
            sb.Append(this[i]);
        }

        sb.Append(")");
        return sb.ToString();
    }

    /// <summary>
    /// Returns the number of elements represented by the `TensorShape`, starting from a given axis. A negative axis counts backwards from the inner dimension.
    /// </summary>
    /// <param name="start">The first axis to count length from.</param>
    /// <returns>The number of elements in the shape.</returns>
    public int Length(int start)
    {
        if (start >= rank)
            return 1;
        if (start < -rank)
            return length;

        start = Axis(start);
        int l = 1;
        fixed (int* shape = &m_D7)
        {
            for (int i = start; i < rank; i++)
            {
                l *= shape[(maxRank - rank) + i];
            }
        }

        return l;
    }

    /// <summary>
    /// Returns the number of elements represented by the `TensorShape`, between the start and end axes. Negative axes counts backwards from the inner dimension.
    /// </summary>
    /// <param name="start">The first axis to count length from.</param>
    /// <param name="end">The exclusive final axis to count length to.</param>
    /// <returns>The number of elements in the shape.</returns>
    public int Length(int start, int end)
    {
        if (start >= rank || end < -rank)
            return 1;
        if (start < -rank)
            start = 0;
        if (end > rank)
            end = rank;

        start = Axis(start);
        if (end < rank)
            end = Axis(end);

        int l = 1;
        fixed (int* shape = &m_D7)
        {
            for (int i = start; i < end; i++)
            {
                l *= shape[(maxRank - rank) + i];
            }
        }

        return l;
    }

    /// <summary>
    /// Returns the positive axis corresponding to a given axis. Negative axes counts backwards from the inner dimension.
    /// </summary>
    /// <param name="axis">The axis to wrap.</param>
    /// <returns>The positive axis.</returns>
    public int Axis(int axis)
    {
        Logger.AssertIsTrue(axis >= -rank && axis < rank, "IndexError: axis {0} is out of bounds shape of rank, {1}", axis, rank);
        return axis >= 0 ? axis : rank + axis;
    }

    /// <summary>
    /// Returns the product of the dimensions of the tensor shape after a given axis. Negative axes counts backwards from the inner dimension.
    ///
    /// The strides of a tensor tell us how many elements we have to skip in flattened memory to move to the next position along a given index.
    /// </summary>
    /// <param name="axis">The axis to calculate the stride length at.</param>
    /// <returns>The stride length at the axis.</returns>
    public int Strides(int axis)
    {
        axis = Axis(axis);

        int trailingLength = 1;
        fixed (int* shape = &m_D7)
        {
            for (int i = (rank - 1); i > axis; i--)
            {
                trailingLength *= shape[(maxRank - rank) + i];
            }
        }

        return trailingLength;
    }

    /// <summary>
    /// Returns the `TensorShape` as an array of integers. For example if the `TensorShape` is (5, 2, 3, 4), the method returns new[] {5, 2, 3, 4}.
    /// </summary>
    /// <returns>An integer array representation of the shape.</returns>
    public int[] ToArray()
    {
        var shape = new int[rank];
        if (rank > 0)
        {
            fixed (int* dst = &shape[0], src = &m_D7)
            {
                Buffer.MemoryCopy(src + (maxRank - rank), dst, shape.Length * sizeof(int), shape.Length * sizeof(int));
            }
        }

        return shape;
    }

    /// <summary>
    /// Creates a `TensorShape` by duplicating `this` and removing the dimensions of size 1. For example, if `this` is (5, 1, 3, 1), the method returns (5, 3).
    /// </summary>
    /// <returns>The squeezed tensor shape.</returns>
    public TensorShape Squeeze()
    {
        Logger.AssertIsTrue(rank != 0, "ValueError: cannot squeeze scalar tensor {0}", this);

        var squeezed = new TensorShape();
        int* dst = &squeezed.m_D7;

        int squeezedDim = 0;
        fixed (int* src = &m_D7)
        {
            for (int i = (rank - 1); i >= 0; i--)
            {
                int dim = src[(maxRank - rank) + i];
                if (dim == 1)
                    continue;

                dst[(maxRank - 1) - squeezedDim] = dim;
                squeezedDim++;
            }
        }

        squeezed.m_Rank = squeezedDim;
        squeezed.m_Length = length;

        return squeezed;
    }

    /// <summary>
    /// Creates a `TensorShape` by duplicating `this` and removing the given axis of size 1. For example, if `this` is (5, 1, 3, 1), and `axis` is 1, the method returns (5, 3, 1).
    /// </summary>
    /// <param name="axis">The axis to squeeze.</param>
    /// <returns>The squeezed tensor shape.</returns>
    public TensorShape Squeeze(int axis)
    {
        Logger.AssertIsTrue(rank != 0, "ValueError: cannot squeeze scalar tensor {0}", this);

        var squeezed = new TensorShape();

        if (rank == 1)
            return squeezed;

        squeezed.m_Rank = rank - 1;

        axis = Axis(axis);

        int* dst = &squeezed.m_D7;

        int squeezedDim = 0;
        fixed (int* src = &m_D7)
        {
            Logger.AssertAreEqual(1, src[(maxRank - rank) + axis], "ValueError: squeezing non unit dimension {0}, {1}", this, axis);
            for (int i = (rank - 1); i >= 0; i--)
            {
                int dim = src[(maxRank - rank) + i];
                if (i == axis)
                    continue;

                dst[(maxRank - 1) - squeezedDim] = dim;

                squeezedDim++;
            }
        }

        squeezed.m_Length = length;

        return squeezed;
    }

    /// <summary>
    /// Creates a `TensorShape` by duplicating `this` and removing the given axes of size 1. For example, if `this` is (5, 1, 3, 1), and `axes` is {1, -1}, the method returns (5, 3).
    /// </summary>
    /// <param name="axes">The axes to squeeze.</param>
    /// <returns>The squeezed tensor shape.</returns>
    public TensorShape Squeeze(ReadOnlySpan<int> axes)
    {
        if (axes == null || axes.Length == 0)
            return Squeeze();

        uint axesBitMask = 0;
        for (int i = 0; i < axes.Length; i++)
            axesBitMask |= 1U << Axis(axes[i]);

        Logger.AssertIsTrue(rank - axes.Length >= 0, "ValueError: squeeze axes  {0} larger than current rank {1}. Cannot unsqueeze scalar tensor", rank, axes.Length);

        var squeezed = new TensorShape();
        squeezed.m_Rank = Math.Max(rank - axes.Length, 0);

        int* dst = &squeezed.m_D7;

        int squeezedDim = 0;
        fixed (int* src = &m_D7)
        {
            for (int i = (rank - 1); i >= 0; i--)
            {
                uint baxis = (axesBitMask >> i) & 1U;
                if (baxis == 1)
                {
                    Logger.AssertAreEqual(1, src[(maxRank - rank) + i], "ValueError: squeezing non unit dimension {0}, {1}", this, i);
                    continue;
                }

                dst[(maxRank - 1) - squeezedDim] = src[(maxRank - rank) + i];
                squeezedDim++;
            }
        }

        squeezed.m_Length = length;

        return squeezed;
    }

    /// <summary>
    /// Creates a `TensorShape` by duplicating `this` and inserting a dimension of size one at a given axis. For example if `this` is (2), and the value of `axis` is 0, the method returns (1, 2).
    /// </summary>
    /// <param name="axis">The axis at which to unsqueeze.</param>
    /// <returns>The unsqueezed tensor shape.</returns>
    public TensorShape Unsqueeze(int axis)
    {
        if (rank == 0)
            return new TensorShape(1);

        Logger.AssertIsTrue(rank != maxRank, "ValueError: TensorShape are capped to rank=8, cannot unsqueeze rank 8 tensorshape {0}", this);

        int unsqueezedRank = rank + 1;
        var unsqueezed = new TensorShape();
        unsqueezed.m_Rank = unsqueezedRank;
        unsqueezed.m_Length = length;

        axis = unsqueezed.Axis(axis);

        int* dst = &unsqueezed.m_D7;

        dst[(maxRank - unsqueezedRank) + axis] = 1;

        int shiftDim = 0;
        fixed (int* src = &m_D7)
        {
            for (int i = 0; i < unsqueezedRank; i++)
            {
                if (i == axis)
                {
                    shiftDim = 1;
                    continue;
                }

                dst[(maxRank - unsqueezedRank) + i] = src[(maxRank - rank) + i - shiftDim];
            }
        }

        return unsqueezed;
    }

    /// <summary>
    /// Creates a `TensorShape` by duplicating `this` and inserting a dimension of size one at a given axis. For example if `this` is (2), and the value of `axis` is 0, the method returns (1, 2).
    /// </summary>
    /// <param name="axes">The axes at which to unsqueeze.</param>
    /// <returns>The unsqueezed tensor shape.</returns>
    public TensorShape Unsqueeze(ReadOnlySpan<int> axes)
    {
        Logger.AssertIsTrue(rank + axes.Length <= maxRank, "ValueError: TensorShape are capped to rank=8, cannot unsqueeze tensorshape {0} to rank greater than 8", this);

        int unsqueezedRank = rank + axes.Length;
        var unsqueezed = new TensorShape();
        unsqueezed.m_Rank = unsqueezedRank;
        unsqueezed.m_Length = length;

        uint axesBitMask = 0;
        for (int i = 0; i < axes.Length; i++)
            axesBitMask |= 1U << unsqueezed.Axis(axes[i]);

        int* dst = &unsqueezed.m_D7;

        int shiftDim = 0;
        fixed (int* src = &m_D7)
        {
            for (int i = (unsqueezedRank - 1); i >= 0; i--)
            {
                uint baxis = (axesBitMask >> i) & 1U;
                if (baxis == 1)
                {
                    dst[(maxRank - unsqueezedRank) + i] = 1;
                    continue;
                }

                dst[(maxRank - unsqueezedRank) + i] = src[(maxRank - 1) - shiftDim];
                shiftDim++;
            }
        }

        return unsqueezed;
    }

    /// <summary>
    /// Creates a `TensorShape` by duplicating `this` and reshaping the dimensions to those given.
    ///
    /// If a dimension in the shape array is -1, Sentis infers the value from the size of the `TensorShape` and the remaining dimensions. Only one dimension can be -1.
    /// </summary>
    /// <param name="shape">The new shape as a span of integers.</param>
    /// <param name="allowZero">When the value is `true`, Sentis sets a dimension to zero if the new shape includes a zero. Otherwise Sentis retains the corresponding size at that axis from the original shape.</param>
    /// <returns></returns>
    public TensorShape Reshape(ReadOnlySpan<int> shape, bool allowZero = false)
    {
        Logger.AssertIsTrue(shape.Length <= maxRank, "ValueError: TensorShape are capped to rank=8, cannot create tensorshape of rank {0}", shape.Length);

        int reshapedRank = shape.Length;
        var reshaped = new TensorShape();
        reshaped.m_Rank = reshapedRank;
        reshaped.m_Length = length;

        int* dst = &reshaped.m_D7;

        int unknownDim = -1;
        int newLength = 1;
        fixed (int* src = &m_D7)
        {
            for (int i = 0; i < shape.Length; i++)
            {
                int shapeValue = shape[i];
                if (shapeValue == -1)
                {
                    Logger.AssertIsTrue(unknownDim == -1, "ValueError: at most one dimension of new shape can be -1", this);
                    unknownDim = i;
                }
                else
                {
                    if (!allowZero && shapeValue == 0)
                    {
                        Logger.AssertIsTrue(i < rank, "ValueError: current index {0} out of bounds of the source shape {1}", i, this);
                        shapeValue = src[(maxRank - rank) + i];
                    }

                    dst[(maxRank - reshapedRank) + i] = shapeValue;
                    newLength *= shapeValue;
                }
            }
        }

        if (unknownDim >= 0)
        {
            Logger.AssertIsTrue(newLength != 0 && (length % newLength) == 0, "ValueError: reshaped length does not match with input length");
            dst[(maxRank - reshapedRank) + unknownDim] = length / newLength;
        }
        else
        {
            Logger.AssertIsTrue(newLength == length, "ValueError: reshaped length does not match with input length");
        }

        return reshaped;
    }

    /// <summary>
    /// Creates a `TensorShape` by duplicating `this` and reshaping to a 2D matrix. The dimensions before `axis` are collapsed into the outer dimension and the remaining axes are collapsed into the inner dimension.
    ///
    /// For example, if `this` is (2, 3, 4), and the value of `axis` is 2, the method returns (2 * 3, 4).
    /// </summary>
    /// <param name="axis">The axis at which to flatten.</param>
    /// <returns>The flattened tensor shape.</returns>
    public TensorShape Flatten(int axis = 1)
    {
        axis = axis >= 0 ? axis : rank + axis;
        var flatten = new TensorShape();

        int* dst = &flatten.m_D7;

        flatten.m_Rank = 2;
        flatten.m_Length = length;

        dst[maxRank - 1] = 1;
        dst[maxRank - 2] = 1;

        fixed (int* src = &m_D7)
        {
            for (int i = 0; i < axis; i++)
            {
                dst[maxRank - 2] *= src[(maxRank - rank) + i];
            }

            for (int i = axis; i < rank; i++)
            {
                dst[(maxRank - 2) + 1] *= src[(maxRank - rank) + i];
            }
        }

        return flatten;
    }

    /// <summary>
    /// Creates a `TensorShape` by applying numpy-style broadcasting between `this` and `other`.
    ///
    /// Sentis broadcasts shapes from innermost to outermost dimensions. Two dimensions are compatible when they're equal, or one of the dimensions is 1.
    /// </summary>
    /// <param name="other">The other tensor shape which which to broadcast.</param>
    /// <returns>The broadcast tensor shape.</returns>
    public TensorShape Broadcast(TensorShape other)
    {
        TensorShape broadcast = new TensorShape();
        broadcast.m_Rank = Math.Max(rank, other.rank);

        int* srcB = &other.m_D7;
        int* dst = &broadcast.m_D7;
        fixed (int* srcA = &m_D7)
        {
            dst[0] = !(rank > 7) || (srcA[0] == 1 && other.rank > 7) ? srcB[0] : srcA[0]; // srcA[0] == 1 && shape.rank > 7 ? srcB[0] : this.rank > 7 ? srcA[0] : srcB[0];
            dst[1] = !(rank > 6) || (srcA[1] == 1 && other.rank > 6) ? srcB[1] : srcA[1]; // srcA[1] == 1 && shape.rank > 6 ? srcB[1] : this.rank > 6 ? srcA[1] : srcB[1];
            dst[2] = !(rank > 5) || (srcA[2] == 1 && other.rank > 5) ? srcB[2] : srcA[2]; // srcA[2] == 1 && shape.rank > 5 ? srcB[2] : this.rank > 5 ? srcA[2] : srcB[2];
            dst[3] = !(rank > 4) || (srcA[3] == 1 && other.rank > 4) ? srcB[3] : srcA[3]; // srcA[3] == 1 && shape.rank > 4 ? srcB[3] : this.rank > 4 ? srcA[3] : srcB[3];
            dst[4] = !(rank > 3) || (srcA[4] == 1 && other.rank > 3) ? srcB[4] : srcA[4]; // srcA[4] == 1 && shape.rank > 3 ? srcB[4] : this.rank > 3 ? srcA[4] : srcB[4];
            dst[5] = !(rank > 2) || (srcA[5] == 1 && other.rank > 2) ? srcB[5] : srcA[5]; // srcA[5] == 1 && shape.rank > 2 ? srcB[5] : this.rank > 2 ? srcA[5] : srcB[5];
            dst[6] = !(rank > 1) || (srcA[6] == 1 && other.rank > 1) ? srcB[6] : srcA[6]; // srcA[6] == 1 && shape.rank > 1 ? srcB[6] : this.rank > 1 ? srcA[6] : srcB[6];
            dst[7] = !(rank > 0) || (srcA[7] == 1 && other.rank > 0) ? srcB[7] : srcA[7]; // srcA[7] == 1 && shape.rank > 0 ? srcB[7] : this.rank > 0 ? srcA[7] : srcB[7];

            Logger.AssertIsTrue((srcA[0] == srcB[0]) || (srcA[0] == 1) || (srcB[0] == 1) || (srcA[0] == 0 && srcB[0] != 0) || (srcB[0] == 0 && srcA[0] != 0), "TensorShape.ValueError: operands could not be broadcast together with shapes {0}, {1}", this, other);
            Logger.AssertIsTrue((srcA[1] == srcB[1]) || (srcA[1] == 1) || (srcB[1] == 1) || (srcA[1] == 0 && srcB[1] != 0) || (srcB[1] == 0 && srcA[1] != 0), "TensorShape.ValueError: operands could not be broadcast together with shapes {0}, {1}", this, other);
            Logger.AssertIsTrue((srcA[2] == srcB[2]) || (srcA[2] == 1) || (srcB[2] == 1) || (srcA[2] == 0 && srcB[2] != 0) || (srcB[2] == 0 && srcA[2] != 0), "TensorShape.ValueError: operands could not be broadcast together with shapes {0}, {1}", this, other);
            Logger.AssertIsTrue((srcA[3] == srcB[3]) || (srcA[3] == 1) || (srcB[3] == 1) || (srcA[3] == 0 && srcB[3] != 0) || (srcB[3] == 0 && srcA[3] != 0), "TensorShape.ValueError: operands could not be broadcast together with shapes {0}, {1}", this, other);
            Logger.AssertIsTrue((srcA[4] == srcB[4]) || (srcA[4] == 1) || (srcB[4] == 1) || (srcA[4] == 0 && srcB[4] != 0) || (srcB[4] == 0 && srcA[4] != 0), "TensorShape.ValueError: operands could not be broadcast together with shapes {0}, {1}", this, other);
            Logger.AssertIsTrue((srcA[5] == srcB[5]) || (srcA[5] == 1) || (srcB[5] == 1) || (srcA[5] == 0 && srcB[5] != 0) || (srcB[5] == 0 && srcA[5] != 0), "TensorShape.ValueError: operands could not be broadcast together with shapes {0}, {1}", this, other);
            Logger.AssertIsTrue((srcA[6] == srcB[6]) || (srcA[6] == 1) || (srcB[6] == 1) || (srcA[6] == 0 && srcB[6] != 0) || (srcB[6] == 0 && srcA[6] != 0), "TensorShape.ValueError: operands could not be broadcast together with shapes {0}, {1}", this, other);
            Logger.AssertIsTrue((srcA[7] == srcB[7]) || (srcA[7] == 1) || (srcB[7] == 1) || (srcA[7] == 0 && srcB[7] != 0) || (srcB[7] == 0 && srcA[7] != 0), "TensorShape.ValueError: operands could not be broadcast together with shapes {0}, {1}", this, other);
        }

        broadcast.RecomputeLength();

        return broadcast;
    }

    /// <summary>
    /// Creates a `TensorShape` with a given rank where all of the dimensions are 1. For example if `rank` is 3, the method returns (1, 1, 1).
    /// </summary>
    /// <param name="rank">The rank of the tensor shape.</param>
    /// <returns>The created tensor shape.</returns>
    public static TensorShape Ones(int rank)
    {
        Logger.AssertIsTrue(rank <= maxRank, "ValueError: TensorShape are capped to rank=8, cannot create empty shape of rank {0}", rank);
        var empty = new TensorShape();
        empty.m_Rank = rank;
        empty.m_Length = 1;

        int* dst = &empty.m_D7;
        for (int i = 0; i < rank; i++)
        {
            dst[(maxRank - rank) + i] = 1;
        }

        return empty;
    }

    /// <summary>
    /// Creates a `TensorShape` of a given rank and with the same inner dimensions by adding outer dimensions of size 1 when needed.
    ///
    /// For example, if the `TensorShape` is (256, 256, 3), and the value of `rank` is 5, the method returns (1, 1, 256, 256, 3).
    /// </summary>
    /// <param name="rank">The rank to which to broadcast to.</param>
    /// <returns>The broadcast tensor shape.</returns>
    public TensorShape BroadcastToRank(int rank)
    {
        Logger.AssertIsTrue(rank >= this.rank, "ValueError: broadcasting to lower rank tensor {0}, {1}", this.rank, rank);
        Logger.AssertIsTrue(rank <= maxRank, "ValueError: TensorShape are capped to rank=8, cannot broadcast shape {0} to rank > 8", this);

        var broadcast = new TensorShape();
        broadcast.m_Rank = rank;
        broadcast.m_Length = length;

        int* dst = &broadcast.m_D7;

        for (int i = 0; i < (rank - this.rank); i++)
        {
            dst[(maxRank - rank) + i] = 1;
        }

        fixed (int* src = &m_D7)
        {
            for (int i = 0; i < this.rank; i++)
            {
                dst[(maxRank - rank) + (i + (rank - this.rank))] = src[(maxRank - this.rank) + i];
            }
        }

        return broadcast;
    }

    /// <summary>
    /// Creates a `TensorShape` by repeating this `TensorShape` a number of times along each axis.
    /// </summary>
    /// <param name="repeats">The repeat counts along each axis as a span of integers.</param>
    /// <returns>The tiled tensor shape.</returns>
    public TensorShape Tile(ReadOnlySpan<int> repeats)
    {
        TensorShape mul = new TensorShape();
        mul.m_Rank = Math.Max(rank, repeats.Length);

        int* dst = &mul.m_D7;
        fixed (int* srcA = &m_D7)
        {
            for (int i = 0; i < mul.m_Rank; i++)
            {
                dst[(maxRank - 1) - i] = (i < rank ? srcA[(maxRank - 1) - i] : 1) * (i < repeats.Length ? repeats[(repeats.Length - 1) - i] : 1);
            }
        }

        mul.RecomputeLength();

        return mul;
    }

    /// <summary>
    /// Creates a `TensorShape` by concatenating this `TensorShape` with `other` along a given axis. The dimensions of the shapes must be equal, except at `axis`.
    ///
    /// For example if `this` is (2, 3, 4, 5), `other` is (2, 2, 4, 5), and the value of `axis` is 1, the method returns (2, 5, 4, 5).
    /// </summary>
    /// <param name="other">The other tensor shape which which to concatenate.</param>
    /// <param name="axis">The axis along which to concatenate.</param>
    /// <returns>The concatenated tensor shape.</returns>
    public TensorShape Concat(TensorShape other, int axis)
    {
        TensorShape concat = new TensorShape(this);
        axis = other.Axis(axis);

        int* src = &other.m_D7;
        int* dst = &concat.m_D7;
        dst[(maxRank - rank) + axis] += src[(maxRank - rank) + axis];

        #if (UNITY_ASSERTIONS)
        for (int r = 0; r < Math.Max(rank, other.rank); r++)
        {
            if (r == axis)
                continue;
            if (src[(maxRank - rank) + r] == 0 || dst[(maxRank - rank) + r] == 0)
                continue;
            Logger.AssertAreEqual(src[(maxRank - rank) + r], dst[(maxRank - rank) + r], "ValueError: all input shapes for Concat must be equal {0}, {1} except on axis ({2}) dim", this, other, axis);
        }
        #endif

        concat.RecomputeLength();

        return concat;
    }

    /// <summary>
    /// Removes a dimension at `axis`. For example, if `this` is (2, 3, 4, 5), the value of `axis` is 1, and the value of `keepDim` is `true`, the method returns (2, 1, 4, 5).
    /// </summary>
    /// <param name="axis">The axis along which to reduce.</param>
    /// <param name="keepDim">When the value is `true`, Sentis replaces the dimension with 1.</param>
    /// <returns>The reduced tensor shape.</returns>
    public TensorShape Reduce(int axis, bool keepDim = true)
    {
        TensorShape reducedShape = new TensorShape(this);
        if (rank == 0)
        {
            if (keepDim == false)
                Logger.AssertIsTrue(rank >= 1, "ValueError: cannot squeeze scalar tensor {0}", this);
            return reducedShape;
        }

        if (this[axis] == 0)
        {
            Logger.AssertIsTrue(keepDim, "ValueError: cannot reduce on dim {0} with value of 0 if 'keepDim' is false", axis);
            reducedShape[axis] = 0;
            return reducedShape;
        }

        reducedShape[axis] = 1;

        if (!keepDim)
            reducedShape = reducedShape.Squeeze(axis);

        return reducedShape;
    }

    /// <summary>
    /// Creates a `TensorShape` by removing the dimensions at `axes`. For example, if `this` is (2, 3, 4, 5), `axis` is {1, 2} and the value of `keepDim` is `true`, the method returns (2, 1, 1, 5).
    /// </summary>
    /// <param name="axes">The axes along which to reduce.</param>
    /// <param name="keepDim">When the value is `true`, Sentis replaces the reduced axes with 1. Otherwise Sentis removes the reduced axes.</param>
    /// <returns>The reduced tensor shape.</returns>
    public TensorShape Reduce(ReadOnlySpan<int> axes, bool keepDim = true)
    {
        if (axes == null || axes.Length == 0)
            return keepDim ? Ones(rank) : new TensorShape();

        TensorShape reducedShape = new TensorShape(this);

        for (int i = 0; i < axes.Length; i++)
        {
            int axis = Axis(axes[i]);
            if (this[axis] == 0)
            {
                Logger.AssertIsTrue(keepDim, "ValueError: cannot reduce on dim {0} with value of 0 if 'keepDim' is false", axis);
                reducedShape[axis] = 0;
                continue;
            }

            reducedShape[axis] = 1;
        }

        if (!keepDim)
            reducedShape = reducedShape.Squeeze(axes);

        return reducedShape;
    }

    /// <summary>
    /// Creates a `TensorShape` by permuting axes. For example, if `this` is (6, 7, 8, 9), and `permutations` is {3, 0, 1, 2}, the method returns (9, 6, 7, 8).
    /// </summary>
    /// <param name="permutations">An array indexing the new tensor axis from the old ones.</param>
    /// <returns>The transposed tensor shape.</returns>
    public TensorShape Transpose(int[] permutations)
    {
        Logger.AssertAreEqual(rank, permutations.Length, "ValueError: shape ranks and permutations length do not match {0}, {1}", this, permutations.Length);

        TensorShape transposed = new TensorShape();
        transposed.m_Rank = rank;
        transposed.m_Length = length;

        int* dst = &transposed.m_D7;
        int newLength = 1;
        fixed (int* src = &m_D7)
        {
            for (var i = 0; i < permutations.Length; ++i)
            {
                Logger.AssertIsTrue(permutations[i] >= 0 && permutations[i] < rank, "ValueError: permutation index out of range");
                dst[(maxRank - rank) + i] = src[(maxRank - rank) + permutations[i]];
                newLength *= dst[(maxRank - rank) + i];
            }
        }

        Logger.AssertIsTrue(newLength == length, "ValueError: incorrect permutations");

        return transposed;
    }

    /// <summary>
    /// Creates a `TensorShape` by reversing axes. For example, if `this` is (6, 7, 8, 9), the method returns (9, 8, 7, 6).
    /// </summary>
    /// <returns>The transposed tensor shape.</returns>
    public TensorShape Transpose()
    {
        TensorShape transposed = new TensorShape();
        transposed.m_Rank = rank;
        transposed.m_Length = length;

        int* dst = &transposed.m_D7;
        fixed (int* src = &m_D7)
        {
            for (var i = 0; i < rank; ++i)
                dst[(maxRank - rank) + i] = src[(maxRank - rank) + ((rank - 1) - i)];
        }

        return transposed;
    }

    /// <summary>
    /// Creates a `TensorShape` by padding axes. For example, if `this` is (1, 2, 3), and `pads` is {0, 0, 1, 0, 2, 2}, the method returns (1, 3, 7).
    /// </summary>
    /// <param name="pads">The lower and upper padding values for each dimension. For example [pad_left, pad_right] for 1D, or [pad_top, pad_bottom, pad_left, pad_right] for 2D.</param>
    /// <returns>The padded tensor shape.</returns>
    public TensorShape Pad(ReadOnlySpan<int> pads)
    {
        Logger.AssertAreEqual(rank * 2, pads.Length, "ValueError: shape ranks and pad length do not match {0}, {1}", this, pads.Length);

        TensorShape padded = new TensorShape(this);

        int* dst = &padded.m_D7;
        for (int i = 0; i < padded.rank; i++)
            dst[(maxRank - rank) + i] += pads[i] + pads[i + padded.rank];

        padded.RecomputeLength();

        return padded;
    }

    /// <summary>
    /// Creates a `TensorShape` that results from performing a matrix multiplication between `this` and `other` with numpy-style broadcasting. For example, if `this` is (5, 2, 3), and `other` is (1, 3, 4), the method returns (5, 2, 4).
    /// </summary>
    /// <param name="other">The right hand tensor shape for the MatMul.</param>
    /// <returns>The resultant tensor shape.</returns>
    public TensorShape MatMul(TensorShape other)
    {
        if (other.rank == 1)
            return MatMul(new TensorShape(other[0], 1)).Squeeze(-1);
        if (rank == 1)
            return new TensorShape(1, this[0]).MatMul(other).Squeeze(-2);

        Logger.AssertIsTrue(this[-1] == other[-2], "ValueError: dimension does not match for matmul {0}, {1}", this, other);

        TensorShape matmul = new TensorShape();
        matmul.m_Rank = Math.Max(rank, other.rank);

        int* srcB = &other.m_D7;
        int* dst = &matmul.m_D7;
        fixed (int* srcA = &m_D7)
        {
            dst[0] = !(rank > 7) || (srcA[0] == 1 && other.rank > 7) ? srcB[0] : srcA[0]; // srcA[0] == 1 && shape.rank > 7 ? srcB[0] : this.rank > 7 ? srcA[0] : srcB[0];
            dst[1] = !(rank > 6) || (srcA[1] == 1 && other.rank > 6) ? srcB[1] : srcA[1]; // srcA[1] == 1 && shape.rank > 6 ? srcB[1] : this.rank > 6 ? srcA[1] : srcB[1];
            dst[2] = !(rank > 5) || (srcA[2] == 1 && other.rank > 5) ? srcB[2] : srcA[2]; // srcA[2] == 1 && shape.rank > 5 ? srcB[2] : this.rank > 5 ? srcA[2] : srcB[2];
            dst[3] = !(rank > 4) || (srcA[3] == 1 && other.rank > 4) ? srcB[3] : srcA[3]; // srcA[3] == 1 && shape.rank > 4 ? srcB[3] : this.rank > 4 ? srcA[3] : srcB[3];
            dst[4] = !(rank > 3) || (srcA[4] == 1 && other.rank > 3) ? srcB[4] : srcA[4]; // srcA[4] == 1 && shape.rank > 3 ? srcB[4] : this.rank > 3 ? srcA[4] : srcB[4];
            dst[5] = !(rank > 2) || (srcA[5] == 1 && other.rank > 2) ? srcB[5] : srcA[5]; // srcA[5] == 1 && shape.rank > 2 ? srcB[5] : this.rank > 2 ? srcA[5] : srcB[5];
            dst[6] = srcA[6];
            dst[7] = srcB[7];

            Logger.AssertIsTrue((srcA[0] == srcB[0]) || (srcA[0] == 1) || (srcB[0] == 1) || (srcA[0] == 0 && srcB[0] != 0) || (srcB[0] == 0 && srcA[0] != 0), "TensorShape.ValueError: operands could not be broadcast together with shapes {0}, {1}", this, other);
            Logger.AssertIsTrue((srcA[1] == srcB[1]) || (srcA[1] == 1) || (srcB[1] == 1) || (srcA[1] == 0 && srcB[1] != 0) || (srcB[1] == 0 && srcA[1] != 0), "TensorShape.ValueError: operands could not be broadcast together with shapes {0}, {1}", this, other);
            Logger.AssertIsTrue((srcA[2] == srcB[2]) || (srcA[2] == 1) || (srcB[2] == 1) || (srcA[2] == 0 && srcB[2] != 0) || (srcB[2] == 0 && srcA[2] != 0), "TensorShape.ValueError: operands could not be broadcast together with shapes {0}, {1}", this, other);
            Logger.AssertIsTrue((srcA[3] == srcB[3]) || (srcA[3] == 1) || (srcB[3] == 1) || (srcA[3] == 0 && srcB[3] != 0) || (srcB[3] == 0 && srcA[3] != 0), "TensorShape.ValueError: operands could not be broadcast together with shapes {0}, {1}", this, other);
            Logger.AssertIsTrue((srcA[4] == srcB[4]) || (srcA[4] == 1) || (srcB[4] == 1) || (srcA[4] == 0 && srcB[4] != 0) || (srcB[4] == 0 && srcA[4] != 0), "TensorShape.ValueError: operands could not be broadcast together with shapes {0}, {1}", this, other);
            Logger.AssertIsTrue((srcA[5] == srcB[5]) || (srcA[5] == 1) || (srcB[5] == 1) || (srcA[5] == 0 && srcB[5] != 0) || (srcB[5] == 0 && srcA[5] != 0), "TensorShape.ValueError: operands could not be broadcast together with shapes {0}, {1}", this, other);
        }

        matmul.RecomputeLength();

        return matmul;
    }

    /// <summary>
    /// Creates a `TensorShape` that results from slicing `this` along given axes with given starts, ends, and steps.
    /// </summary>
    /// <param name="starts">The start indices along each of the `axes`.</param>
    /// <param name="ends">The end indices along each of the `axes`.</param>
    /// <param name="axes">The optional axes along which to slice. The default value is [0, 1, 2...rank-1].</param>
    /// <param name="steps">The optional step sizes for each of the `axes`. The default value is [1, 1, 1...1]</param>
    /// <returns>The sliced tensor shape.</returns>
    public TensorShape Slice(ReadOnlySpan<int> starts, ReadOnlySpan<int> ends, ReadOnlySpan<int> axes, ReadOnlySpan<int> steps)
    {
        Logger.AssertAreEqual(starts.Length, ends.Length, "ValueError: starts and ends length do not match {0}, {1}", starts.Length, ends.Length);
        if (axes != null)
            Logger.AssertAreEqual(starts.Length, axes.Length, "ValueError: starts and axes length do not match {0}, {1}", starts.Length, axes.Length);
        if (steps != null)
            Logger.AssertAreEqual(starts.Length, steps.Length, "ValueError: starts and steps length do not match {0}, {1}", starts.Length, steps.Length);

        TensorShape strided = new TensorShape(this);

        int* dst = &strided.m_D7;

        for (int i = 0; i < starts.Length; i++)
        {
            int axis = axes != null ? Axis(axes[i]) : i;
            int step = steps != null ? steps[i] : 1;

            dst[(maxRank - rank) + axis] = ShapeInference.SliceDim(dst[(maxRank - rank) + axis], starts[i], ends[i], step);
        }

        strided.RecomputeLength();

        return strided;
    }

    internal TensorShape Split(int axis, int start, int end)
    {
        Assert.IsTrue(0 <= start && start <= end && end <= this[axis], "Split.InputError: start and end must obey 0 <= start <= end <= dim");

        var strided = new TensorShape(this);

        axis = Axis(axis);

        var dst = &strided.m_D7;
        dst[(maxRank - rank) + axis] = end - start;

        strided.RecomputeLength();
        return strided;
    }

    /// <summary>
    /// Compares two `TensorShape` objects. Returns `true` if the two objects have the same rank, and all their dimensions are equal.
    /// </summary>
    /// <param name="a">The first shape to compare.</param>
    /// <param name="b">The second shape to compare.</param>
    /// <returns>Whether the two shapes are equal.</returns>
    public static bool operator ==(TensorShape a, TensorShape b)
    {
        if (a.rank != b.rank)
            return false;

        if (a.length != b.length)
            return false;

        for (var i = 0; i < a.rank; ++i)
        {
            if (a[i] != b[i])
                return false;
        }

        return true;
    }

    /// <summary>
    /// Compares two `TensorShape` objects.
    /// </summary>
    /// <param name="a">The first shape to compare.</param>
    /// <param name="b">The second shape to compare.</param>
    /// <returns>Whether the two shapes are not equal.</returns>
    public static bool operator !=(TensorShape a, TensorShape b)
    {
        return !(a == b);
    }

    /// <summary>
    /// Determines whether the specified object is equal to the current `TensorShape`.
    /// </summary>
    /// <param name="obj">The object to compare to the shape.</param>
    /// <returns>Whether the object is equal to the shape.</returns>
    public override bool Equals(object obj)
    {
        // Check for null values and compare run-time types.
        if (obj == null || GetType() != obj.GetType())
            return false;

        return this == (TensorShape)obj;
    }

    /// <summary>
    /// Serves as the default hash function.
    /// </summary>
    /// <returns>The hash code of the tensor shape.</returns>
    public override int GetHashCode()
    {
        return m_Rank ^ m_D7 ^ m_D6 ^ m_D5 ^ m_D4 ^ m_D3 ^ m_D2 ^ m_D1 ^ m_D0;
    }

    /// <summary>
    /// Converts the indexes of the `TensorShape` into a flat index.
    ///
    /// shape: (5,1,2,3,4,5,6,7) =&gt; 2 * (1*2*3*4*5*6*7) + 0 * (2*3*4*5*6*7) + 1 * (3*4*5*6*7) + 0 * (4*5*6*7) + 3 * (5*6*7) + 2 * (6*7) + 1 * (7) + 5 = 13326
    ///
    /// index: (2,0,1,0,3,2,1,5)
    /// </summary>
    /// <param name="d7">The index along axis 7.</param>
    /// <param name="d6">The index along axis 6.</param>
    /// <param name="d5">The index along axis 5.</param>
    /// <param name="d4">The index along axis 4.</param>
    /// <param name="d3">The index along axis 3.</param>
    /// <param name="d2">The index along axis 2.</param>
    /// <param name="d1">The index along axis 1.</param>
    /// <param name="d0">The index along axis 0.</param>
    /// <returns>The raveled index.</returns>
    public int RavelIndex(int d7, int d6, int d5, int d4, int d3, int d2, int d1, int d0)
    {
        return (((((((d7 * m_D6 + d6) * m_D5 + d5) * m_D4 + d4) * m_D3 + d3) * m_D2 + d2) * m_D1 + d1) * m_D0) + d0;
    }

    /// <summary>
    /// Converts the indexes of the `TensorShape` into a flat index.
    ///
    /// shape: (1,2,3,4,5,6,7) => 0 * (2*3*4*5*6*7) + 1 * (3*4*5*6*7) + 0 * (4*5*6*7) + 3 * (5*6*7) + 2 * (6*7) + 1 * (7) + 5 = 3246
    ///
    /// index: (0,1,0,3,2,1,5)
    /// </summary>
    /// <param name="d6">The index along axis 6.</param>
    /// <param name="d5">The index along axis 5.</param>
    /// <param name="d4">The index along axis 4.</param>
    /// <param name="d3">The index along axis 3.</param>
    /// <param name="d2">The index along axis 2.</param>
    /// <param name="d1">The index along axis 1.</param>
    /// <param name="d0">The index along axis 0.</param>
    /// <returns>The raveled index.</returns>
    public int RavelIndex(int d6, int d5, int d4, int d3, int d2, int d1, int d0)
    {
        return ((((((d6 * m_D5 + d5) * m_D4 + d4) * m_D3 + d3) * m_D2 + d2) * m_D1 + d1) * m_D0) + d0;
    }

    /// <summary>
    /// Converts the indexes of the `TensorShape` into a flat index.
    ///
    /// shape: (2,3,4,5,6,7) => 1 * (3*4*5*6*7) + 0 * (4*5*6*7) + 3 * (5*6*7) + 2 * (6*7) + 1 * (7) + 5 = 3246
    ///
    /// index: (1,0,3,2,1,5)
    /// </summary>
    /// <param name="d5">The index along axis 5.</param>
    /// <param name="d4">The index along axis 4.</param>
    /// <param name="d3">The index along axis 3.</param>
    /// <param name="d2">The index along axis 2.</param>
    /// <param name="d1">The index along axis 1.</param>
    /// <param name="d0">The index along axis 0.</param>
    /// <returns>The raveled index.</returns>
    public int RavelIndex(int d5, int d4, int d3, int d2, int d1, int d0)
    {
        return ((((d5 * m_D4 + d4) * m_D3 + d3) * m_D2 + d2) * m_D1 + d1) * m_D0 + d0;
    }

    /// <summary>
    /// Converts the indexes of the `TensorShape` into a flat index.
    ///
    /// shape: (3,4,5,6,7) => 0 * (4*5*6*7) + 3 * (5*6*7) + 2 * (6*7) + 1 * (7) + 5 = 726
    ///
    /// index: (0,3,2,1,5)
    /// </summary>
    /// <param name="d4">The index along axis 4.</param>
    /// <param name="d3">The index along axis 3.</param>
    /// <param name="d2">The index along axis 2.</param>
    /// <param name="d1">The index along axis 1.</param>
    /// <param name="d0">The index along axis 0.</param>
    /// <returns>The raveled index.</returns>
    public int RavelIndex(int d4, int d3, int d2, int d1, int d0)
    {
        return (((d4 * m_D3 + d3) * m_D2 + d2) * m_D1 + d1) * m_D0 + d0;
    }

    /// <summary>
    /// Converts the indexes of the `TensorShape` into a flat index.
    ///
    /// shape: (4,5,6,7) => 3 * (5*6*7) + 2 * (6*7) + 1 * (7) + 5 = 726
    ///
    /// index: (3,2,1,5)
    /// </summary>
    /// <param name="d3">The index along axis 3.</param>
    /// <param name="d2">The index along axis 2.</param>
    /// <param name="d1">The index along axis 1.</param>
    /// <param name="d0">The index along axis 0.</param>
    /// <returns>The raveled index.</returns>
    public int RavelIndex(int d3, int d2, int d1, int d0)
    {
        return ((d3 * m_D2 + d2) * m_D1 + d1) * m_D0 + d0;
    }

    /// <summary>
    /// Converts the indexes of the `TensorShape` into a flat index.
    ///
    /// shape: (5,6,7) => 2 * (6*7) + 1 * (7) + 5 = 96
    ///
    /// index: (2,1,5)
    /// </summary>
    /// <param name="d2">The index along axis 2.</param>
    /// <param name="d1">The index along axis 1.</param>
    /// <param name="d0">The index along axis 0.</param>
    /// <returns>The raveled index.</returns>
    public int RavelIndex(int d2, int d1, int d0)
    {
        return (d2 * m_D1 + d1) * m_D0 + d0;
    }

    /// <summary>
    /// Converts the indexes of the `TensorShape` into a flat index.
    ///
    /// shape: (6,7) => 1 * (7) + 5 = 12
    ///
    /// index: (1,5)
    /// </summary>
    /// <param name="d1">The index along axis 1.</param>
    /// <param name="d0">The index along axis 0.</param>
    /// <returns>The raveled index.</returns>
    public int RavelIndex(int d1, int d0)
    {
        return d1 * m_D0 + d0;
    }
}
} // namespace Unity.Sentis

