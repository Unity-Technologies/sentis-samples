using System;
using System.Runtime.CompilerServices;
using System.Text;

[assembly: InternalsVisibleTo("Unity.Sentis.Tests")]
[assembly: InternalsVisibleTo("Unity.Sentis.EditorTests")]
[assembly: InternalsVisibleTo("Unity.Sentis.CPUBackend")]

namespace Unity.Sentis {

/// <summary>
/// Represents a set of indices corresponding to each axis of a tensor
/// </summary>
[Serializable]
public unsafe struct TensorIndex
{
    /// <summary>
    /// TensorIndex cannot have a bigger rank than maxRank
    /// </summary>
    public const int maxRank = 8;

#pragma warning disable CS0649

    // disable warning: Field 'field' is never assigned to, and will always have its default value 'value'
    // fields are accessed using unsafe code (ptr)
    // TensorIndices are 0 padded from right to left to match TensorDim slices
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
    /// Rank of a TensorIndex:
    /// Ex:
    /// (5)       -> rank:1
    /// (3,5)     -> rank:2
    /// (7,3,5)   -> rank:3
    /// (6,7,3,5) -> rank:4
    /// </summary>
    public int rank => m_Rank;

    /// <summary>
    /// Creates a rank-8 tensor index (d7, d6, d5, d4, d3, d2, d1, d0)
    /// Ex: (2,3,4,5,6,7,8,9)
    /// </summary>
    /// <param name="d7">Axis 7.</param>
    /// <param name="d6">Axis 6.</param>
    /// <param name="d5">Axis 5.</param>
    /// <param name="d4">Axis 4.</param>
    /// <param name="d3">Axis 3.</param>
    /// <param name="d2">Axis 2.</param>
    /// <param name="d1">Axis 1.</param>
    /// <param name="d0">Axis 0.</param>
    public TensorIndex(int d7, int d6, int d5, int d4, int d3, int d2, int d1, int d0)
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
    }

    /// <summary>
    /// Creates a rank-7 tensor index (d6, d5, d4, d3, d2, d1, d0)
    /// Ex: (3,4,5,6,7,8,9)
    /// </summary>
    /// <param name="d6">Axis 6.</param>
    /// <param name="d5">Axis 5.</param>
    /// <param name="d4">Axis 4.</param>
    /// <param name="d3">Axis 3.</param>
    /// <param name="d2">Axis 2.</param>
    /// <param name="d1">Axis 1.</param>
    /// <param name="d0">Axis 0.</param>
    public TensorIndex(int d6, int d5, int d4, int d3, int d2, int d1, int d0)
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
    }

    /// <summary>
    /// Creates a rank-6 tensor index (d5, d4, d3, d2, d1, d0)
    /// Ex: (4,5,6,7,8,9)
    /// </summary>
    /// <param name="d5">Axis 5.</param>
    /// <param name="d4">Axis 4.</param>
    /// <param name="d3">Axis 3.</param>
    /// <param name="d2">Axis 2.</param>
    /// <param name="d1">Axis 1.</param>
    /// <param name="d0">Axis 0.</param>
    public TensorIndex(int d5, int d4, int d3, int d2, int d1, int d0)
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
    }

    /// <summary>
    /// Creates a rank-5 tensor index (d4, d3, d2, d1, d0)
    /// Ex: (5,6,7,8,9)
    /// </summary>
    /// <param name="d4">Axis 4.</param>
    /// <param name="d3">Axis 3.</param>
    /// <param name="d2">Axis 2.</param>
    /// <param name="d1">Axis 1.</param>
    /// <param name="d0">Axis 0.</param>
    public TensorIndex(int d4, int d3, int d2, int d1, int d0)
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
    }

    /// <summary>
    /// Creates a rank-4 tensor index (d3, d2, d1, d0)
    /// Ex: (6,7,8,9)
    /// </summary>
    /// <param name="d3">Axis 3.</param>
    /// <param name="d2">Axis 2.</param>
    /// <param name="d1">Axis 1.</param>
    /// <param name="d0">Axis 0.</param>
    public TensorIndex(int d3, int d2, int d1, int d0)
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
    }

    /// <summary>
    /// Creates a rank-3 tensor index (d2, d1, d0)
    /// Ex: (7,8,9)
    /// </summary>
    /// <param name="d2">Axis 2.</param>
    /// <param name="d1">Axis 1.</param>
    /// <param name="d0">Axis 0.</param>
    public TensorIndex(int d2, int d1, int d0)
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
    }
    /// <summary>
    /// Creates a rank-2 tensor index (d1, d0)
    /// Ex: (8,9)
    /// </summary>
    /// <param name="d1">Axis 1.</param>
    /// <param name="d0">Axis 0.</param>
    public TensorIndex(int d1, int d0)
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
    }
    /// <summary>
    /// Creates a rank-1 tensor index (d0)
    /// Ex: (9)
    /// </summary>
    /// <param name="d0">Axis 0.</param>
    public TensorIndex(int d0)
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
    }

    /// <summary>
    /// Instantiates and returns a copy of another `TensorIndex`.
    /// </summary>
    /// <param name="index">The `TensorIndex` to copy.</param>
    public TensorIndex(TensorIndex index)
    {
        m_Rank = index.rank;

        m_D7 = index.m_D7;
        m_D6 = index.m_D6;
        m_D5 = index.m_D5;
        m_D4 = index.m_D4;
        m_D3 = index.m_D3;
        m_D2 = index.m_D2;
        m_D1 = index.m_D1;
        m_D0 = index.m_D0;
    }

    /// <summary>
    /// Creates a tensor index given an input int[] representing the index
    /// Ex: TensorIndex(new [] {3,4,5,6}) = (3,4,5,6)
    /// </summary>
    /// <param name="index">The index as an array.</param>
    public TensorIndex(int[] index)
        : this()
    {
        Logger.AssertIsTrue(index.Length <= maxRank, "ValueError: TensorIndex are capped to rank=8, cannot create tensorindex of rank {0}", index.Length);

        m_Rank = index.Length;

        if (index.Length == 0)
            return;

        fixed (int* dst = &m_D7, src = &index[0])
        {
            Buffer.MemoryCopy(src, dst + (maxRank - rank), index.Length * sizeof(int), index.Length * sizeof(int));
        }
    }

    /// <summary>
    /// Creates index with zeros 0 of specified rank
    /// Ex:
    /// EmptyOfRank(rank: 3) => (0, 0, 0)
    /// </summary>
    /// <param name="rank">The rank of the tensor index.</param>
    /// <returns>The created tensor index.</returns>
    public static TensorIndex Zeros(int rank)
    {
        Logger.AssertIsTrue(rank <= maxRank, "ValueError: TensorIndex are capped to rank=8, cannot create empty index of rank {0}", rank);
        var empty = new TensorIndex();
        empty.m_Rank = rank;

        int* dst = &empty.m_D7;
        for (int i = 0; i < rank; i++)
        {
            dst[(maxRank - rank) + i] = 1;
        }

        return empty;
    }

    /// <summary>
    /// Accesses internal index layout:
    /// index (3,4,5,6), rank 4,  (internally) = ( 0, 0, 0, 0, 3, 4, 5, 6), rank 8
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
        fixed (int* index = &m_D7)
        {
            return index[axis];
        }
    }

    /// <summary>
    /// Gets/Sets tensor index at a given axis
    /// Ex:
    /// index  (3, 4, 5, 6)
    /// index   0, 1, 2, 3
    ///        -4,-3,-2,-1
    /// index  (7, 3, 2)
    /// index   0, 1, 2
    ///        -3,-2,-1
    /// </summary>
    /// <param name="axis">The axis to get the index at.</param>
    public int this[int axis]
    {
        get
        {
            axis = Axis(axis);

            fixed (int* index = &m_D7)
            {
                return index[(maxRank - rank) + axis];
            }
        }

        set
        {
            axis = Axis(axis);

            fixed (int* index = &m_D7)
            {
                index[(maxRank - rank) + axis] = value;
            }
        }
    }

    /// <summary>
    /// Returns a string that represents the `TensorIndex`.
    /// </summary>
    /// <returns>The string representation of the tensor index.</returns>
    public override string ToString()
    {
        StringBuilder sb = new StringBuilder();
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
    /// Wraps axis to positive index between 0,rank
    /// (5,2,3,4)
    /// axis = -1 =&gt; axis_out = 3
    /// axis = 1 =&gt; axis_out = 1
    /// </summary>
    /// <param name="axis">The axis to wrap.</param>
    /// <returns>The wrapped axis.</returns>
    public int Axis(int axis)
    {
        Logger.AssertIsTrue(axis >= -rank && axis < rank, "IndexError: axis {0} is out of bounds index of rank, {1}", axis, rank);
        return axis >= 0 ? axis : rank + axis;
    }

    /// <summary>
    /// Compares two `TensorIndex` objects
    /// Two TensorIndices are equal if they have the same rank and all their dimensions are equal
    /// </summary>
    /// <param name="a">The first tensor index to compare.</param>
    /// <param name="b">The second tensor index to compare.</param>
    /// <returns>Whether the tensor indexes are equal.</returns>
    public static bool operator ==(TensorIndex a, TensorIndex b)
    {
        if (a.rank != b.rank)
            return false;

        for (var i = 0; i < a.rank; ++i)
        {
            if (a[i] != b[i])
                return false;
        }

        return true;
    }

    /// <summary>
    /// Compares two `TensorIndex` objects
    /// </summary>
    /// <param name="a">The first tensor index to compare.</param>
    /// <param name="b">The second tensor index to compare.</param>
    /// <returns>Whether the tensor indexes are not equal.</returns>
    public static bool operator !=(TensorIndex a, TensorIndex b)
    {
        return !(a == b);
    }

    /// <summary>
    /// Determines whether the specified object is equal to the current `TensorIndex`.
    /// </summary>
    /// <param name="obj">The object to compare to the tensor index.</param>
    /// <returns>Whether the object is equal to the tensor index.</returns>
    public override bool Equals(object obj)
    {
        // Check for null values and compare run-time types.
        if (obj == null || GetType() != obj.GetType())
            return false;

        return this == (TensorIndex)obj;
    }

    /// <summary>
    /// Serves as the default hash function.
    /// </summary>
    /// <returns>The hashed tensor index.</returns>
    public override int GetHashCode()
    {
        return rank ^ m_D7 ^ m_D6 ^ m_D5 ^ m_D4 ^ m_D3 ^ m_D2 ^ m_D1 ^ m_D0;
    }
}
} // namespace Unity.Sentis

