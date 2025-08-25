using System;

namespace Unity.ML.Tokenization
{
    /// <summary>
    ///     Integer range structure.
    /// </summary>
    public readonly struct Range : IComparable<Range>, IEquatable<Range>
    {
        /// <summary>
        ///     Creates a <see cref="Range" /> instance specifying the bounds of it.
        /// </summary>
        /// <param name="from">
        ///     The inclusive lower bound of the range.
        /// </param>
        /// <param name="to">
        ///     The exclusive upper bound of the range.
        /// </param>
        /// <returns>
        ///     The range instance.
        /// </returns>
        public static Range FromTo(int from, int to)
        {
            return new Range(from, to - from);
        }

        /// <summary>
        ///     The inclusive lower bound of the range.
        /// </summary>
        public readonly int Offset;

        /// <summary>
        ///     The length of the range.
        /// </summary>
        public readonly int Length;

        /// <summary>
        ///     Initializes a new instance of the <see cref="Range" /> type.
        /// </summary>
        /// <param name="offset">
        ///     The inclusive lower bound of the range.
        /// </param>
        /// <param name="length">
        ///     The length of the range.
        /// </param>
        /// <exception cref="ArgumentOutOfRangeException">
        ///     <paramref name="length" /> cannot be lower than <c>0</c>.
        /// </exception>
        public Range(int offset, int length)
        {
            if (length < 0)
                throw new ArgumentOutOfRangeException(nameof(length), length,
                    "Must be positive or zero.");

            Offset = offset;
            Length = length;
        }

        /// <summary>
        ///     The inclusive lower bound of the range.
        /// </summary>
        public int From => Offset;

        /// <summary>
        ///     The exclusive upper bound of the range.
        /// </summary>
        public int To => Offset + Length;

        /// <inheritdoc />
        public int CompareTo(Range other)
        {
            var offset = other.Offset - Offset;
            return offset != 0 ? offset : other.Length - Length;
        }

        /// <inheritdoc />
        public override bool Equals(object obj)
        {
            return obj is Range other && Equals(other);
        }

        /// <inheritdoc />
        public bool Equals(Range other)
        {
            return Offset == other.Offset && Length == other.Length;
        }

        /// <inheritdoc />
        public override int GetHashCode()
        {
            return HashCode.Combine(Offset, Length);
        }

        /// <inheritdoc />
        public override string ToString()
        {
            return $@"{Offset}→{Length}";
        }
    }
}
