using System;
using System.Collections;
using System.Collections.Generic;
using JetBrains.Annotations;

namespace Unity.ML.Tokenization
{
    /// <summary>
    ///     Represents a portion of a <see cref="string" /> value.
    /// </summary>
    /// <remarks>
    ///     This type is required as <see cref="ReadOnlySpan{T}" /> has some blocking constraints.
    /// </remarks>
    public partial struct SubString
        : IEquatable<string>,
            IComparable<string>,
            IEquatable<SubString>,
            IComparable<SubString>,
            IEnumerable<char>
    {
        const string k_NullSourceExceptionMessage =
            "The underlying source of this substring is null";

        /// <summary>
        ///     Creates a <see cref="SubString" /> instance from a full <see cref="string" /> value.
        /// </summary>
        /// <param name="input">
        ///     The original <see cref="string" /> value.
        ///     The resulting <see cref="SubString" /> will cover the whole value of
        ///     <paramref name="input" />.
        /// </param>
        /// <returns>
        ///     A <see cref="SubString" /> instance covering the whole <paramref name="input" />.
        /// </returns>
        public static implicit operator SubString(string input) => new(input);

        /// <summary>
        ///     Gets a <see cref="string" /> value from the portion of the source
        ///     <see cref="string" /> of this <see cref="SubString" />.
        /// </summary>
        /// <param name="input">
        ///     The <see cref="SubString" /> value to convert to a <see cref="string" /> value.
        /// </param>
        /// <returns>
        ///     The <see cref="string" /> representing the value of this <see cref="SubString" />.
        /// </returns>
        public static implicit operator string(SubString input) => input.ToString();

        /// <summary>
        ///     Creates a <see cref="SubString" /> instance from a <see cref="string" /> source and
        ///     the bounds of the portion to keep.
        /// </summary>
        /// <param name="source">
        ///     The source <see cref="string" /> to build this <see cref="SubString" /> from.
        /// </param>
        /// <param name="from">
        ///     The lower bound of the portion of <paramref name="source" /> to keep.
        /// </param>
        /// <param name="to">
        ///     The upper bound of the portion of <paramref name="source" /> to keep.
        /// </param>
        /// <returns>
        ///     A <see cref="SubString" /> value.
        /// </returns>
        public static SubString FromTo(string source, int from, int to) =>
            new(source, from, to - from);

        /// <summary>
        ///     The computed hash code of the portion.
        ///     <see cref="SubString" /> uses itw owns implementation of <see cref="string" /> hash
        ///     code computation because the standard one is not exposed as a helper method by the
        ///     standard library at the moment (recent versions of .NET exposes it).
        ///     It is a nullable value as it is not computed while not required.
        /// </summary>
        int? m_HashCode;

        /// <summary>
        ///     The number of UTF-8 characters of this portion.
        /// </summary>
        int? m_UtfLength;

        readonly string m_Source;
        readonly int m_Offset;
        readonly int m_Length;

        /// <summary>
        ///     Initializes a new instance of the <see cref="SubString" /> type.
        /// </summary>
        /// <param name="source">
        ///     The source <see cref="string" /> from which this <see cref="SubString" /> is built.
        /// </param>
        /// <param name="offset">
        ///     The lower bound of the portion of <see cref="Source" /> to keep.
        /// </param>
        /// <param name="length">
        ///     The length of the portion of <see cref="Source" /> to keep.
        /// </param>
        public SubString([CanBeNull] string source, int offset, int length)
        {
            m_Source = source;

            if (offset < 0 || offset > (source?.Length ?? 0))
                throw new ArgumentOutOfRangeException(nameof(offset), offset, null);
            if (length < 0 || offset + length > (source?.Length ?? 0))
                throw new ArgumentOutOfRangeException(nameof(length), length, null);

            m_Offset = offset;
            m_Length = length;
            m_HashCode = null;
            m_UtfLength = default;
        }

        /// <summary>
        ///     Initializes a new instance of the <see cref="SubString" /> type.
        /// </summary>
        /// <param name="source">
        ///     The source <see cref="string" /> from which this <see cref="SubString" /> is built.
        ///     This constructor keeps the whole <paramref name="source" />.
        /// </param>
        public SubString(string source) : this(source, 0, source?.Length ?? 0)
        { }

        /// <summary>
        ///     Tells whether the portion covers the source string.
        /// </summary>
        public bool IsApplied =>
            m_Source is null
                ? throw new NullReferenceException(k_NullSourceExceptionMessage)
                : m_Offset == 0 && m_Length == m_Source.Length;

        /// <summary>
        ///     The source <see cref="string" /> from which this <see cref="SubString" /> is built.
        /// </summary>
        public readonly string Source => m_Source;

        /// <summary>
        ///     The lower bound of the portion of <see cref="Source" /> to keep.
        /// </summary>
        public readonly int Offset => m_Offset;

        /// <summary>
        ///     The number of <see cref="char" /> of this portion.
        /// </summary>
        public readonly int Length => m_Length;

        /// <summary>
        ///     The number of Utf-8 valid characters of this portion.
        /// </summary>
        public int UtfLength => m_UtfLength ??= GetUtfLength();

        /// <summary>
        ///     Tells whether the substring does not reference any valid source.
        /// </summary>
        public bool IsNull => m_Source is null;

        /// <summary>
        ///     Tells whether this instance is empty.
        /// </summary>
        public bool IsEmpty =>
            m_Source is null ? throw new NullReferenceException(k_NullSourceExceptionMessage) : m_Length == 0;

        /// <summary>
        ///     Tells whether this instance is null, empty or if it just contains white spaces.
        /// </summary>
        public bool IsNullOrWhiteSpace
        {
            get
            {
                var source = m_Source;
                if (source is null)
                    return true;

                for (int i = m_Offset, limit = m_Offset + m_Length; i < limit; i++)
                    if (!char.IsWhiteSpace(source[i]))
                        return false;

                return true;
            }
        }

        /// <summary>
        ///     Returns a new <see cref="SubString" /> value which source <see cref="string" /> is the
        ///     portion of this one.
        /// </summary>
        /// <returns>
        ///     A new <see cref="SubString" /> value which source is the portion of this one.
        /// </returns>
        /// <remarks>
        ///     If the hash code has already been computed for this <see cref="SubString" />, it is
        ///     copied to the new one.
        /// </remarks>
        public SubString Apply() =>
            IsApplied ? this : new(m_Source.Substring(m_Offset, m_Length)) {m_HashCode = m_HashCode};

        /// <summary>
        ///     Gets a portion of this instance.
        /// </summary>
        /// <param name="offset">
        ///     The zero-based starting character position of a substring in this instance.
        /// </param>
        /// <param name="length">
        ///     The number of characters in the substring.
        /// </param>
        /// <returns>
        ///     A new <see cref="SubString" /> instance.
        /// </returns>
        public SubString Sub(int offset, int length) =>
            m_Source is null
                ? throw new NullReferenceException(k_NullSourceExceptionMessage)
                : new SubString(m_Source, offset + m_Offset, length);

        /// <summary>
        ///     Gets a portion of this instance.
        /// </summary>
        /// <param name="offset">
        ///     The zero-based starting character position of a substring in this instance.
        /// </param>
        /// <returns>
        ///     A new <see cref="SubString" /> instance.
        /// </returns>
        public SubString Sub(int offset) =>
            m_Source is null
                ? throw new NullReferenceException(k_NullSourceExceptionMessage)
                : new SubString(m_Source, offset + m_Offset, m_Length - offset);

        /// <summary>
        ///     Gets a portion of this instance, considering the unicode characters instead of chars.
        /// </summary>
        /// <param name="offset">
        ///     The zero-based starting character position of a substring in this instance.
        /// </param>
        /// <param name="length">
        ///     The number of characters in the substring.
        /// </param>
        /// <returns>
        ///     A new <see cref="SubString" /> instance.
        /// </returns>
        public SubString UtfSub(int offset, int length)
        {
            if (m_Source is null)
                throw new NullReferenceException(k_NullSourceExceptionMessage);

            var charOffset = 0;

            for (var i = 0; i < offset; i++)
            {
                if (charOffset >= m_Length)
                    throw new ArgumentOutOfRangeException(nameof(offset));

                while (charOffset < m_Length && char.IsSurrogate(m_Source[m_Offset + charOffset]))
                    charOffset++;

                charOffset++;
            }

            var charLength = 0;

            for (var i = 0; i < length; i++)
            {
                if (charOffset + charLength >= m_Length)
                    throw new ArgumentOutOfRangeException(nameof(length));

                while (charOffset + charLength < m_Length
                    && char.IsSurrogate(m_Source[m_Offset + charOffset + charLength]))
                    charLength++;

                charLength++;
            }

            return new(m_Source, m_Offset + charOffset, charLength);
        }

        /// <summary>
        ///     Gets a portion of this instance, considering the unicode characters instead of chars.
        /// </summary>
        /// <param name="offset">
        ///     The zero-based starting character position of a substring in this instance.
        /// </param>
        /// <returns>
        ///     A new <see cref="SubString" /> instance.
        /// </returns>
        public SubString UtfSub(int offset)
        {
            if (m_Source is null)
                throw new NullReferenceException(k_NullSourceExceptionMessage);

            var charOffset = 0;

            for (var i = 0; i < offset; i++)
            {
                if (charOffset >= m_Length)
                    throw new ArgumentOutOfRangeException(nameof(offset));

                while (charOffset < m_Length && char.IsSurrogate(m_Source[m_Offset + charOffset]))
                    charOffset++;

                charOffset++;
            }

            return new(m_Source, m_Offset + charOffset, m_Length - charOffset);
        }

        /// <summary>
        ///     Tells whether this <see cref="SubString"/> starts with the specified
        ///     <paramref name="prefix"/>.
        /// </summary>
        /// <param name="prefix">
        ///     The pattern to compare to the beginning of this <see cref="SubString"/>.
        /// </param>
        /// <returns>
        ///     Whether this <see cref="SubString"/> starts with the specified
        ///     <paramref name="prefix"/>.
        /// </returns>
        /// <exception cref="ArgumentNullException">
        ///     <paramref name="prefix"/> cannot ne <c>null</c>.
        /// </exception>
        public bool StartsWith(SubString prefix)
        {
            if (m_Source is null)
                throw new NullReferenceException(k_NullSourceExceptionMessage);

            if (prefix.m_Source is null)
                throw new ArgumentNullException(nameof(prefix));

            var prefixLength = prefix.m_Length;
            if (m_Offset + prefixLength > m_Source.Length)
                return false;

            unsafe
            {
                fixed (char* pSource = m_Source)
                    fixed (char* pPrefixSource = prefix.m_Source)
                    {
                        var pMe = pSource + m_Offset;
                        var pPrefix = pPrefixSource + prefix.m_Offset;
                        for (var i = 0; i < prefixLength; i++)
                            if (pPrefix[i] != pMe[i])
                                return false;
                    }
            }

            return true;
        }

        /// <inheritdoc />
        public int CompareTo(string other) => CompareTo((SubString) other);

        /// <inheritdoc />
        public unsafe int CompareTo(SubString other)
        {
            if (m_Source is null)
                throw new NullReferenceException(k_NullSourceExceptionMessage);

            var (from, length) = (m_Offset, m_Length);
            var (otherFrom, otherLength) = (other.m_Offset, other.m_Length);

            length = length < otherLength ? length : otherLength;

            fixed (char* pSource = m_Source)
                fixed (char* pOtherSource = other.m_Source)
                {
                    var pSub = pSource + from;
                    var pOtherSub = pOtherSource + otherFrom;

                    for (var i = 0; i < length; i++)
                    {
                        var comp = pSub[i].CompareTo(pOtherSub[i]);
                        if (comp != 0)
                            return comp;
                    }
                }

            return length - otherLength;
        }

        /// <summary>
        ///     Deconstructs this <see cref="SubString" />.
        /// </summary>
        /// <param name="source">
        ///     The source <see cref="string" /> from which this <see cref="SubString" /> is built.
        /// </param>
        /// <param name="offset">
        ///     The lower bound of the portion of <see cref="Source" /> to keep.
        /// </param>
        /// <param name="length">
        ///     The length of the portion of <see cref="Source" /> to keep.
        /// </param>
        public void Deconstruct(out string source, out int offset, out int length)
        {
            if (m_Source is null)
                throw new NullReferenceException(k_NullSourceExceptionMessage);

            source = m_Source;
            offset = m_Offset;
            length = m_Length;
        }

        public int IndexOf(SubString sub, int startIndex = 0)
        {
            if (m_Source is null)
                throw new NullReferenceException(k_NullSourceExceptionMessage);

            for(int i = startIndex, limit = m_Length - sub.Length; i <= limit; i++)
            {
                var found = true;
                for (var j = 0; j < sub.Length; j++)
                {
                    var c1 = this[i + j];
                    var c2 =  sub[j];
                    if (c1 == c2)
                        continue;
                    found = false;
                    break;
                }

                if (found)
                    return i;
            }

            return -1;
        }

        /// <summary>
        ///     Computes a hashcode of this <see cref="SubString"/> instance with its
        ///     <paramref name="prefix"/> and <paramref name="suffix"/> like if they were combined.
        /// </summary>
        /// <param name="prefix">
        ///     A prefix value to combine with this.
        /// </param>
        /// <param name="suffix">
        ///     A suffix value to combine with this.
        /// </param>
        /// <returns>
        ///     The hashcode of "{prefix}{this}{suffix}"
        /// </returns>
        public int GetHashCode(SubString? prefix, SubString? suffix)
        {
            var hashA = 5381;
            var hashB = hashA;

            if(prefix.HasValue)
                Hash(prefix.Value, ref hashA, ref hashB);

            Hash(this, ref hashA, ref hashB);

            if(suffix.HasValue)
                Hash(suffix.Value, ref hashA, ref hashB);

            var hashCode = hashA + hashB * 1566083941;
            return hashCode;

            void Hash(SubString s, ref int a, ref int b)
            {
                var (source, i, limit) = (s.m_Source, s.m_Offset, s.m_Offset + s.m_Length);
                while (i < limit)
                {
                    var c = source[i];
                    a = ((a << 5) + a) ^ c;

                    if (++i == limit)
                        break;

                    c = source[i];
                    b = ((b << 5) + b) ^ c;
                    i++;
                }
            }
        }

        /// <inheritdoc />
        public bool Equals(string other) => other != null && Equals((SubString) other);

        /// <inheritdoc />
        public bool Equals(SubString other)
        {
            if (m_Source is null)
                throw new NullReferenceException(k_NullSourceExceptionMessage);

            if (other.m_Source is null)
                return false;

            var length = m_Length;
            if (length != other.m_Length)
                return false;

            unsafe
            {
                fixed (char* pSource = m_Source)
                    fixed (char* pOther = other.m_Source)
                    {
                        var pChar = pSource + m_Offset;
                        var pOtherChar = pOther + other.m_Offset;

                        for (var i = 0; i < length; i++)
                            if (pChar[i] != pOtherChar[i])
                                return false;
                    }
            }

            return true;
        }

        /// <inheritdoc />
        public override int GetHashCode()
        {
            if (m_Source is null)
                throw new NullReferenceException(k_NullSourceExceptionMessage);

            m_HashCode ??= GetHashCode(default, default);

            return m_HashCode.Value;
        }

        /// <inheritdoc />
        public override string ToString() => IsApplied ? m_Source : m_Source.Substring(m_Offset, m_Length);

        int GetUtfLength()
        {
            var count = 0;
            for (int i = m_Offset, limit = m_Offset + m_Length; i < limit; i++)
            {
                while (i + 1 < limit && char.IsSurrogate(m_Source[i]))
                    i++;

                count++;
            }

            return count;
        }

        /// <summary>
        ///     Gets the sequence of <see cref="char" /> from the portion of the source
        ///     <see cref="string" /> covered by this <see cref="SubString" />.
        /// </summary>
        /// <returns>
        ///     The sequence of <see cref="char" /> from the portion of the source
        ///     <see cref="string" /> covered by this <see cref="SubString" />.
        /// </returns>
        IEnumerable<char> GetChars()
        {
            if (m_Source is null)
                throw new NullReferenceException(k_NullSourceExceptionMessage);

            var (source, from, to) = (m_Source, m_Offset, m_Offset + m_Length);
            for (var i = from; i < to; i++)
                yield return source[i];
        }

        IEnumerator IEnumerable.GetEnumerator() => GetChars().GetEnumerator();

        IEnumerator<char> IEnumerable<char>.GetEnumerator() => GetChars().GetEnumerator();

        /// <summary>
        ///     Gets the character ar <paramref name="index"/>.
        /// </summary>
        /// <param name="index">
        ///     Index of the character to get.
        /// </param>
        public char this[int index]
        {
            get
            {
                if (m_Source is null)
                    throw new NullReferenceException(k_NullSourceExceptionMessage);
                return m_Source[m_Offset + index];
            }
        }
    }
}
