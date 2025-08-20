using System;
using System.Collections.Generic;
using System.Linq;

namespace Unity.ML.Tokenization.PostProcessors.Templating
{
    /// <summary>
    ///     Represents a post processing template.
    /// </summary>
    public class Template
    {
        /// <summary>
        ///     The elements representing the template.
        /// </summary>
        readonly Piece[] m_Pieces;

        /// <summary>
        ///     Initializes a new instance of the <see cref="Template" /> type from a sequence of
        ///     <see cref="Piece" />s.
        /// </summary>
        /// <param name="pieces">
        ///     The elements of the template.
        /// </param>
        /// <exception cref="ArgumentNullException">
        ///     <paramref name="pieces" /> cannot be <see langword="null" />.
        /// </exception>
        public Template(IEnumerable<Piece> pieces)
        {
            if (pieces is null)
                throw new ArgumentNullException(nameof(pieces));

            m_Pieces = pieces.ToArray();

            for (var i = 0; i < m_Pieces.Length; i++)
                if (m_Pieces[i] is null)
                    throw new ArgumentNullException(nameof(pieces),
                        $"{nameof(Piece)} #{i} is null.");
        }

        /// <summary>
        ///     Initializes a new instance of the <see cref="Template" /> type from
        ///     <see cref="string" /> representation.
        /// </summary>
        /// <param name="repr">
        ///     The <see cref="string" /> representation of the template.
        /// </param>
        /// <exception cref="ArgumentNullException">
        ///     <paramref name="repr" /> is <see langword="null" /> or whitespace.
        /// </exception>
        public Template(string repr)
        {
            if (string.IsNullOrWhiteSpace(repr))
                throw new ArgumentNullException(nameof(repr));

            m_Pieces = Parse(repr).ToArray();
        }

        /// <summary>
        ///     The sequence of elements.
        /// </summary>
        public IEnumerable<Piece> Pieces => m_Pieces;

        /// <summary>
        ///     Iterates along the <paramref name="repr" /> <see cref="string" /> starting at
        ///     <paramref name="offset" /> to find the next no-whitespace character.
        /// </summary>
        /// <param name="repr">
        ///     The source <see cref="string" /> to check.
        /// </param>
        /// <param name="offset">
        ///     The position in the <see cref="string" /> to start looking at whitespace.
        /// </param>
        static void EatWhiteSpaces(string repr, ref int offset)
        {
            while (offset < repr.Length && char.IsWhiteSpace(repr[offset]))
                offset++;
        }

        /// <summary>
        ///     Parses an <see cref="int" /> within a <see cref="string" />.
        /// </summary>
        /// <param name="repr">
        ///     The source <see cref="string" /> to parse the <see cref="int" /> into.
        /// </param>
        /// <param name="offset">
        ///     The position in the <see cref="string" /> to start parsing the <see cref="int" />.
        /// </param>
        /// <returns>
        ///     The parsed <see cref="int" />.
        /// </returns>
        /// <exception cref="FormatException">
        ///     <paramref name="repr" /> at the specified <paramref name="offset" /> doesn't match
        ///     with a <see cref="int" /> representation.
        /// </exception>
        static int ParseInt(string repr, ref int offset)
        {
            var len = 0;
            while (offset + len < repr.Length
                   && char.IsNumber(repr, offset + len))
                len++;

            if (len == 0)
                throw new FormatException($"Invalid integer representation at {offset}");

            var value = int.Parse(repr.AsSpan(offset, len));
            offset += len;
            return value;
        }

        /// <summary>
        ///     Parses the <paramref name="repr" /> into an instance of <see cref="Sequence" />.
        /// </summary>
        /// <param name="repr">
        ///     The source <see cref="string" /> to parse the <see cref="Sequence" /> from.
        /// </param>
        /// <param name="offset">
        ///     The starting position of the representation of the <see cref="Sequence" /> in
        ///     <paramref name="repr" />.
        /// </param>
        /// <returns>
        ///     The parsed instance of <see cref="Sequence" />.
        /// </returns>
        /// <exception cref="ArgumentOutOfRangeException">
        ///     <paramref name="offset" /> is not in the range of <paramref name="repr" />.
        /// </exception>
        /// <exception cref="FormatException">
        ///     <paramref name="repr" /> of <paramref name="offset" /> should start with a <c>$</c>.
        /// </exception>
        /// <exception cref="IndexOutOfRangeException">
        ///     <paramref name="offset" /> has reach the end of <paramref name="repr" /> but didn't
        ///     reach the end of the <see cref="string" /> representation of a
        ///     <see cref="Sequence" />.
        /// </exception>
        static Sequence ParseSequence(string repr, ref int offset)
        {
            if (offset < 0 || offset >= repr.Length)
                throw new ArgumentOutOfRangeException(nameof(offset), offset, null);

            if (repr[offset] is not '$')
                throw new FormatException("A sequence must start of the character '$'");

            offset++;
            if (offset >= repr.Length)
                throw new IndexOutOfRangeException(
                    "End of string. Expected: Sequence identifier or type id.");

            var sequence = SequenceIdentifier.A;
            var typeId = 0;

            var c = repr[offset];

            switch (c)
            {
                // format: $sequence_id[:type_id]
                case 'a' or 'A' or 'b' or 'B':
                {
                    sequence = c is 'a' or 'A' ? SequenceIdentifier.A : SequenceIdentifier.B;
                    offset++;

                    // looking for type_id if available
                    if (offset < repr.Length && repr[offset] is ':')
                    {
                        offset++;
                        if (offset >= repr.Length)
                            throw new IndexOutOfRangeException(
                                $"Expected: type id at {offset}. Actual: end of string.");

                        typeId = ParseInt(repr, ref offset);
                    }

                    break;
                }

                // format: $type_id
                case >= '0' and <= '9':
                    typeId = ParseInt(repr, ref offset);
                    break;

                // wrong format
                default:
                    throw new FormatException(
                        $"Expected sequence identifier or type id at {offset}. Actual: {c}");
            }

            return new Sequence(sequence, typeId);
        }

        /// <summary>
        ///     Parses the <paramref name="repr" /> into an instance of <see cref="SpecialToken" />.
        /// </summary>
        /// <param name="repr">
        ///     The source <see cref="string" /> to parse the <see cref="SpecialToken" /> from.
        /// </param>
        /// <param name="offset">
        ///     The starting position of the representation of the <see cref="SpecialToken" /> in
        ///     <paramref name="repr" />.
        /// </param>
        /// <returns>
        ///     The parsed instance of <see cref="SpecialToken" />.
        /// </returns>
        /// <exception cref="ArgumentOutOfRangeException">
        ///     <paramref name="offset" /> is not in the range of <paramref name="repr" />.
        /// </exception>
        /// <exception cref="FormatException">
        ///     <paramref name="repr" /> at <paramref name="offset" /> doesn't represent a
        ///     <see cref="SpecialToken" />.
        /// </exception>
        static SpecialToken ParseToken(string repr, ref int offset)
        {
            if (offset < 0 || offset >= repr.Length)
                throw new ArgumentOutOfRangeException(nameof(offset), offset, null);

            // get token
            string token;
            {
                var len = 0;
                while (offset + len < repr.Length
                       && !char.IsWhiteSpace(repr[offset + len])
                       && repr[offset + len] is not ':')
                    len++;

                if (len == 0)
                    throw new FormatException("Expected token with minimum length of 1 character");

                token = repr.Substring(offset, len);
                offset += len;
            }

            // get type_Id
            var typeId = 0;
            if (offset < repr.Length && repr[offset] is ':')
            {
                offset++;
                typeId = ParseInt(repr, ref offset);
            }

            return new(token,
                typeId == 0 ? SequenceIdentifier.A : SequenceIdentifier.B);
        }

        /// <summary>
        ///     Parses the <paramref name="repr" /> into a sequence of <see cref="Piece" />s.
        /// </summary>
        /// <param name="repr">
        ///     The source <see cref="string" /> to parse <see cref="Piece" />s from.
        /// </param>
        /// <returns>
        ///     The sequence of parsed <see cref="Piece" />s.
        /// </returns>
        public static IEnumerable<Piece> Parse(string repr)
        {
            var offset = 0;

            EatWhiteSpaces(repr, ref offset);

            while (offset < repr.Length)
            {
                if (repr[offset] is '$')
                    yield return ParseSequence(repr, ref offset);
                else
                    yield return ParseToken(repr, ref offset);

                EatWhiteSpaces(repr, ref offset);
            }
        }

        /// <inheritdoc />
        public override string ToString() => string.Join(' ', m_Pieces.AsEnumerable());
    }
}
