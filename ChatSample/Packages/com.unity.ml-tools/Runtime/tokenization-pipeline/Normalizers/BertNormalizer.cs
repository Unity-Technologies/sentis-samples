using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;

namespace Unity.ML.Tokenization.Normalizers
{
    /// <summary>
    ///     Normalizes raw text input for Bert model.
    /// </summary>
    public class BertNormalizer : INormalizer
    {
        /// <summary>
        ///     Initializes a new instance of the type <see cref="BertNormalizer" />
        /// </summary>
        /// <param name="cleanText">
        ///     If <see langword="true" />, removes control characters and replaces whitespaces by
        ///     the classic one.
        /// </param>
        /// <param name="handleCjkChars">
        ///     If <see langword="true" />, puts spaces around each chinese character.
        /// </param>
        /// <param name="stripAccents">
        ///     If <see langword="true" />, strips all accents.
        ///     If set to <see langword="null" />, it takes the value of <paramref name="lowerCase" />
        ///     (original BERT implementation).
        /// </param>
        /// <param name="lowerCase">
        ///     If <see langword="true" />, converts the input to lowercase.
        /// </param>
        public BertNormalizer(
            bool cleanText = true,
            bool handleCjkChars = true,
            bool? stripAccents = null,
            bool lowerCase = true)
        {
            CleanText = cleanText;
            HandleCjkChars = handleCjkChars;
            LowerCase = lowerCase;
            StripAccents = stripAccents ?? lowerCase;
        }

        /// <summary>
        ///     If <see langword="true" />, removes control characters and replaces whitespaces by
        ///     the classic one.
        /// </summary>
        public bool CleanText { get; }

        /// <summary>
        ///     If <see langword="true" />, puts spaces around each chinese character.
        /// </summary>
        public bool HandleCjkChars { get; }

        /// <summary>
        ///     If <see langword="true" />, strips all accents.
        /// </summary>
        public bool StripAccents { get; }

        /// <summary>
        ///     If <see langword="true" />, converts the input to lowercase.
        /// </summary>
        public bool LowerCase { get; }

        /// <summary>
        ///     Tells whether <paramref name="c" /> is a CJK Unicode block character.
        /// </summary>
        /// <param name="c">
        ///     The character to test.
        /// </param>
        /// <returns>
        ///     Whether <paramref name="c" /> is a chinese character.
        /// </returns>
        /// <seealso href="https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)" />
        /// <seealso href="https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_Extension_A" />
        /// <seealso href="https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_Extension_B" />
        /// <seealso href="https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_Extension_C" />
        /// <seealso href="https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_Extension_D" />
        /// <seealso href="https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_Extension_E" />
        /// <seealso href="https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_Extension_F" />
        /// <seealso href="https://en.wikipedia.org/wiki/CJK_Compatibility_Ideographs" />
        /// <seealso href="https://en.wikipedia.org/wiki/CJK_Compatibility_Ideographs_Supplement" />
        static bool IsCjk(char c)
        {
            return Convert.ToUInt32(c) is
                // CJK Unified Ideographs
                >= 0x4e00 and <= 0x9fff or
                // CJK Compatibility Ideographs
                >= 0xf900 and <= 0xfaff or
                // CJK Unified Ideographs Extension A
                >= 0x3400 and <= 0x4dbf or
                // CJK Unified Ideographs Extension B
                >= 0x20000 and <= 0x2a6df or
                // CJK Unified Ideographs Extension C
                >= 0x2a700 and <= 0x2b739 or
                // CJK Unified Ideographs Extension D
                >= 0x2b740 and <= 0x2b81f or
                // CJK Unified Ideographs Extension E
                >= 0x2b820 and <= 0x2ceaf or
                // CJK Compatibility Ideographs Supplement
                >= 0x2f800 and <= 0x2fa1f;
        }

        /// <summary>
        ///     Tells whether the given character is a replacement character.
        /// </summary>
        /// <param name="c">
        ///     The character to test.
        /// </param>
        /// <returns>
        ///     Whether the given character is a replacement character.
        /// </returns>
        static bool IsReplacementChar(char c)
        {
            return Convert.ToUInt32(c) == 0xfffd;
        }

        /// <summary>
        ///     Cleans the <paramref name="input" /> from replacement characters, whitespaces and
        ///     control characters.
        /// </summary>
        /// <param name="input">
        ///     The sequence of <see cref="char" /> to clean.
        /// </param>
        /// <returns>
        ///     The sequence, cleaned.
        /// </returns>
        static IEnumerable<char> ApplyCleanText(IEnumerable<char> input)
        {
            return input
                .Where(c => !IsReplacementChar(c))
                .Select(c => char.IsWhiteSpace(c) ? ' ' : c)
                .Where(c => !char.IsControl(c));
        }

        /// <summary>
        ///     Surround CJK characters with a single whitespace.
        /// </summary>
        /// <param name="input">
        ///     The sequence of <see cref="char" /> in which to search for CJK characters.
        /// </param>
        /// <returns>
        ///     The updated sequence of <see cref="char" />.
        /// </returns>
        static IEnumerable<char> ApplyHandleCjkChars(IEnumerable<char> input)
        {
            foreach (var c in input)
            {
                var isCjk = IsCjk(c);
                if (isCjk) yield return ' ';
                yield return c;
                if (isCjk) yield return ' ';
            }
        }

        /// <summary>
        ///     Removes accents using <see cref="NormalizationForm.FormD" /> and
        ///     ignore <see cref="UnicodeCategory.NonSpacingMark" /> characters.
        /// </summary>
        /// <param name="input">
        ///     The sequence of <see cref="char" /> to update.
        /// </param>
        /// <returns>
        ///     The updated sequence of <see cref="char" />.
        /// </returns>
        static IEnumerable<char> ApplyStripAccents(IEnumerable<char> input)
        {
            return string
                .Concat(input)
                .Normalize(NormalizationForm.FormD)
                .Where(c =>
                    CharUnicodeInfo.GetUnicodeCategory(c) != UnicodeCategory.NonSpacingMark);
        }

        /// <summary>
        ///     Turns the <see cref="char" /> of the <paramref name="input" /> into their lowercase
        ///     version.
        /// </summary>
        /// <param name="input">
        ///     The sequence of <see cref="char" /> to update.
        /// </param>
        /// <returns>
        ///     The updated sequence.
        /// </returns>
        static IEnumerable<char> ApplyLowerCase(IEnumerable<char> input)
        {
            return input.Select(char.ToLowerInvariant);
        }

        /// <inheritdoc />
        public SubString Normalize(SubString input)
        {
            var builder = input.AsEnumerable();

            if (CleanText)
                builder = ApplyCleanText(builder);

            if (HandleCjkChars)
                builder = ApplyHandleCjkChars(builder);

            if (StripAccents)
                builder = ApplyStripAccents(builder);

            if (LowerCase)
                builder = ApplyLowerCase(builder);

            return new string(builder.ToArray());
        }
    }
}
