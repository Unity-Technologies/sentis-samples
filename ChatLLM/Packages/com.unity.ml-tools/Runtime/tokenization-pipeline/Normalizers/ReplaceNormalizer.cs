using System;
using JetBrains.Annotations;

namespace Unity.ML.Tokenization.Normalizers
{
    /// <summary>
    ///     Replaces a specified pattern by another string.
    /// </summary>
    public class ReplaceNormalizer : INormalizer
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="ReplaceNormalizer" /> type.
        /// </summary>
        /// <param name="pattern">
        ///     The pattern to look for in the input string.
        /// </param>
        /// <param name="replacement">
        ///     The string to replace the pattern with.
        /// </param>
        /// <exception cref="ArgumentNullException">
        ///     The <paramref name="pattern" /> cannot be null or empty.
        /// </exception>
        public ReplaceNormalizer([NotNull] string pattern, [CanBeNull] string replacement)
        {
            if (string.IsNullOrEmpty(pattern))
                throw new ArgumentNullException(nameof(pattern));

            Pattern = pattern;
            Replacement = replacement ?? string.Empty;
        }

        /// <summary>
        ///     The pattern to look for in the input string.
        /// </summary>
        public string Pattern { get; }

        /// <summary>
        ///     The string to replace the pattern with.
        /// </summary>
        public string Replacement { get; }

        /// <inheritdoc />
        public SubString Normalize(SubString input) =>
            input.ToString().Replace(Pattern, Replacement);
    }
}
