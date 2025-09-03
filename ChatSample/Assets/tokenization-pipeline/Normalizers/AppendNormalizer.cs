using System;
using JetBrains.Annotations;

namespace Unity.ML.Tokenization.Normalizers
{
    /// <summary>
    ///     Adds a suffix to the input string.
    /// </summary>
    public class AppendNormalizer : INormalizer
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="AppendNormalizer" /> type.
        /// </summary>
        /// <param name="suffix">
        ///     The suffix to add to the input string when passed to
        ///     <see cref="INormalizer.Normalize" />.
        /// </param>
        /// <exception cref="ArgumentNullException">
        ///     The <paramref name="suffix" /> cannot be <c>null</c> or empty.
        /// </exception>
        public AppendNormalizer([NotNull] string suffix)
        {
            if (string.IsNullOrEmpty(suffix))
                throw new ArgumentNullException(nameof(suffix));

            Suffix = suffix;
        }

        /// <summary>
        ///     The prefix to add to the input string when passed to
        ///     <see cref="INormalizer.Normalize" />.
        /// </summary>
        public string Suffix { get; }

        /// <inheritdoc />
        public SubString Normalize(SubString input) => $"{input}{Suffix}";
    }
}
