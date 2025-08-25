using System;
using JetBrains.Annotations;

namespace Unity.ML.Tokenization.Normalizers
{
    /// <summary>
    ///     Adds a prefix to the input string.
    /// </summary>
    public class PrependNormalizer : INormalizer
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="PrependNormalizer" /> type.
        /// </summary>
        /// <param name="prefix">
        ///     The prefix to add to the input string when passed to
        ///     <see cref="INormalizer.Normalize" />.
        /// </param>
        /// <exception cref="ArgumentNullException">
        ///     The <paramref name="prefix" /> cannot be <c>null</c> or empty.
        /// </exception>
        public PrependNormalizer([NotNull] string prefix)
        {
            if (string.IsNullOrEmpty(prefix))
                throw new ArgumentNullException(nameof(prefix));

            Prefix = prefix;
        }

        /// <summary>
        ///     The prefix to add to the input string when passed to
        ///     <see cref="INormalizer.Normalize" />.
        /// </summary>
        public string Prefix { get; }

        /// <inheritdoc />
        public SubString Normalize(SubString input) => $"{Prefix}{input}";
    }
}
