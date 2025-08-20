using System;
using System.Collections.Generic;
using JetBrains.Annotations;

namespace Unity.ML.Tokenization
{
    /// <summary>
    ///     Contains the result of a tokenization pipeline ran by a <see cref="TokenizationPipeline" />
    ///     instance.
    /// </summary>
    public class Encoding : IEncoding
    {
        /// <inheritdoc cref="IEncoding.Attention" />
        readonly int[] m_Attention;

        /// <inheritdoc cref="IEncoding.Ids" />
        readonly int[] m_Ids;

        /// <inheritdoc cref="IEncoding.Overflow" />
        Encoding m_Overflow;

        /// <inheritdoc cref="IEncoding.TypeIds" />
        readonly int[] m_TypeIds;

        /// <summary>
        ///     Initializes a new instance of the <see cref="Encoding" /> type.
        /// </summary>
        /// <param name="ids">
        ///     The token <see cref="Ids" />.
        /// </param>
        /// <param name="attention">
        ///     The <see cref="Attention" /> mask.
        /// </param>
        /// <exception cref="ArgumentNullException">
        ///     <paramref name="ids" /> and <paramref name="attention" /> cannot be <c>null</c>.
        /// </exception>
        internal Encoding([NotNull] int[] ids, [NotNull] int[] attention)
        {
            m_Ids = ids ?? throw new ArgumentNullException(nameof(ids));
            m_Attention = attention ?? throw new ArgumentNullException(nameof(attention));
            // TODO: support typeIds. At the moment, it is assumed to be only zeros.
            m_TypeIds = new int[m_Ids.Length];
        }

        /// <inheritdoc />
        public int Length => m_Ids.Length;

        /// <inheritdoc />
        public IReadOnlyCollection<int> Ids => m_Ids;

        /// <inheritdoc />
        public IReadOnlyCollection<int> Attention => m_Attention;

        /// <inheritdoc />
        public IEncoding Overflow => m_Overflow;

        /// <inheritdoc />
        public IReadOnlyCollection<int> TypeIds => m_TypeIds;

        /// <summary>
        ///     Sets the next encoding instance storing the overflowing tokens.
        /// </summary>
        /// <param name="overflow">
        ///     The encoding storing the overflowing tokens.
        /// </param>
        /// <returns>
        ///     <see langword="this" />
        /// </returns>
        internal Encoding SetOverflow(Encoding overflow)
        {
            m_Overflow = overflow;
            return this;
        }
    }
}
