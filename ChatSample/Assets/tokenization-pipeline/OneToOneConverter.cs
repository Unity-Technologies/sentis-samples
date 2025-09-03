using System;
using System.Collections.Generic;
using JetBrains.Annotations;

namespace Unity.ML.Tokenization
{
    /// <summary>
    ///     Caches the result of an <see cref="IOneToOneConverter{TFrom,TTo}" />
    /// </summary>
    /// <typeparam name="TFrom">
    ///     The input type.
    /// </typeparam>
    /// <typeparam name="TTo">
    ///     The converted type.
    /// </typeparam>
    class OneToOneCachedConverter<TFrom, TTo> : IOneToOneConverter<TFrom, TTo>
    {
        readonly Dictionary<TFrom, TTo> m_Cache;
        readonly IOneToOneConverter<TFrom, TTo> m_Converter;

        /// <summary>
        ///     Initializes a new instance of the <see cref="OneToOneCachedConverter{TFrom,TTo}"/>
        /// type.
        /// </summary>
        /// <param name="converter">
        ///     The converter to call if the conversion is not cached.
        /// </param>
        public OneToOneCachedConverter(IOneToOneConverter<TFrom, TTo> converter) : this(converter,
            EqualityComparer<TFrom>.Default)
        {
        }

        /// <summary>
        ///     Initializes a new instance of the <see cref="OneToOneCachedConverter{TFrom,TTo}"/>
        /// type.
        /// </summary>
        /// <param name="converter">
        ///     The converter to call if the conversion is not cached.
        /// </param>
        /// <param name="inputComparer">
        ///     Compares the input to find the cached conversion.
        /// </param>
        /// <exception cref="ArgumentNullException">
        ///     <paramref name="inputComparer"/> cannot be <c>null</c>.
        /// </exception>
        public OneToOneCachedConverter(
            [NotNull] IOneToOneConverter<TFrom, TTo> converter,
            [NotNull] IEqualityComparer<TFrom> inputComparer)
        {
            if (inputComparer == null)
                throw new ArgumentNullException(nameof(inputComparer));

            m_Cache = new(inputComparer);
            m_Converter = converter ?? throw new ArgumentNullException(nameof(converter));
        }

        /// <inheritdoc cref="IOneToOneConverter{TFrom,TTo}.Convert"/>
        public TTo Convert(TFrom input)
        {
            if (m_Cache.TryGetValue(input, out var output))
                return output;

            output = m_Converter.Convert(input);
            m_Cache.Add(input, output);
            return output;
        }

        TTo IOneToOneConverter<TFrom, TTo>.Convert(TFrom input) => Convert(input);
    }
}
