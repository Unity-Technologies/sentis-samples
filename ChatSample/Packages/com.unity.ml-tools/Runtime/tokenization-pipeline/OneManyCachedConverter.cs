using System;
using System.Collections.Generic;
using System.Linq;
using JetBrains.Annotations;

namespace Unity.ML.Tokenization
{
    class OneToManyCachedConverter<TFrom, TTo> : IOneToManyConverter<TFrom, TTo>
    {
        readonly Dictionary<TFrom, TTo[]> m_Cache;
        readonly IOneToManyConverter<TFrom, TTo> m_Converter;

        public OneToManyCachedConverter(IOneToManyConverter<TFrom, TTo> converter) : this(
            converter, EqualityComparer<TFrom>.Default)
        { }

        public OneToManyCachedConverter(
            [NotNull] IOneToManyConverter<TFrom, TTo> converter,
            [NotNull] IEqualityComparer<TFrom> inputComparer)
        {
            if (inputComparer == null)
                throw new ArgumentNullException(nameof(inputComparer));

            m_Cache = new(inputComparer);
            m_Converter = converter ?? throw new ArgumentNullException(nameof(converter));
        }

        public void Convert(TFrom input, IOutput<TTo> output)
        {
            if (!m_Cache.TryGetValue(input, out var cached))
            {
                var o = new Output<TTo>();

                m_Converter.Convert(input, o);
                cached = o.ToArray();
                m_Cache.Add(input, cached);
            }

            output.Add(cached);
        }
    }
}
