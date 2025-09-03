using System;
using System.Linq;

namespace Unity.ML.Tokenization.Normalizers
{
    public class SequenceNormalizer : INormalizer
    {
        readonly INormalizer[] m_Normalizers;

        public SequenceNormalizer(params INormalizer[] normalizers)
        {
            if (normalizers.Any(n => n == null))
                throw new ArgumentNullException(nameof(normalizers));

            m_Normalizers = normalizers.ToArray();
        }

        public SubString Normalize(SubString input) =>
            m_Normalizers.Aggregate(input, (current, normalizer) => normalizer.Normalize(current));
    }
}
