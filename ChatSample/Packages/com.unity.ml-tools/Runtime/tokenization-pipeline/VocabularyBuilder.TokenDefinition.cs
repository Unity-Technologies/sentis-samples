using System;
using JetBrains.Annotations;

namespace Unity.ML.Tokenization
{
    partial class VocabularyBuilder
    {
        class TokenDefinition : ITokenDefinition
        {
            public readonly int Id;
            public readonly string Key;
            public readonly bool IsSpecial;
            public readonly string Value;

            string m_Repr;

            public TokenDefinition(
                int id,
                [NotNull] string key,
                [NotNull] string value,
                bool isSpecial)
            {
                Id = id;
                Key = key;
                Value = value;
                IsSpecial = isSpecial;
            }

            string ITokenDefinition.Value => Value;

            int ITokenDefinition.Id => Id;

            string ITokenDefinition.Key => Key;

            bool ITokenDefinition.IsSpecial => IsSpecial;

            bool IEquatable<ITokenDefinition>.Equals(ITokenDefinition other) => Equals(other);

            int IComparable<ITokenDefinition>.CompareTo(ITokenDefinition other) => CompareTo(other);

            public bool Equals(ITokenDefinition other) =>
                other is not null
                && Id == other.Id
                && Value.Equals(other.Value, StringComparison.InvariantCulture)
                && Key.Equals(other.Key, StringComparison.InvariantCulture)
                && IsSpecial == other.IsSpecial;

            public int CompareTo(ITokenDefinition other)
            {
                var keyComp = string.Compare(Key, other.Key, StringComparison.Ordinal);
                return keyComp == 0 ? IsSpecial.CompareTo(other.IsSpecial) : keyComp;
            }

            public override bool Equals(object obj) =>
                obj is ITokenDefinition other && Equals(other);

            public override int GetHashCode() => Id;

            public override string ToString() => m_Repr ??= $"{(IsSpecial ? @"★" : "")}{Key}:{Id}";
        }
    }
}
