using System.Collections.Generic;

namespace Unity.ML.Tokenization
{
    partial class VocabularyBuilder
    {
        class KeyComparer : IEqualityComparer<TokenDefinition>
        {
            public bool Equals(TokenDefinition x, TokenDefinition y) => x!.Key == y!.Key;
            public int GetHashCode(TokenDefinition obj) => obj.Key.GetHashCode();
        }
    }
}
