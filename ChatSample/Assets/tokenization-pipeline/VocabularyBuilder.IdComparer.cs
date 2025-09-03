using System.Collections.Generic;

namespace Unity.ML.Tokenization
{
    partial class VocabularyBuilder
    {
        class IdComparer : IEqualityComparer<TokenDefinition>
        {
            public bool Equals(TokenDefinition x, TokenDefinition y) => x!.Id == y!.Id;
            public int GetHashCode(TokenDefinition obj) => obj.Id;
        }
    }
}
