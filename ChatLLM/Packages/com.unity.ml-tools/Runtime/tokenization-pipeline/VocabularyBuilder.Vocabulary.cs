using System.Collections.Generic;

namespace Unity.ML.Tokenization
{
    partial class VocabularyBuilder
    {
        class Vocabulary : IVocabulary
        {
            readonly Dictionary<int, TokenDefinition> m_DefinitionsById;
            readonly Dictionary<int, List<TokenDefinition>> m_DefinitionsByValue;

            public Vocabulary(IEnumerable<TokenDefinition> definitions)
            {
                m_DefinitionsById = new();
                m_DefinitionsByValue = new();

                foreach (var definition in definitions)
                {
                    var hashCode = new SubString(definition.Key).GetHashCode();

                    if (!m_DefinitionsByValue.TryGetValue(hashCode, out var listDefinitions))
                    {
                        listDefinitions = new();
                        m_DefinitionsByValue.Add(hashCode, listDefinitions);
                    }

                    listDefinitions.Add(definition);

                    m_DefinitionsById.Add(definition.Id, definition);
                }
            }

            public IEnumerable<ITokenDefinition> Definitions => m_DefinitionsById.Values;

            public bool TryGetToken(
                SubString value,
                out ITokenDefinition definition,
                SubString? prefix = null)
            {
                definition = default;

                var hashCode = value.GetHashCode(prefix, default);

                if (!m_DefinitionsByValue.TryGetValue(hashCode, out var listDefinitions))
                    return false;

                foreach (var candidate in listDefinitions)
                {
                    if (!Compare(candidate.Key, value, prefix))
                        continue;

                    definition = candidate;
                    return true;
                }

                return false;

                bool Compare(SubString a, SubString b, SubString? prefixB)
                {
                    var ai = 0;

                    if (prefixB.HasValue)
                    {
                        var pb = prefixB.Value;
                        if (pb.Length > a.Length)
                            return false;

                        for (int pbi = 0, pbl = pb.Length; pbi < pbl; pbi++)
                        {
                            if (a[ai++] != pb[pbi])
                                return false;
                        }
                    }

                    if (a.Length - ai != b.Length)
                        return false;

                    for (int bi = 0, bl = b.Length; bi < bl; bi++)
                    {
                        if (a[ai++] != b[bi])
                            return false;
                    }

                    return true;
                }
            }

            public bool TryGetToken(int id, out ITokenDefinition definition)
            {
                var found = m_DefinitionsById.TryGetValue(id, out var def);
                definition = def;
                return found;
            }

            bool IVocabulary.TryGetToken(
                SubString key,
                out ITokenDefinition definition,
                SubString? prefix) =>
                TryGetToken(key, out definition, prefix);

            bool IVocabulary.TryGetToken(int id, out ITokenDefinition definition) =>
                TryGetToken(id, out definition);

            IEnumerable<ITokenDefinition> IVocabulary.Definitions => Definitions;
        }
    }
}
