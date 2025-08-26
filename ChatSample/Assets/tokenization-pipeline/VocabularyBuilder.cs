using System;
using System.Collections.Generic;
using JetBrains.Annotations;

namespace Unity.ML.Tokenization
{
    /// <summary>
    ///     Assistant for <see cref="IVocabulary" /> instance creation.
    /// </summary>
    public partial class VocabularyBuilder
    {
        /// <summary>
        ///     Keeps the list of definitions, assuring that they are unique by value and special
        ///     state.
        /// </summary>
        readonly HashSet<TokenDefinition> m_DefinitionsByKey = new(new KeyComparer());

        /// <summary>
        ///     Assures that token ids are unique.
        /// </summary>
        readonly HashSet<TokenDefinition> m_DefinitionsById = new(new IdComparer());

        /// <summary>
        ///     Adds a new definition to the builder.
        /// </summary>
        /// <param name="id">
        ///     The <see cref="int" /> id of the definition.
        /// </param>
        /// <param name="key">
        ///     The <see cref="string" /> identifier of the definition.
        /// </param>
        /// <param name="value">
        ///     The <see cref="string" /> representation of the definition.
        /// </param>
        /// <param name="special">
        ///     Tells whether the token is special.
        /// </param>
        /// <returns>
        ///     <see langword="this" />
        /// </returns>
        /// <exception cref="ArgumentNullException">
        ///     <paramref name="value" /> cannot be <see langword="null" />.
        /// </exception>
        public VocabularyBuilder Add(
            int id,
            [NotNull] string key,
            [NotNull] string value,
            bool special = false)
        {
            if (key == null)
                throw new ArgumentNullException(nameof(key));

            if (value == null)
                throw new ArgumentNullException(nameof(value));

            var definition = new TokenDefinition(id, key, value, special);
            return Add(definition);
        }

        VocabularyBuilder Add(TokenDefinition definition)
        {
            if (m_DefinitionsByKey.Contains(definition) || m_DefinitionsById.Contains(definition))
                throw new($"A similar definition already exists for {definition}.");

            m_DefinitionsByKey.Add(definition);
            m_DefinitionsById.Add(definition);

            return this;
        }

        /// <summary>
        ///     Builds a <see cref="IVocabulary" /> instance with all the added definitions.
        /// </summary>
        /// <returns>
        ///     A <see cref="IVocabulary" /> instance with all the added definitions.
        /// </returns>
        public IVocabulary Build() => new Vocabulary(m_DefinitionsById);

        /// <summary>
        ///     Removes the definitions.
        /// </summary>
        /// <returns>
        ///     <see langword="this" />
        /// </returns>
        public VocabularyBuilder Clear()
        {
            m_DefinitionsByKey.Clear();
            m_DefinitionsById.Clear();
            return this;
        }
    }
}
