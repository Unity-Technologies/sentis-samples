using System.Collections.Generic;

namespace Unity.ML.Tokenization
{
    /// <summary>
    ///     Stores <see cref="ITokenDefinition"/> values and provides some lookup methods.
    /// </summary>
    public interface IVocabulary
    {
        /// <summary>
        ///     Tries to find a <see cref="ITokenDefinition" /> from its
        ///     <see cref="ITokenDefinition.Key" />.
        /// </summary>
        /// <param name="key">
        ///     The key of the <see cref="ITokenDefinition" /> instance to look for.
        /// </param>
        /// <param name="definition">
        ///     The token definition, if found.
        /// </param>
        /// <returns>
        ///     Whether an <see cref="ITokenDefinition" /> instance has been found.
        /// </returns>
        bool TryGetToken(SubString key, out ITokenDefinition definition, SubString? prefix = null);

        /// <summary>
        ///     Tries to find a <see cref="ITokenDefinition" /> from its
        ///     <see cref="ITokenDefinition.Id" />.
        /// </summary>
        /// <param name="id">
        ///     The id of the <see cref="ITokenDefinition" /> instance to look for.
        /// </param>
        /// <param name="definition">
        ///     The token definition, if found.
        /// </param>
        /// <returns>
        ///     Whether an <see cref="ITokenDefinition" /> instance has been found.
        /// </returns>
        bool TryGetToken(int id, out ITokenDefinition definition);

        IEnumerable<ITokenDefinition> Definitions { get; }
    }
}
