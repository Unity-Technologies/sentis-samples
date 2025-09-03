using System;
using System.Collections.Generic;
using System.Linq;
using JetBrains.Annotations;

namespace Unity.ML.Tokenization.Tokenizers
{
    /// <summary>
    ///     Turns a string input into a sequence of <see cref="ITokenDefinition" /> instances using
    ///     the Byte-Pair Encoding strategy.
    /// </summary>
    public partial class BpeTokenizer : TokenizerBase
    {
        /// <summary>
        ///     The converter in charge of optimizing the output of the <see cref="m_Tokenizer" />
        ///     using the merge rules.
        /// </summary>
        IManyToManyConverter<ITokenDefinition, ITokenDefinition> m_Merger;

        /// <summary>
        ///     The converter in charge of turning each character of the string into an instance of
        ///     <see cref="ITokenDefinition" /> using the given vocabulary.
        /// </summary>
        IOneToManyConverter<SubString, ITokenDefinition> m_Tokenizer;

        IVocabulary m_Vocabulary;

        /// <summary>
        ///     Convert a substring into a sequence of <see cref="ITokenDefinition" /> instances
        ///     using the Byte-Pair Encoding strategy.
        /// </summary>
        /// <param name="vocabulary">
        ///     The map associating token string representation with their ids.
        /// </param>
        /// <param name="merges">
        ///     The list of mergeable token pairs, ordered by priority.
        /// </param>
        /// <param name="unknownConfig">
        ///     The config for unknown tokens.
        ///     <list>
        ///         <item>
        ///             <term><c>token</c></term>
        ///             <description>
        ///                 the string representation of the token to use.
        ///             </description>
        ///         </item>
        ///         <item>
        ///             <term><c>fuse</c></term>
        ///             <description>
        ///                 tells whether the sequence of unknown tokens is fused.
        ///             </description>
        ///         </item>
        ///         <item>
        ///             <term><c>byteFallback</c></term>
        ///             <description>
        ///                 tells if tokenized byte must be searched before using the unknown token.
        ///             </description>
        ///         </item>
        ///     </list>
        /// </param>
        /// <param name="decoratorConfig">
        ///     Decorates token string representation.
        ///     <list>
        ///         <item>
        ///             <term><c>subWordPrefix</c></term>
        ///             <description>
        ///                 string prepended to a token value representing a piece of a word while
        ///                 not being at the beginning of it.
        ///             </description>
        ///         </item>
        ///         <item>
        ///             <term><c>wordSuffix</c></term>
        ///             <description>
        ///                 string appended to a token value representing a piece of a word which
        ///                 being at the end of it.
        ///             </description>
        ///         </item>
        ///     </list>
        /// </param>
        public BpeTokenizer(
            [NotNull] IVocabulary vocabulary,
            [CanBeNull] IEnumerable<(string a, string b)> merges = null,
            [CanBeNull] (string token, bool fuse, bool byteFallback)? unknownConfig = default,
            [CanBeNull] (string subWordPrefix, string wordSuffix)? decoratorConfig = default)
        {
            m_Vocabulary = vocabulary ?? throw new ArgumentNullException(nameof(vocabulary));

            // building the unknown token configuration.

            var unknown = (token: default(ITokenDefinition), unknownConfig?.fuse ?? false,
                unknownConfig?.byteFallback ?? false);

            if (unknownConfig?.token is not null)
            {
                if (!vocabulary.TryGetToken(unknownConfig.Value.token, out var definition) || !definition.IsSpecial)
                    throw new ArgumentOutOfRangeException(
                        nameof(unknownConfig.Value.token), unknownConfig.Value.token, null);

                unknown.token = definition;
            }

            // building the token decorator configuration.

            var decorator = (subWordPrefix: string.Empty, wordSuffix: string.Empty);

            if (decoratorConfig.HasValue)
            {
                if (decoratorConfig.Value.subWordPrefix != null)
                    decorator.subWordPrefix = decoratorConfig.Value.subWordPrefix;

                if (decoratorConfig.Value.wordSuffix != null)
                    decorator.wordSuffix = decoratorConfig.Value.wordSuffix;
            }

            // creating the default tokenizer
            var stringToTokenSequence = new InternalTokenizer(vocabulary, unknown, decorator);

            // creating the merger
            var merger = BuildMerger(vocabulary, merges, decorator);

            Init(vocabulary, stringToTokenSequence, merger);
        }

        /// <summary>
        ///     This constructor is used for unit testing.
        /// </summary>
        /// <param name="vocabulary">
        ///     The ID &lt;-> Value map.
        /// </param>
        /// <param name="tokenizer">
        ///     An implementation of the string->token conversion.
        /// </param>
        /// <param name="merger">
        ///     An implementation of the token merging process.
        /// </param>
        internal BpeTokenizer(
            IVocabulary vocabulary,
            IOneToManyConverter<SubString, ITokenDefinition> tokenizer,
            IManyToManyConverter<ITokenDefinition, ITokenDefinition> merger)
        {
            Init(vocabulary, tokenizer, merger);
        }

        /// <summary>
        ///     Builds the merger.
        /// </summary>
        /// <param name="vocabulary">
        ///     The value->ids map of token definitions.
        /// </param>
        /// <param name="merges">
        ///     The list of mergeable pairs, ordered from the most frequent to the rarest.
        /// </param>
        /// <param name="decorator">
        ///     Configuration for the token string representation.
        ///     <ul>
        ///         <li>
        ///             <c>subWordPrefix</c>: string prepended to a token value representing a piece
        ///             of a word while not being at the beginning of it.
        ///         </li>
        ///         <li>
        ///             <c>wordSuffix</c>: string appended to a token value representing a piece of
        ///             a word which being at the end of it.
        ///         </li>
        ///     </ul>
        /// </param>
        /// <returns>
        ///     The merger instance.
        /// </returns>
        /// <exception cref="ArgumentException">
        /// </exception>
        static IManyToManyConverter<ITokenDefinition, ITokenDefinition> BuildMerger(
            IVocabulary vocabulary,
            IEnumerable<(string a, string b)> merges,
            (string subWordPrefix, string wordSuffix) decorator)
        {
            IManyToManyConverter<ITokenDefinition, ITokenDefinition> merger;

            // If no merge rules, returning an instance of DefaultMerger, which does nothing.
            if (merges is null)
            {
                merger = new DefaultMerger();
            }
            else
            {
                var mergeDefinitions = merges.Select(
                    (t, rank) =>
                    {
                        if (!vocabulary.TryGetToken(t.a, out var a) || a.IsSpecial)
                            throw new ArgumentException(
                                $"Token {t.a} not found in the vocabulary", nameof(merges));

                        if (!vocabulary.TryGetToken(t.b, out var b) || b.IsSpecial)
                            throw new ArgumentException(
                                $"Token {t.b} not found in the vocabulary", nameof(merges));

                        var mergedKey = string.Concat(
                            a.Key, b.Key[decorator.subWordPrefix.Length..]);
                        if (!vocabulary.TryGetToken(mergedKey, out var mergedToken) || mergedToken.IsSpecial)
                            throw new ArgumentException(
                                $"Merged key '{mergedKey}' not found in the vocabulary");

                        return (a, b, mergedToken, rank);
                    });

                merger = new Merger(mergeDefinitions);
            }

            return merger;
        }

        /// <summary>
        ///     Initializes the <see cref="BpeTokenizer" /> instance.
        /// </summary>
        /// <param name="vocabulary">
        ///     The ID &lt;-> Value map
        /// </param>
        /// <param name="tokenizer">
        ///     An implementation of the string->token conversion.
        /// </param>
        /// <param name="merger">
        ///     An implementation of the token merging process.
        /// </param>
        void Init(
            IVocabulary vocabulary,
            IOneToManyConverter<SubString, ITokenDefinition> tokenizer,
            IManyToManyConverter<ITokenDefinition, ITokenDefinition> merger)
        {
            m_Vocabulary = vocabulary;
            m_Tokenizer = tokenizer;
            m_Merger = merger;
        }

        /// <inheritdoc />
        protected override void TokenizeInternal(IEnumerable<SubString> inputs, IOutput<int> output)
        {
            using var definitionOutputHandle =
                PoolUtility.GetOutputOfTokenDefinitionPool().Get(out var tokenizerOutput);

            using var mergeOutputHandle =
                PoolUtility.GetOutputOfTokenDefinitionPool().Get(out var mergeOutput);

            foreach(var input in inputs)
                m_Tokenizer.Convert(input, tokenizerOutput);

            m_Merger.Convert(tokenizerOutput, mergeOutput);

            output.Add(mergeOutput.Select(definition => definition.Id));
        }

        /// <inheritdoc />
        protected override void
            DeTokenizeInternal(IEnumerable<int> input, bool skipSpecialTokens, IOutput<string> output)
        {
            foreach (var id in input)
            {
                var found = m_Vocabulary.TryGetToken(id, out var token);
                if (!found || skipSpecialTokens && token.IsSpecial)
                    continue;

                output.Add(token.Value);
            }
        }
    }
}
