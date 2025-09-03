using System;
using System.Collections.Generic;
using System.Linq;
using JetBrains.Annotations;
using Unity.ML.Tokenization.Decoders;
using Unity.ML.Tokenization.Normalizers;
using Unity.ML.Tokenization.Padding;
using Unity.ML.Tokenization.PostProcessors;
using Unity.ML.Tokenization.PreTokenizers;
using Unity.ML.Tokenization.Tokenizers;
using Unity.ML.Tokenization.Truncators;

namespace Unity.ML.Tokenization
{
    /// <summary>
    ///     This type is the entry point of the tokenization/detokenization pipeline.
    ///     The pipeline is composed of six steps, and turns an input string into an
    ///     <see cref="IEncoding" /> chain:
    ///     <list type="number">
    ///         <item>
    ///             <term>Pretokenization</term>
    ///             <description>
    ///                 Splits the result of the normalization step into small pieces (example:
    ///                 split by whitespace).
    ///             </description>
    ///         </item>
    ///         <item>
    ///             <term>Encoding</term>
    ///             <description>
    ///                 Central step of the tokenization, this one turns each piece from the
    ///                 pretokenizaztion process into sequence of <see cref="int" /> ids.
    ///                 See <see cref="ITokenizer" /> for more details.
    ///             </description>
    ///         </item>
    ///         <item>
    ///             <term>Truncation</term>
    ///             <description>
    ///                 Splits the sequence of ids from the encoding step into smaller subsequences.
    ///                 The most frequent truncation rule in "max length".
    ///                 See <see cref="ITruncator" /> for more details.
    ///             </description>
    ///         </item>
    ///         <item>
    ///             <term>Postprocessing</term>
    ///             <description>
    ///                 Transforms each subsequences of generated from the truncation.
    ///                 The most common transformation is adding <c>[CLS]</c> and <c>[SEP]</c>
    ///                 tokens before and after the sequence.
    ///                 See <see cref="IPostProcessor" /> for more details.
    ///             </description>
    ///         </item>
    ///         <item>
    ///             <term>Padding</term>
    ///             <description>
    ///                 Pads each subsequence from the postprocessing to match the expected sequence
    ///                 size.
    ///             </description>
    ///         </item>
    ///     </list>
    /// </summary>
    public partial class TokenizationPipeline : ITokenizationPipeline
    {
        readonly Pool<List<(int, int)>> m_ListOfIntIntPool;
        readonly Pool<List<int>> m_ListOfIntPool;
        readonly Pool<List<List<(int, int)>>> m_ListOfListOfIntIntPool;
        readonly Pool<List<List<int>>> m_ListOfListOfIntPool;
        readonly Pool<OutputCollection<(int, int)>> m_OutputCollectionOfIntIntPool;
        readonly Pool<Output<string>> m_OutputOfStringPool;
        readonly Pool<Output<int>> m_OutputOfIntPool;
        readonly Pool<OutputCollection<int>> m_OutputCollectionOfIntPool;

        readonly Pool<List<(SubString chunk, ITokenDefinition token)>> m_ProcessableChunkPool;

        readonly IVocabulary m_Vocabulary;

        readonly ITokenizer m_Tokenizer;
        readonly INormalizer m_Normalizer;
        readonly IPreTokenizer m_PreTokenizer;

        readonly IPadding m_Padding;

        readonly IPostProcessor m_PostProcessor;
        readonly ITruncator m_Truncator;
        readonly IDecoder m_Decoder;

        /// <summary>
        ///     Initializes a new instance of the <see cref="TokenizationPipeline" /> type.
        /// </summary>
        /// <param name="tokenizer">
        ///     The <see cref="ITokenizer" /> encoding to use to turn the strings into tokens.
        /// </param>
        /// <param name="normalizer">
        ///     Normalizes portions of the input.
        /// </param>
        /// <param name="preTokenizer">
        ///     The pretokenization rules.
        /// </param>
        /// <param name="postProcessor">
        ///     The post processing of the token sequence.
        ///     See <see cref="IPostProcessor" />.
        /// </param>
        /// <param name="truncator">
        ///     The truncation rules.
        ///     See <see cref="ITruncator" />.
        /// </param>
        /// <param name="paddingProcessor">
        ///     The padding rules.
        /// </param>
        /// <param name="decoder">
        ///     Modifiers applied to the decoded token sequence.
        /// </param>
        /// <param name="vocabulary">
        ///     The ID &lt;-> Value map.
        /// </param>
        /// <exception cref="ArgumentNullException">
        ///     <paramref name="tokenizer" /> cannot be <see langword="null" />.
        /// </exception>
        public TokenizationPipeline(
            [NotNull] ITokenizer tokenizer,
            [CanBeNull] INormalizer normalizer = null,
            [CanBeNull] IPreTokenizer preTokenizer = null,
            [CanBeNull] IPostProcessor postProcessor = null,
            [CanBeNull] ITruncator truncator = null,
            [CanBeNull] IPadding paddingProcessor = null,
            [CanBeNull] IDecoder decoder = null,
            [CanBeNull] IVocabulary vocabulary = null)
        {
            m_Vocabulary = vocabulary;

            m_Tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));

            m_Normalizer = normalizer ?? new DefaultNormalizer();
            m_PreTokenizer = preTokenizer ?? new DefaultPreTokenizer();
            m_PostProcessor = postProcessor ?? new DefaultPostProcessor();
            m_Truncator = truncator ?? new DefaultTruncator();
            m_Padding = paddingProcessor ?? new DefaultPadding();
            m_Decoder = decoder ?? new DefaultDecoder();

            // Pools initialization
            {
                m_ListOfIntPool = PoolUtility.GetListOfIntPool();
                m_ListOfIntIntPool = new(() => new(), list => list.Clear());
                m_ListOfListOfIntPool = new(() => new(), list => list.Clear());
                m_ListOfListOfIntIntPool = new(() => new(), list => list.Clear());

                m_OutputOfIntPool = PoolUtility.GetOutputOfIntPool();
                m_OutputOfStringPool = PoolUtility.GetOutputOfStringPool();
                m_OutputCollectionOfIntPool = new(() => new(), o => o.Reset());
                m_OutputCollectionOfIntIntPool = new(() => new(), o => o.Reset());

                m_ProcessableChunkPool = new(() => new(), list => list.Clear());
            }
        }

        /// <inheritdoc cref="ITokenizationPipeline.Encode" />
        public Encoding Encode(
            [NotNull] string inputA,
            [CanBeNull] string inputB = default,
            bool addSpecialTokens = true)
        {
            if (inputA is null)
                throw new ArgumentNullException(nameof(inputA));

            return EncodeInternal(inputA, inputB, addSpecialTokens);
        }

        /// <inheritdoc cref="ITokenizationPipeline.Decode" />
        public string Decode([NotNull] IEnumerable<int> input, bool skipSpecialTokens = false)
        {
            if (input is null)
                throw new ArgumentNullException(nameof(input));

            using var tokenizerOutputHandle = m_OutputOfStringPool.Get(out var tokenizerOutput);
            m_Tokenizer.DeTokenize(input, skipSpecialTokens, tokenizerOutput);

            using var finalHandle = m_OutputOfStringPool.Get(out var final);
            m_Decoder.Decode(tokenizerOutput, final);

            return string.Concat(final);
        }

        /// <inheritdoc cref="ITokenizationPipeline.Encode" />
        Encoding EncodeInternal(
            [NotNull] string inputA,
            [CanBeNull] string inputB,
            bool addSpecialTokens)
        {
            var isPair = inputB is not null;

            // 1. Tokenization
            using var sequenceAHandle = m_OutputOfIntPool.Get(out var sequenceA);
            using var sequenceBHandle = m_OutputOfIntPool.Get(out var sequenceB);

            TokenizeInput(inputA, sequenceA);

            if (isPair)
                TokenizeInput(inputB, sequenceB);

            // 2. Truncation
            using var truncatedAHandle = m_ListOfListOfIntPool.Get(out var truncatedA);
            using var truncatedBHandle = m_ListOfListOfIntPool.Get(out var truncatedB);

            Truncate(
                sequenceA, isPair ? sequenceB : null, addSpecialTokens, truncatedA, truncatedB,
                m_ListOfIntPool.Get);

            // 3. Post Processing
            using var tokenHandle = m_ListOfListOfIntPool.Get(out var tokens);

            PostProcess(
                truncatedA, isPair ? truncatedB : default, addSpecialTokens, tokens,
                m_ListOfIntPool.Get);

            // truncatedA content can be released.
            foreach (var list in truncatedA)
                m_ListOfIntPool.Release(list);

            // truncatedB content can be released.
            if (isPair)
                foreach (var list in truncatedB)
                    m_ListOfIntPool.Release(list);

            // 4. Padding
            using var paddedHandle = m_ListOfListOfIntIntPool.Get(out var padded);

            Pad(tokens, padded, m_ListOfIntIntPool.Get);

            // tokens content can be released.
            foreach (var token in tokens)
                m_ListOfIntPool.Release(token);

            using var handleIds = m_ListOfIntPool.Get(out var ids);
            using var handleAttention = m_ListOfIntPool.Get(out var attentions);

            Encoding head = null;
            Encoding parent = null;

            foreach (var sequence in padded)
            {
                foreach (var (id, attention) in sequence)
                {
                    ids.Add(id);
                    attentions.Add(attention);
                }

                var encoding = new Encoding(ids.ToArray(), attentions.ToArray());

                ids.Clear();
                attentions.Clear();

                if (parent is not null)
                    parent.SetOverflow(encoding);
                else
                    head = encoding;

                parent = encoding;
            }

            // padded content can be released.
            foreach (var list in padded)
                m_ListOfIntIntPool.Release(list);

            return head;

            void TokenizeInput(string input, IOutput<int> output)
            {
                using var _ = m_ProcessableChunkPool.Get(out var chunks);
                using var __ = m_ProcessableChunkPool.Get(out var processed);
                chunks.Add((input, null));

                // 1. Find chunks to process
                if (m_Vocabulary is not null)
                {
                    var specials = m_Vocabulary.Definitions.Where(def => def.IsSpecial);

                    foreach (var special in specials)
                    {
                        foreach (var (chunk, token) in chunks)
                        {
                            if (token != null)
                            {
                                processed.Add((chunk, token));
                                continue;
                            }

                            var startAt = 0;
                            var foundAt = chunk.IndexOf(special.Key, startAt);

                            while (foundAt >= startAt)
                            {
                                if (startAt < foundAt)
                                {
                                    var processableChunk = chunk.Sub(startAt, foundAt - startAt);
                                    processed.Add((processableChunk, null));
                                }

                                processed.Add((special.Key, special));

                                startAt = foundAt + special.Key.Length;

                                foundAt = chunk.IndexOf(special.Key, startAt);
                            }

                            if (startAt < chunk.Length)
                            {
                                var payload = chunk.Sub(startAt, chunk.Length - startAt);
                                processed.Add((payload, null));
                            }
                        }

                        (chunks, processed) = (processed, chunks);
                        processed.Clear();
                    }
                }


                // 2. Normalization + Pre Tokenization
                using var tempOutputHandle = PoolUtility.GetOutputOfSubStringPool().Get(out var tempOutput);
                foreach (var (chunk, token) in chunks)
                {
                    if (token != null)
                    {
                        output.Add(token.Id);
                        continue;
                    }

                    var normalizedChunk = m_Normalizer.Normalize(chunk);
                    m_PreTokenizer.PreTokenize(normalizedChunk, tempOutput);
                    m_Tokenizer.Tokenize(tempOutput, output);
                    tempOutput.Reset();
                }
            }

            void PostProcess(
                IEnumerable<List<int>> pInputA,
                IEnumerable<List<int>> pInputB,
                bool pAddSpecialTokens,
                ICollection<List<int>> output,
                Func<List<int>> getList)
            {
                using var tokenOutputHandle = m_OutputCollectionOfIntPool.Get(out var tokenOutput);
                tokenOutput.Init(output, getList);

                m_PostProcessor.PostProcess(pInputA, pInputB, pAddSpecialTokens, tokenOutput);
            }

            void Truncate(
                IEnumerable<int> pInputA,
                IEnumerable<int> pInputB,
                bool pAddSpecialTokens,
                ICollection<List<int>> outputA,
                ICollection<List<int>> outputB,
                Func<List<int>> getList)
            {
                using var truncationOutputAHandle =
                    m_OutputCollectionOfIntPool.Get(out var truncationOutputA);
                truncationOutputA.Init(outputA, getList);

                using var truncationOutputBHandle =
                    m_OutputCollectionOfIntPool.Get(out var truncationOutputB);
                truncationOutputB.Init(outputB, getList);

                var numAddedTokens = pAddSpecialTokens
                    ? m_PostProcessor.GetNumAddedTokens(pInputB is not null)
                    : 0;

                m_Truncator.Truncate(
                    pInputA, pInputB, numAddedTokens, truncationOutputA, truncationOutputB);
            }

            void Pad(
                IEnumerable<List<int>> input,
                ICollection<List<(int, int)>> target,
                Func<List<(int, int)>> getList)
            {
                using var paddingOutputHandle =
                    m_OutputCollectionOfIntIntPool.Get(out var paddingOutput);

                paddingOutput.Init(target, getList);

                var il = input.ToArray();
                // {
                //     var sb = new StringBuilder();
                //     foreach (var sequence in il)
                //         sb.Append("[").AppendJoin(",", sequence).Append("],");
                //     Debug.Log($"Pad Input: {sb}");
                // }
                // {
                //     var sb = new StringBuilder();
                //     foreach (var sequence in il)
                //         sb.Append("[").AppendJoin(",", sequence).Append("],");
                //     Debug.Log($"Pad Input: {sb}");
                // }

                m_Padding.Pad(il, paddingOutput);
            }
        }

        /// <inheritdoc />
        IEncoding ITokenizationPipeline.
            Encode(string inputA, string inputB, bool addSpecialTokens) =>
            Encode(inputA, inputB, addSpecialTokens);

        /// <inheritdoc />
        string ITokenizationPipeline.Decode(IEnumerable<int> input, bool skipSpecialTokens) =>
            Decode(input, skipSpecialTokens);
    }
}
