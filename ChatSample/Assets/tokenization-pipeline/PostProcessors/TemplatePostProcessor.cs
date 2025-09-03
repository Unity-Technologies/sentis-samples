using System;
using System.Collections.Generic;
using Unity.ML.Tokenization.PostProcessors.Templating;

namespace Unity.ML.Tokenization.PostProcessors
{
    /// <summary>
    ///     Post processor using the templating approach.
    /// </summary>
    public class TemplatePostProcessor : PostProcessorBase
    {
        static void CheckTemplate(
            Template template,
            string nameofTemplate,
            IReadOnlyDictionary<string, ITokenDefinition> specialTokens,
            bool isSingle)
        {
            var sequenceAFound = false;
            var sequenceBFound = false;

            foreach (var piece in template.Pieces)
                switch (piece)
                {
                    case SpecialToken specialToken
                        when !specialTokens.ContainsKey(specialToken.Value):
                        throw new KeyNotFoundException(
                            $"Token {specialToken.Value} in template {nameofTemplate} not found in the special tokens list.");
                    case Sequence sequence:
                    {
                        sequenceAFound |= sequence.Identifier is SequenceIdentifier.A;
                        sequenceBFound |= sequence.Identifier is SequenceIdentifier.B;
                        break;
                    }
                }

            if (!sequenceAFound)
                throw new FormatException(
                    "Sequence B cannot be used in a single sequence template.");

            switch (isSingle)
            {
                case true when sequenceBFound:
                    throw new FormatException(
                        "Sequence B cannot be used in a single sequence template.");
                case false when !sequenceBFound:
                    throw new FormatException("Sequence B must appears in the template.");
            }
        }

        readonly Template m_PairSequenceTemplate;
        readonly Template m_SingleSequenceTemplate;
        readonly Dictionary<string, ITokenDefinition> m_SpecialTokens;

        /// <summary>
        ///     Initializes a new instance of the <see cref="TemplatePostProcessor"/> type.
        /// </summary>
        /// <param name="single">
        ///     <see cref="Template" /> for processing of single sequence.
        /// </param>
        /// <param name="pair">
        ///     <see cref="Template"/> for processing of paired sequences.
        /// </param>
        /// <param name="specialTokens">
        ///     Special tokens used in the templates.
        /// </param>
        public TemplatePostProcessor(
            Template single,
            Template pair,
            IEnumerable<ITokenDefinition> specialTokens)
        {
            m_SpecialTokens = new();

            foreach (var token in specialTokens)
                m_SpecialTokens[token.Key] = token;

            if (single is not null)
            {
                CheckTemplate(single, nameof(single), m_SpecialTokens, true);
                m_SingleSequenceTemplate = single;
            }

            if (pair is not null)
            {
                CheckTemplate(pair, nameof(pair), m_SpecialTokens, false);
                m_PairSequenceTemplate = pair;
            }
        }

        /// <inheritdoc />
        public override int GetNumAddedTokens(bool isPair) => 2;

        /// <inheritdoc />
        protected override void PostProcessInternal(
            IEnumerable<int> tokensA,
            IEnumerable<int> tokensB,
            bool addSpecialTokens,
            IOutput<int> output)
        {
            var template = tokensB is not null ? m_PairSequenceTemplate : m_SingleSequenceTemplate;
            PostProcess(tokensA, tokensB, template, addSpecialTokens, output);
        }

        void PostProcess(
            IEnumerable<int> tokensA,
            IEnumerable<int> tokensB,
            Template template,
            bool addSpecialTokens,
            IOutput<int> output)
        {
            foreach (var piece in template.Pieces)
                switch (piece)
                {
                    case Sequence sequence:
                        var tokens = sequence.Identifier == SequenceIdentifier.A
                            ? tokensA
                            : tokensB;
                        if (tokens is not null)
                            output.Add(tokens);
                        break;
                    case SpecialToken specialToken:
                        if (addSpecialTokens)
                            output.Add(m_SpecialTokens[specialToken.Value].Id);
                        break;
                }
        }
    }
}
