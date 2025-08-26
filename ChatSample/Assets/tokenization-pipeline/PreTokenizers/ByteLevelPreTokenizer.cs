using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Unity.ML.Tokenization.PreTokenizers
{
    /// <summary>
    ///     Pre tokenize an input using ByteLevel rules.
    /// </summary>
    public partial class ByteLevelPreTokenizer : PreTokenizerBase
    {
        IOneToManyConverter<SubString, SubString> m_InputSplitter;
        Pool<StringBuilder> m_StringBuilderPool;
        IOneToManyConverter<SubString, byte> m_SubStringToBytes;
        IOneToManyConverter<SubString, SubString> m_Utf8CharSplitter;

        /// <summary>
        ///     Initializes a new instance of the <see cref="ByteLevelPreTokenizer" /> type.
        /// </summary>
        /// <param name="addPrefixSpace">
        ///     Adds a whitespace at the beginning of the input if it doesn't start with one.
        /// </param>
        /// <param name="gpt2Regex">
        ///     Uses the GPT2 regex to split the input into smaller <see cref="SubString" />s.
        /// </param>
        public ByteLevelPreTokenizer(bool addPrefixSpace = true, bool gpt2Regex = true)
        {
            Init(
                gpt2Regex ? new Gpt2Splitter() : new DefaultSplitter(), Utf8CharSplitter.Instance,
                new OneToManyCachedConverter<SubString, byte>(
                    SubStringToByteConverter.Instance, SubString.Comparer),
                PoolUtility.GetStringBuilderPool(), addPrefixSpace);
        }

        internal ByteLevelPreTokenizer(
            IOneToManyConverter<SubString, SubString> splitter,
            IOneToManyConverter<SubString, SubString> stringToUtf8Chars,
            IOneToManyConverter<SubString, byte> stringToBytes,
            Pool<StringBuilder> stringBuilderPool,
            bool addPrefixSpace)
        {
            Init(splitter, stringToUtf8Chars, stringToBytes, stringBuilderPool, addPrefixSpace);
        }

        /// <summary>
        ///     Adds a whitespace at the beginning of the input if it doesn't start with one.
        /// </summary>
        public bool AddPrefixSpace { get; private set; }

        void Init(
            IOneToManyConverter<SubString, SubString> inputSplitter,
            IOneToManyConverter<SubString, SubString> utf8CharsSplitter,
            IOneToManyConverter<SubString, byte> subStringToBytes,
            Pool<StringBuilder> stringBuilderPool,
            bool addPrefixSpace)
        {
            m_InputSplitter = inputSplitter;
            m_Utf8CharSplitter = utf8CharsSplitter;
            m_SubStringToBytes = subStringToBytes;
            m_StringBuilderPool = stringBuilderPool;

            AddPrefixSpace = addPrefixSpace;
        }

        /// <inheritdoc />
        protected override void PreTokenizeInternal(SubString input, IOutput<SubString> output)
        {
            if (AddPrefixSpace && !input.StartsWith(" "))
                input = $" {input}";

            using var splitOutputHandle =
                PoolUtility.GetOutputOfSubStringPool().Get(out var sOutput);
            m_InputSplitter.Convert(input, sOutput);

            foreach (var subString in sOutput)
            {
                using var utfCharHandle =
                    PoolUtility.GetOutputOfSubStringPool().Get(out var splitOutput);
                m_Utf8CharSplitter.Convert(subString, splitOutput);

                using var byteHandle = PoolUtility.GetOutputOfBytePool().Get(out var byteOutput);
                foreach (var split in splitOutput)
                    m_SubStringToBytes.Convert(split, byteOutput);

                using var _ = m_StringBuilderPool.Get(out var builder);
                foreach (var b in byteOutput)
                    builder.Append(ByteLevelHelper.BytesChars[b]);

                output.Add(builder.ToString());
            }
        }
    }
}
