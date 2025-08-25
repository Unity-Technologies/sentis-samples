namespace Unity.ML.Tokenization.Tokenizers
{
    partial class BpeTokenizer
    {
        /// <summary>
        ///     Represents a token of from the output of the char->token step as a linked list
        ///     element and facilitates the merging process.
        /// </summary>
        struct Symbol
        {
            /// <summary>
            ///     The token definition of the character.
            /// </summary>
            public ITokenDefinition Definition;

            /// <summary>
            ///     The position of the symbol in the linked list.
            /// </summary>
            public int Position;

            /// <summary>
            ///     The index of the previous symbol in the linked list.
            /// </summary>
            public int Previous;

            /// <summary>
            ///     The index of the next symbol in the linked list.
            /// </summary>
            public int Next;

            /// <summary>
            ///     Tells whether this symbol is discarded, meaning that it shouldn't be used
            ///     anymore.
            /// </summary>
            public bool Discarded;
        }
    }
}
