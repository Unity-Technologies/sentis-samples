using System.Collections.Generic;
using System.Text;

namespace Unity.ML.Tokenization
{
    static class PoolUtility
    {
        static Pool<StringBuilder> s_StringBuilderPool;

        static Pool<List<SubString>> s_SubStringPool;

        static Pool<List<int>> s_ListOfIntPool;

        static Pool<List<byte>> s_ListOfBytePool;

        static Pool<List<string>> s_ListOfStringPool;

        static Pool<List<Token>> s_ListOfTokenPool;

        static Pool<List<ITokenDefinition>> s_ListOfTokenDefinitionPool;

        static Pool<Output<SubString>> s_OutputOfSubStringPool;

        static Pool<Output<string>> s_OutputOfStringPool;

        static Pool<Output<int>> s_OutputOfIntPool;

        static Pool<Output<byte>> s_OutputOfBytePool;

        static Pool<Output<Token>> s_OutputOfTokenPool;

        static Pool<Output<ITokenDefinition>> s_OutputOfTokenDefinitionPool;

        public static Pool<StringBuilder> GetStringBuilderPool() =>
            s_StringBuilderPool ??= new(() => new(), sb => sb.Clear());

        public static Pool<List<SubString>> GetListOfSubStringPool() =>
            s_SubStringPool ??= new(() => new(), l => l.Clear());

        public static Pool<List<int>> GetListOfIntPool() =>
            s_ListOfIntPool ??= new(() => new(), l => l.Clear());

        public static Pool<List<Token>> GetListOfTokenPool() =>
            s_ListOfTokenPool ??= new(() => new(), l => l.Clear());

        public static Pool<List<byte>> GetListOfBytePool() =>
            s_ListOfBytePool ??= new(() => new(), l => l.Clear());

        public static Pool<List<string>> GetListOfStringPool() =>
            s_ListOfStringPool ??= new(() => new(), l => l.Clear());

        public static Pool<Output<SubString>> GetOutputOfSubStringPool() =>
            s_OutputOfSubStringPool ??= new(() => new(), o => o.Reset());

        public static Pool<Output<string>> GetOutputOfStringPool() =>
            s_OutputOfStringPool ??= new(() => new(), o => o.Reset());

        public static Pool<Output<int>> GetOutputOfIntPool() =>
            s_OutputOfIntPool ??= new(() => new(), o => o.Reset());

        public static Pool<Output<byte>> GetOutputOfBytePool() =>
            s_OutputOfBytePool ??= new(() => new(), o => o.Reset());

        public static Pool<Output<Token>> GetOutputOfTokenPool() =>
            s_OutputOfTokenPool ??= new(() => new(), o => o.Reset());

        public static Pool<Output<ITokenDefinition>> GetOutputOfTokenDefinitionPool() =>
            s_OutputOfTokenDefinitionPool ??= new(() => new(), o => o.Reset());

        public static Pool<List<ITokenDefinition>> GetListOfTokenDefinitionPool() =>
            s_ListOfTokenDefinitionPool ??= new(() => new(), l => l.Clear());

    }
}
