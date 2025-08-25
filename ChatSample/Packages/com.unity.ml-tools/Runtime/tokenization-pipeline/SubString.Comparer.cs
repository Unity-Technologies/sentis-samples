using System.Collections.Generic;

namespace Unity.ML.Tokenization
{
    partial struct SubString
    {
        public static IEqualityComparer<SubString> Comparer = new ComparerImpl();

        class ComparerImpl : IEqualityComparer<SubString>
        {
            public bool Equals(SubString x, SubString y) => x.Equals(y);

            public int GetHashCode(SubString subString) => subString.GetHashCode();
        }
    }
}
