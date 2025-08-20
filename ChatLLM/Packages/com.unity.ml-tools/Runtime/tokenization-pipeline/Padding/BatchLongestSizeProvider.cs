using System.Collections.Generic;
using System.Linq;

namespace Unity.ML.Tokenization.Padding
{
    public class BatchLongestSizeProvider : IPaddingSizeProvider
    {
        public int GetPaddingSize(IEnumerable<int> sizes) => sizes.Max();
    }
}
