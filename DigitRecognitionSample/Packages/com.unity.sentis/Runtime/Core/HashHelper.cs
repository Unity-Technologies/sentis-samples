using System;

namespace Unity.Sentis
{
    static class HashHelper
    {
        // https://www.boost.org/doc/libs/1_55_0/doc/html/hash/combine.html
        public static void HashCombine<T>(ref long seed, T v)
        {
            seed ^= v.GetHashCode() + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
    }
}
