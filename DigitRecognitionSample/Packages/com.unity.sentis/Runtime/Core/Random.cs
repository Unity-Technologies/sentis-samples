using System;
using System.Runtime.InteropServices;

namespace Unity.Sentis
{
    /// <summary>
    /// Helper struct class for converting between ints, uints and floats in bytes without allocation
    /// </summary>
    [StructLayout(LayoutKind.Explicit)]
    struct Seed
    {
        [FieldOffset(0)]
        public int intSeed;
        [FieldOffset(0)]
        public uint uintSeed;
        [FieldOffset(0)]
        public float floatSeed;
    }

    /// <summary>
    /// Represents a pseudo-random number generator used by Sentis.
    /// </summary>
    public class Random
    {
        /// <summary>
        /// Static global System.Random used for random values when no seed provided
        /// </summary>
        static System.Random s_Random = new System.Random();

        /// <summary>
        /// Sets the global Sentis random state for random values without an explicit seed.
        /// </summary>
        /// <param name="seed">The seed to set the state to</param>
        public static void SetSeed(int seed)
        {
            s_Random = new System.Random(seed);
        }

        // Local System.Random used for random values when seed is provided
        System.Random m_Random;

        // Returns either local or global System.Random corresponding to given seed or not
        System.Random SystemRandom => m_Random ?? s_Random;

        internal Random() { }

        internal Random(float seed)
        {
            m_Random = new System.Random(new Seed { floatSeed = seed }.intSeed);
        }

        // Returns float with random bytes to be used as seed for Random Op
        internal float NextFloatSeed()
        {
            return new Seed { intSeed = SystemRandom.Next(int.MinValue, int.MaxValue) }.floatSeed;
        }

        // Returns uint with random bytes to be used as seed inside Op and be passed to a job or as a seed for Mathematics.Random
        internal static uint GetOpSeed(float? seed)
        {
            return seed.HasValue ? new Seed { floatSeed = seed.Value }.uintSeed : new Seed { intSeed = s_Random.Next(int.MinValue, int.MaxValue) }.uintSeed;
        }
    }
}
