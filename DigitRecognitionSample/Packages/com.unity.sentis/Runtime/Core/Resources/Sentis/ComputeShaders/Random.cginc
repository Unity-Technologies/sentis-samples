// following Unity.Mathematics.Random
inline uint WangHash(uint n)
{
    // https://gist.github.com/badboy/6267743#hash-function-construction-principles
    // Wang hash: this has the property that none of the outputs will
    // collide with each other, which is important for the purposes of
    // seeding a random number generator.  This was verified empirically
    // by checking all 2^32 uints.
    n = (n ^ 61u) ^ (n >> 16);
    n *= 9u;
    n = n ^ (n >> 4);
    n *= 0x27d4eb2du;
    n = n ^ (n >> 15);

    return n;
}

inline uint4 WangHash(uint4 n)
{
    // https://gist.github.com/badboy/6267743#hash-function-construction-principles
    // Wang hash: this has the property that none of the outputs will
    // collide with each other, which is important for the purposes of
    // seeding a random number generator.  This was verified empirically
    // by checking all 2^32 uints.
    n = (n ^ 61u) ^ (n >> 16);
    n *= 9u;
    n = n ^ (n >> 4);
    n *= 0x27d4eb2du;
    n = n ^ (n >> 15);

    return n;
}

inline uint NextState(uint state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

inline uint4 NextState(uint4 state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

inline float ToFloat(uint state)
{
    return asfloat(0x3f800000 | state >> 9) - 1.0f;
}

inline float4 ToFloat4(uint4 state)
{
    return asfloat(0x3f800000 | state >> 9) - 1.0f;
}

inline float GetRandomNormal(uint state)
{
    float u, v, s;
    do {
        state = NextState(state);
        u = ToFloat(state) * 2 - 1;
        state = NextState(state);
        v = ToFloat(state) * 2 - 1;
        s = u * u + v * v;
    } while (s >= 1 || s == 0);
    return u * sqrt(-2.0f * log(s) / s);
}
