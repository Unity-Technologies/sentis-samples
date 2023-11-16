#define MAX_THREAD_COUNT 64
#if (MAX_THREAD_COUNT>=256)
#define NUMTHREADS(t256,t128,t64) [numthreads t256]
#define NUMTHREAD(t256, t128, t64) t256
#elif (MAX_THREAD_COUNT>=128)
#define NUMTHREADS(t256,t128,t64) [numthreads t128]
#define NUMTHREAD(t256,t128,t64) t128
#elif (MAX_THREAD_COUNT>=64)
#define NUMTHREADS(t256,t128,t64) [numthreads t64]
#define NUMTHREAD(t256,t128,t64) t64
#endif

//Simulate C# pow(x<0, n is int) to avoid NaNs on GPU
//https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-pow
//https://docs.microsoft.com/en-us/dotnet/api/system.math.pow?view=net-6.0
float SignedPow(float f, float e)
{
    // handle negative f
    float v = pow(abs(f), e);
    float s = (abs(e % 2) == 1) ?
        sign(f):    // exponent is odd  => sign(f) * pow(abs(f), e)
        1;          // exponent is even => pow(abs(f), e)
    return v * s;
}

float4 SignedPow(float4 A, float4 B)
{
    float4 O;
    O.x = SignedPow(A.x, B.x);
    O.y = SignedPow(A.y, B.y);
    O.z = SignedPow(A.z, B.z);
    O.w = SignedPow(A.w, B.w);
    return O;
}

// @TODO: move all code below into a separate and appropriately named file(s)
//
#define FLT_MAX asfloat(0x7F7FFFFF) //  3.402823466 E + 38
#define FLT_MIN asfloat(0xFF7FFFFF) // -3.402823466 E + 38
#define INT_MAX 0x7FFFFFFF //  2147483647
#define INT_MIN 0x80000000 // –2147483648
#define SHAPE_MAXRANK 8
#define FLT_EPSILON 1e-6

float fastfma(float a, float b, float c)
{
    return dot(float2(a,c), float2(b, 1));
}

// Neumaier's improved Kahan–Babuška algorithm for compensated summation
// see: https://en.wikipedia.org/wiki/Kahan_summation_algorithm
float neumaierAdd(float sum, float value, inout float floatingPointAccuracyCompensation)
{
    float newSum = sum + value;
    if (abs(sum) >= abs(value))
        floatingPointAccuracyCompensation += (sum - newSum) + value;
    else
        floatingPointAccuracyCompensation += (value - newSum) + sum;
    return newSum;
}
