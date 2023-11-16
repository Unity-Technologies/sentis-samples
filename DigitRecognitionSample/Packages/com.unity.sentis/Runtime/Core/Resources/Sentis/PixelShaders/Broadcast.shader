Shader "Hidden/Sentis/Broadcast"
{
    Properties
    {
    }
    SubShader
    {
        // No culling or depth
        Cull Off ZWrite Off ZTest Always

        Pass
        {
            CGPROGRAM
            // TODO: use Scriban to generate variants
            #pragma multi_compile Add Sub Mul Div Pow Min Max FMod Mean AddInt SubInt MulInt DivInt PowInt MinInt MaxInt ModInt FModInt And Equal Greater GreaterOrEqual Less LessOrEqual EqualInt GreaterInt GreaterOrEqualInt LessInt LessOrEqualInt Or Xor PRelu

            #pragma vertex vert
            #pragma fragment frag

            #include "../ComputeShaders/Tensor.cginc"
            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            #if defined(AddInt) | defined(SubInt) | defined(MulInt) | defined(DivInt) | defined(MinInt) | defined(MaxInt) | defined(ModInt) | defined(FModInt) | defined(And) | defined(EqualInt) | defined(GreaterInt) | defined(GreaterOrEqualInt) | defined(LessInt) | defined(LessOrEqualInt) | defined(Or) | defined(Xor)
            #define O_DTYPE4 int4
            #define A_DTYPE4 int4
            #define B_DTYPE4 int4
            DECLARE_TENSOR(A, int);
            DECLARE_TENSOR(B, int);
            #elif defined(Equal) | defined(Greater) | defined(GreaterOrEqual) | defined(Less) | defined(LessOrEqual)
            #define O_DTYPE4 int4
            #define A_DTYPE4 float4
            #define B_DTYPE4 float4
            DECLARE_TENSOR(A, float);
            DECLARE_TENSOR(B, float);
            #elif defined(PowInt)
            #define O_DTYPE4 float4
            #define A_DTYPE4 float4
            #define B_DTYPE4 int4
            DECLARE_TENSOR(A, float);
            DECLARE_TENSOR(B, int);
            #else
            #define O_DTYPE4 float4
            #define A_DTYPE4 float4
            #define B_DTYPE4 float4
            DECLARE_TENSOR(A, float);
            DECLARE_TENSOR(B, float);
            #endif

            uint DimO[8];
            uint StridesA[8];
            uint StridesB[8];
            uint DimAxisA, DimAxisB;

            #ifdef Mean
            float alpha, beta;
            #endif

            bool IsInfOrNaN(float x) {
                return !(x < 0. || x > 0. || x == 0.) || (x != 0. && x * 2. == x);
            }

            O_DTYPE4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                uint blockIndexA = 0;
                uint blockIndexB = 0;
                uint n = blockIndexO;
                [unroll]
                for (uint j = 0; j < 8; j++)
                {
                    uint k = (n % DimO[j]);
                    n /= DimO[j];
                    blockIndexA += k * StridesA[j];
                    blockIndexB += k * StridesB[j];
                }

                O_DTYPE4 v = 0;

                A_DTYPE4 va = SampleBlockA(blockIndexA);
                B_DTYPE4 vb = SampleBlockB(blockIndexB);

                va = DimAxisA == 1 ? va.x : va;
                vb = DimAxisB == 1 ? vb.x : vb;

                #if defined(Add) | defined(AddInt)
                    v = va + vb;
                #endif
                #if defined(Sub) | defined(SubInt)
                    v = va - vb;
                #endif
                #if defined(Mul) | defined(MulInt)
                    v = va * vb;
                #endif
                #ifdef Div
                    float4 u = va / vb;
                    v.x = IsInfOrNaN(u.x) ? 0.0f : u.x;
                    v.y = IsInfOrNaN(u.y) ? 0.0f : u.y;
                    v.z = IsInfOrNaN(u.z) ? 0.0f : u.z;
                    v.w = IsInfOrNaN(u.w) ? 0.0f : u.w;
                #endif
                #ifdef DivInt
                    v = va / vb;
                #endif
                #if defined(Pow) | defined(PowInt)
                    O_DTYPE4 u = SignedPow(va, vb);
                    v.x = IsInfOrNaN(u.x) ? 0.0f : u.x;
                    v.y = IsInfOrNaN(u.y) ? 0.0f : u.y;
                    v.z = IsInfOrNaN(u.z) ? 0.0f : u.z;
                    v.w = IsInfOrNaN(u.w) ? 0.0f : u.w;
                #endif
                #if defined(Min) | defined(MinInt)
                    v = min(va, vb);
                #endif
                #if defined(Max) | defined(MaxInt)
                    v = max(va, vb);
                #endif
                #ifdef Mean
                    v = alpha * va + beta * vb;
                #endif
                #ifdef FMod
                    float4 u = fmod(va, vb);
                    v.x = IsInfOrNaN(u.x) ? 0.0f : u.x;
                    v.y = IsInfOrNaN(u.y) ? 0.0f : u.y;
                    v.z = IsInfOrNaN(u.z) ? 0.0f : u.z;
                    v.w = IsInfOrNaN(u.w) ? 0.0f : u.w;
                #endif
                #ifdef FModInt
                    v = va % vb;
                #endif
                #ifdef ModInt
                    v = ((va % vb) + vb) % vb;
                #endif
                #ifdef And
                    v = va & vb;
                #endif
                #if defined(Equal) | defined(EqualInt)
                    v = va == vb;
                #endif
                #if defined(Greater) | defined(GreaterInt)
                    v = va > vb;
                #endif
                #if defined(GreaterOrEqual) | defined(GreaterOrEqualInt)
                    v = va >= vb;
                #endif
                #if defined(Less) | defined(LessInt)
                    v = va < vb;
                #endif
                #if defined(LessOrEqual) | defined(LessOrEqualInt)
                    v = va <= vb;
                #endif
                #ifdef Or
                    v = va | vb;
                #endif
                #ifdef Xor
                    v = va ^ vb;
                #endif
                #ifdef PRelu
                    v = (0.5f * (1.0f + vb)) * va + (0.5f * (1.0f - vb)) * abs(va);
                #endif

                return v;
            }
            ENDCG
        }
    }
}
