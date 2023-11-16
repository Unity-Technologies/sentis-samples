Shader "Hidden/Sentis/Random"
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
            #pragma multi_compile RandomUniform RandomNormal Bernoulli BernoulliInt

            #pragma vertex vert
            #pragma fragment frag


            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"
            #include "../ComputeShaders/Random.cginc"

            DECLARE_TENSOR_BLOCK_STRIDE_O;

            #ifdef BernoulliInt
            #define DTYPE4 int4
            #else
            #define DTYPE4 float4
            #endif

            #if defined(Bernoulli) | defined(BernoulliInt)
            DECLARE_TENSOR(X, float);
            #endif

            float low;
            float high;
            float mean;
            float scale;
            uint seed;

            DTYPE4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                int3 lowerAxisUpper = UnravelO(blockIndexO);
                uint4 unblocked4 = UnblockAxis(lowerAxisUpper[1]);
                uint4 index4 = lowerAxisUpper[0] + StrideAxisO * (unblocked4 + DimAxisO * lowerAxisUpper[2]);
                bool4 mask4 = (index4 < LengthO && unblocked4 < DimAxisO) ? 1 : 0;
                index4 += seed;
                // index may not be uint.MaxValue, in this case move to distant value
                // following Unity.Mathematics.Random
                uint4 randomState4 = WangHash((index4 != 4294967295u ? index4 : 2147483647u) + 62u);
                #if defined(Bernoulli) | defined(BernoulliInt)
                return mask4 * (SampleBlockX(blockIndexO) > ToFloat4(randomState4) ? 1 : 0);
                #endif
                #ifdef RandomUniform
                randomState4 = NextState(randomState4);
                return mask4 * (low + (high - low) * ToFloat4(randomState4));
                #endif
                #ifdef RandomNormal
                float4 v = 0;
                if (mask4.x)
                    v.x = mean + scale * GetRandomNormal(randomState4.x);
                if (mask4.y)
                    v.y = mean + scale * GetRandomNormal(randomState4.y);
                if (mask4.z)
                    v.z = mean + scale * GetRandomNormal(randomState4.z);
                if (mask4.w)
                    v.w = mean + scale * GetRandomNormal(randomState4.w);
                return v;
                #endif
            }
            ENDCG
        }
    }
}
