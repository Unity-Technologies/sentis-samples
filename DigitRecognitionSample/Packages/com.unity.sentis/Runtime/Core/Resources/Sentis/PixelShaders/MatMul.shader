Shader "Hidden/Sentis/MatMul"
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
            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            DECLARE_TENSOR(A, float);
            DECLARE_TENSOR(B, float);

            uint DimO[8];
            uint StridesA[8];
            uint StridesB[8];
            uint DimAxisA;
            uint Kdiv4;

            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                uint blockIndexA = 0;
                uint blockIndexB = 0;
                uint n = blockIndexO;
                uint wDiv4 = (n % DimO[0]);
                n /= DimO[0];
                uint h = (n % DimO[1]);
                n /= DimO[1];
                [unroll]
                for (uint j = 2; j < 8; j++)
                {
                    uint k = (n % DimO[j]);
                    n /= DimO[j];
                    blockIndexA += k * StridesA[j];
                    blockIndexB += k * StridesB[j];
                }

                blockIndexA += Kdiv4 * h;

                float4 acc4 = 0.0f;

                for (uint cDiv4 = 0; cDiv4 < Kdiv4; cDiv4++)
                {
                    uint4 c4 = UnblockAxis(cDiv4);
                    float4 mask = c4 < DimAxisA ? 1 : 0;
                    float4 v = mask * SampleBlockA(cDiv4 + blockIndexA);

                    uint4 kIndex = blockIndexB + wDiv4 + StridesB[1] * c4;
                    float4 b0 = SampleBlockB(kIndex.x);
                    float4 b1 = SampleBlockB(kIndex.y);
                    float4 b2 = SampleBlockB(kIndex.z);
                    float4 b3 = SampleBlockB(kIndex.w);

                    acc4 += mul(v, float4x4(b0, b1, b2, b3));
                }

                return acc4;
            }
            ENDCG
        }
    }
}
