Shader "Hidden/Sentis/Where"
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
            #pragma multi_compile WhereFloat WhereInt

            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            #if defined(WhereInt)
            #define DTYPE4 int4
            DECLARE_TENSOR(A, int);
            DECLARE_TENSOR(B, int);
            #else
            #define DTYPE4 float4
            DECLARE_TENSOR(A, float);
            DECLARE_TENSOR(B, float);
            #endif

            DECLARE_TENSOR(X, int);

            uint DimO[8];
            uint StridesA[8];
            uint StridesB[8];
            uint StridesX[8];
            uint DimAxisA, DimAxisB, DimAxisX;

            DTYPE4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                uint blockIndexA = 0;
                uint blockIndexB = 0;
                uint blockIndexX = 0;
                uint n = blockIndexO;
                [unroll]
                for (uint j = 0; j < 8; j++)
                {
                    uint k = (n % DimO[j]);
                    n /= DimO[j];
                    blockIndexA += k * StridesA[j];
                    blockIndexB += k * StridesB[j];
                    blockIndexX += k * StridesX[j];
                }

                DTYPE4 va = SampleBlockA(blockIndexA);
                DTYPE4 vb = SampleBlockB(blockIndexB);
                uint4 vx = SampleBlockX(blockIndexX);

                va = DimAxisA == 1 ? va.x : va;
                vb = DimAxisB == 1 ? vb.x : vb;
                vx = DimAxisX == 1 ? vx.x : vx;

                return vx ? va : vb;
            }
            ENDCG
        }
    }
}
