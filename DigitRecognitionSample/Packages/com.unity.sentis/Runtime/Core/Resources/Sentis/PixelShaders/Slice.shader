Shader "Hidden/Sentis/Slice"
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
            #pragma multi_compile _ BLOCKWISE
            #pragma multi_compile _ INT
            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            DECLARE_TENSOR_BLOCK_STRIDE_O;

            #ifdef INT
            #define DTYPE4 int4
            DECLARE_TENSOR(X, int);
            #else
            #define DTYPE4 float4
            DECLARE_TENSOR(X, float);
            #endif
            DECLARE_TENSOR_BLOCK_STRIDE(X);

            uint DimO[8];
            uint StridesX[8];
            uint OffsetX;

            DTYPE4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                DTYPE4 v = 0;
                #ifdef BLOCKWISE
                uint blockIndexO = GetBlockIndexO(screenPos);
                uint blockIndexX = OffsetX;
                uint n = blockIndexO;
                [unroll]
                for (uint j = 0; j < 8; j++)
                {
                    blockIndexX += (n % DimO[j]) * StridesX[j];
                    n /= DimO[j];
                }
                v = SampleBlockX(blockIndexX);
                #else
                uint4 indexO4 = GetIndexO(screenPos);
                uint4 n4 = indexO4;
                uint4 indexX4 = OffsetX;
                for (uint j = 0; j < 8; j++)
                {
                    indexX4 += (n4 % DimO[j]) * StridesX[j];
                    n4 /= DimO[j];
                }
                v = SampleElementsX(indexX4);
                #endif

                return v;
            }
            ENDCG
        }
    }
}
