Shader "Hidden/Sentis/Tile"
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
            #pragma multi_compile _ INT
            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"
            #include "../ComputeShaders/Tensor.cginc"

            #ifdef INT
            #define DTYPE4 int4
            DECLARE_TENSOR(X, int);
            #else
            #define DTYPE4 float4
            DECLARE_TENSOR(X, float);
            #endif

            uint DimO[8];
            uint DimX[8];

            DTYPE4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                uint blockIndexX = 0;
                uint n = blockIndexO;
                uint strideX = 1;
                [unroll]
                for (uint j = 0; j < SHAPE_MAXRANK; j++)
                {
                    blockIndexX += ((n % DimO[j]) % DimX[j]) * strideX;
                    n /= DimO[j];
                    strideX *= DimX[j];
                }

                DTYPE4 v = SampleBlockX(blockIndexX);

                return v;
            }
            ENDCG
        }
    }
}
