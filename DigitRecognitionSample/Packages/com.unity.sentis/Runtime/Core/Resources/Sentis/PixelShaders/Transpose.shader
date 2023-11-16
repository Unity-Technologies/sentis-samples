Shader "Hidden/Sentis/Transpose"
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

            #ifdef INT
            #define DTYPE4 int4
            DECLARE_TENSOR(X, int);
            #else
            #define DTYPE4 float4
            DECLARE_TENSOR(X, float);
            #endif

            uint DimO[8];
            uint StridesX[8];

            DTYPE4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                uint blockIndexX = 0;
                uint n = blockIndexO;
                [unroll]
                for (uint j = 0; j < 8; j++)
                {
                    blockIndexX += (n % DimO[j]) * StridesX[j];
                    n /= DimO[j];
                }

                DTYPE4 v = SampleBlockX(blockIndexX);

                return v;
            }
            ENDCG
        }
    }
}
