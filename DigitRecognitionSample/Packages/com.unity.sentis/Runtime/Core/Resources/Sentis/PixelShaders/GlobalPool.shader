Shader "Hidden/Sentis/GlobalPool"
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
            #pragma multi_compile AVGPOOL MAXPOOL AVGSQUAREPOOL

            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            #define FLT_MIN -3.402823466e+38F

            DECLARE_TENSOR(X, float);

            uint SpatialSizeX, DimAxis;
            float Normalization;

            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                uint cDiv4 = blockIndexO % DimAxis;
                uint n = blockIndexO / DimAxis;

                #if defined(AVGPOOL) | defined(AVGSQUAREPOOL)
                float4 acc4 = 0.0f;
                #elif defined(MAXPOOL)
                float4 acc4 = FLT_MIN;
                #endif

                uint offsetX = SpatialSizeX * (cDiv4 + DimAxis * n);

                for (uint j = 0; j < SpatialSizeX; j++)
                {
                    uint blockIndexX = j + offsetX;
                    float4 v = SampleBlockX(blockIndexX);
                    #if defined(AVGPOOL)
                    acc4 += v;
                    #elif defined(AVGSQUAREPOOL)
                    acc4 += v * v;
                    #elif defined(MAXPOOL)
                    acc4 = max(v, acc4);
                    #endif
                }
                #if defined(AVGPOOL) | defined(AVGSQUAREPOOL)
                acc4 *= Normalization;
                #endif

                return acc4;
            }
            ENDCG
        }
    }
}
