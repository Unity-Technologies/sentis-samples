Shader "Hidden/Sentis/TextureTensorDataUpload"
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
            #pragma multi_compile Float Int
            #pragma vertex vert
            #pragma fragment frag

            #if defined(Int)
            #define O_INT
            #endif

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            DECLARE_TENSOR_BLOCK_STRIDE_O;
            DECLARE_TENSOR(X, float);

            #if defined(Int)
            inline int4 IntFromFloat(float4 a)
            {
                uint4 n = asuint(a);
                return asint((n & 0xbfffffffu) + ((n >> 1) & 0x40000000u));
            }
            int4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint4 index4 = GetIndexO(screenPos);
                float4 v = SampleElementsX(index4 >> 2, index4 & 3);
                return IntFromFloat(v);
            }
            #else
            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint4 index4 = GetIndexO(screenPos);
                return SampleElementsX(index4 >> 2, index4 & 3);
            }
            #endif
            ENDCG
        }
    }
}
