Shader "Hidden/Sentis/TextureTensorDataDownload"
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

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            #if defined(Int)
            DECLARE_TENSOR(X, int);
            #else
            DECLARE_TENSOR(X, float);
            #endif

            DECLARE_TENSOR_BLOCK_STRIDE(X);

            inline float4 IntToFloat(int4 a)
            {
                a = clamp(a, -1073741824, 1073741823);
                uint4 n = asuint(a);

                return asfloat((n & 0xbfffffffu) + (((0xffffffffu ^ n) << 1) & 0x40000000u));
            }

            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                #ifdef Float
                uint4 index4 = UnblockAxis(blockIndexO);
                return SampleElementsX(index4);
                #elif Int
                uint4 index4 = UnblockAxis(blockIndexO);
                int4 v = SampleElementsX(index4);
                return IntToFloat(v);
                #else
                #endif
            }
            ENDCG
        }
    }
}
