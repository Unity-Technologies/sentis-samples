Shader "Hidden/Sentis/Cast"
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
            #pragma multi_compile FloatToInt IntToFloat

            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            DECLARE_TENSOR_BLOCK_STRIDE_O;

            #ifdef FloatToInt
            DECLARE_TENSOR(X, float);
            int4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                float4 v = SampleBlockX(blockIndexO);
                return int4(v.x, v.y, v.z, v.w);
            }
            #else
            DECLARE_TENSOR(X, int);
            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                int4 v = SampleBlockX(blockIndexO);
                return float4(v.x, v.y, v.z, v.w);
            }
            #endif

            ENDCG
        }
    }
}
