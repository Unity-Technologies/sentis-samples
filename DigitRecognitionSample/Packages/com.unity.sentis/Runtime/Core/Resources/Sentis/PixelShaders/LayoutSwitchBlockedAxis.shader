Shader "Hidden/Sentis/LayoutSwitchBlockedAxis"
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
            #pragma multi_compile FLOAT INT
            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            DECLARE_TENSOR_BLOCK_STRIDE_O;

            #ifdef FLOAT
            DECLARE_TENSOR(X, float);
            DECLARE_TENSOR_BLOCK_STRIDE(X);
            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint4 index4 = GetIndexO(screenPos);
                return SampleElementsX(index4);
            }
            #else
            DECLARE_TENSOR(X, int);
            DECLARE_TENSOR_BLOCK_STRIDE(X);
            int4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint4 index4 = GetIndexO(screenPos);
                return SampleElementsX(index4);
            }
            #endif
            ENDCG
        }
    }
}
