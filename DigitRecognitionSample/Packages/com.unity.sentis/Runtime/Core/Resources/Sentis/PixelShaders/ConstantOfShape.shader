Shader "Hidden/Sentis/ConstantOfShape"
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
            DECLARE_TENSOR(A, int);
            DECLARE_TENSOR(B, int);
            int memValueInt;

            int4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                return memValueInt;
            }
            #else
            DECLARE_TENSOR(A, float);
            DECLARE_TENSOR(B, float);

            float memValueFloat;

            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                return memValueFloat;
            }
            #endif
            ENDCG
        }
    }
}
