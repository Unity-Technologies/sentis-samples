Shader "Hidden/Sentis/IsInfNaN"
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
            #pragma multi_compile IsInf IsNaN

            #pragma vertex vert
            #pragma fragment frag

            #define O_INT

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            DECLARE_TENSOR(X, float);

            bool detectNegative;
            bool detectPositive;

            int4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                float4 v = SampleBlockX(blockIndexO);
                int4 vOut;
                #ifdef IsInf
                    vOut = isinf(v) && ((v > 0 && detectPositive) || (v < 0 && detectNegative)) ? 1 : 0;
                #endif
                #ifdef IsNaN
                    vOut = isnan(v) ? 1 : 0;
                #endif
                return vOut;
            }
            ENDCG
        }
    }
}
