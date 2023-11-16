Shader "Hidden/Sentis/Range"
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
            #define DTYPE int
            #define DTYPE4 int4
            int rangeStartInt;
            int rangeDeltaInt;
            #else
            #define DTYPE float
            #define DTYPE4 float4
            float rangeStartFloat;
            float rangeDeltaFloat;
            #endif
            DECLARE_TENSOR_BLOCK_STRIDE_O;

            DTYPE4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                uint4 index4 = UnblockAxis(blockIndexO);
                #ifdef INT
                return rangeStartInt + rangeDeltaInt * index4;
                #else
                return rangeStartFloat + rangeDeltaFloat * index4;
                #endif
            }
            ENDCG
        }
    }
}
