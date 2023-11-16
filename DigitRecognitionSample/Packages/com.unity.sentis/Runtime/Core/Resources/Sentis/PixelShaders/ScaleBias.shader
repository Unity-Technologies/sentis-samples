Shader "Hidden/Sentis/ScaleBias"
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
            #pragma multi_compile _ BLOCK_C
            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            DECLARE_TENSOR(X, float);
            DECLARE_TENSOR(S, float);
            DECLARE_TENSOR(B, float);

            uint StrideAxis, DimBlockedO;
            uint StrideC, DimC;

            uint Dim0;

            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);

                float4 v = SampleBlockX(blockIndexO);
                #ifdef BLOCK_C
                uint cDiv4 = (blockIndexO / StrideAxis) % DimBlockedO;
                float4 scale = SampleBlockS(cDiv4);
                float4 bias = SampleBlockB(cDiv4);
                return scale * v + bias;
                #else
                uint c = (blockIndexO / StrideC) % DimC;
                uint cDiv4 = c >> 2;
                uint cMod4 = c & 3;
                float scale = SampleBlockS(cDiv4)[cMod4];
                float bias = SampleBlockB(cDiv4)[cMod4];
                return scale * v + bias;
                #endif
            }
            ENDCG
        }
    }
}
