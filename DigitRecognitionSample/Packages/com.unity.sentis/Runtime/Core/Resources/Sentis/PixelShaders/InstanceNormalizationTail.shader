Shader "Hidden/Sentis/InstanceNormalizationTail"
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
            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            DECLARE_TENSOR(X, float);
            DECLARE_TENSOR(S, float);
            DECLARE_TENSOR(A, float);
            DECLARE_TENSOR(B, float);
            DECLARE_TENSOR(K, float);

            uint StrideAxis, O_channelsDiv4;
            float epsilon;

            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                float4 v = SampleBlockX(blockIndexO);

                uint blockIndexA = (blockIndexO / StrideAxis);
                uint blockIndexS = blockIndexA % O_channelsDiv4;
                float4 mean = SampleBlockA(blockIndexA);
                float4 meanSqr = SampleBlockK(blockIndexA);
                float4 variance = meanSqr - mean * mean;
                float4 scale = SampleBlockS(blockIndexS);
                float4 bias = SampleBlockB(blockIndexS);
                return scale * (v - mean) / sqrt(variance + epsilon) + bias;
            }
            ENDCG
        }
    }
}
