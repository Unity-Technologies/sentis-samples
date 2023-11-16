Shader "Hidden/Sentis/LayerNormalizationTail"
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

            uint reduceLength;
            float epsilon;

            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                float4 v = SampleBlockX(blockIndexO);

                uint blockIndexA = blockIndexO / reduceLength;
                uint blockIndexS = blockIndexO % reduceLength;
                float4 mean = SampleBlockA(blockIndexA);
                float4 meanSqr = SampleBlockK(blockIndexA);
                float4 variance = max(0, meanSqr - mean * mean); // avoid NaNs
                float scale = SampleBlockS(blockIndexS).x;
                float bias = SampleBlockB(blockIndexS).x;
                return scale * (v - mean) / sqrt(variance + epsilon) + bias;
            }
            ENDCG
        }
    }
}
