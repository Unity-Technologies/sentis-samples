Shader "Hidden/Sentis/BatchNormalization"
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
            DECLARE_TENSOR(B, float);
            DECLARE_TENSOR(M, float);
            DECLARE_TENSOR(V, float);

            float epsilon;
            uint O_channels, O_width, O_channelsDiv4;

            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);

                float4 v = SampleBlockX(blockIndexO);

                uint cDiv4 = (blockIndexO / O_width) % O_channelsDiv4;
                float4 scale = SampleBlockS(cDiv4);
                float4 bias = SampleBlockB(cDiv4);
                float4 mean = SampleBlockM(cDiv4);
                float4 variance = SampleBlockV(cDiv4);

                v = (v - mean);

                if (cDiv4 * 4 + 0 < O_channels)
                    v.x /= sqrt(variance.x + epsilon);
                if (cDiv4 * 4 + 1 < O_channels)
                    v.y /= sqrt(variance.y + epsilon);
                if (cDiv4 * 4 + 2 < O_channels)
                    v.z /= sqrt(variance.z + epsilon);
                if (cDiv4 * 4 + 3 < O_channels)
                    v.w /= sqrt(variance.w + epsilon);


                v = scale * v + bias;

                return v;
            }
            ENDCG
        }
    }
}
