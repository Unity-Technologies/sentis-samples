Shader "Hidden/Sentis/Softmax"
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
            #pragma multi_compile SOFTMAXEND LOGSOFTMAXEND

            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            DECLARE_TENSOR(X, float);
            DECLARE_TENSOR(S, float);
            DECLARE_TENSOR(B, float);

            uint StrideAxisX, DimAxisX;

            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                uint2 lowerUpper = Unravel(uint1(StrideAxisX), blockIndexO);
                lowerUpper[1] /= DimAxisX;
                uint blockIndexSB = Ravel(uint1(StrideAxisX), lowerUpper);
                float4 x = SampleBlockX(blockIndexO);
                float4 s = SampleBlockS(blockIndexSB);
                float4 b = SampleBlockB(blockIndexSB);
                #ifdef LOGSOFTMAXEND
                float4 v = (x - b) - log(s);
                #else // SOFTMAXEND
                float4 u = exp(x - b) / s;
                float4 v;
                v.x = s.x == 0 ? 0.0f : u.x;
                v.y = s.y == 0 ? 0.0f : u.y;
                v.z = s.z == 0 ? 0.0f : u.z;
                v.w = s.w == 0 ? 0.0f : u.w;
                #endif
                return v;
            }
            ENDCG
        }
    }
}
