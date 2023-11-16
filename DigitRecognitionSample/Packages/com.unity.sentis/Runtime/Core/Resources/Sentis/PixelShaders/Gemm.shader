Shader "Hidden/Sentis/Gemm"
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
            #pragma multi_compile _ TRANSPOSE_X
            #pragma multi_compile _ TRANSPOSE_W

            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            DECLARE_TENSOR(X, float);
            DECLARE_TENSOR(W, float);

            uint M, K, Kdiv4, Ndiv4;

            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                uint nDiv4 = blockIndexO % Ndiv4;
                uint m = blockIndexO / Ndiv4;
                #ifdef TRANSPOSE_X
                uint xOffset = m;
                #else
                uint xOffset = Kdiv4 * m;
                #endif
                #ifdef TRANSPOSE_W
                uint wOffset = nDiv4 * K;
                #else
                uint wOffset = nDiv4;
                #endif

                float4 acc4 = 0.0f;

                for (uint kDiv4 = 0; kDiv4 < Kdiv4; kDiv4++)
                {
                    uint4 k = UnblockAxis(kDiv4);
                    float4 mask = k < K ? 1 : 0;
                    #ifdef TRANSPOSE_X
                    float4 v = mask * SampleBlockX(M * kDiv4 + xOffset);
                    #else
                    float4 v = mask * SampleBlockX(kDiv4 + xOffset);
                    #endif

                    #ifdef TRANSPOSE_W
                    uint4 wIndex = k + wOffset;
                    #else
                    uint4 wIndex = Ndiv4 * k + wOffset;
                    #endif
                    float4 w0 = SampleBlockW(wIndex.x);
                    float4 w1 = SampleBlockW(wIndex.y);
                    float4 w2 = SampleBlockW(wIndex.z);
                    float4 w3 = SampleBlockW(wIndex.w);

                    acc4 += mul(v, float4x4(w0, w1, w2, w3));
                }

                return acc4;
            }
            ENDCG
        }
    }
}
