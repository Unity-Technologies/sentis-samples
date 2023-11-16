Shader "Hidden/Sentis/Dense"
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
            #pragma multi_compile Gemm Dense
            #pragma multi_compile None Relu

            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            DECLARE_TENSOR(X, float);
            DECLARE_TENSOR(W, float);
            DECLARE_TENSOR(B, float);

            uint DimAxisX;
            uint DimBlockedO, DimBlockedX, DimBlockedW;

            float4 ApplyFusedActivation(float4 v)
            {
                #ifdef Relu
                return max(v, 0);
                #endif
                return v;
            }

            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                uint2 index = Unravel(uint1(DimBlockedO), blockIndexO);
                uint kDiv4 = index[0];
                uint upper = index[1];
                uint xOffset = DimBlockedX * upper;

                float4 acc4 = 0.0f;
                #ifdef Dense
                acc4 = SampleBlockB(kDiv4);
                #endif

                for (uint cDiv4 = 0; cDiv4 < DimBlockedX; cDiv4++)
                {
                    uint4 c4 = UnblockAxis(cDiv4);
                    float4 mask = c4 < DimAxisX ? 1 : 0;
                    float4 v = mask * SampleBlockX(cDiv4 + xOffset);

                    uint4 kIndex = kDiv4 + DimBlockedW * c4;
                    float4 w0 = SampleBlockW(kIndex.x);
                    float4 w1 = SampleBlockW(kIndex.y);
                    float4 w2 = SampleBlockW(kIndex.z);
                    float4 w3 = SampleBlockW(kIndex.w);

                    acc4 += mul(v, float4x4(w0,w1,w2,w3));
                }

                return ApplyFusedActivation(acc4);
            }
            ENDCG
        }
    }
}
