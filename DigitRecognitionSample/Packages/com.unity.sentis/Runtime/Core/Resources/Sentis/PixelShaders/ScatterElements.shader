Shader "Hidden/Sentis/ScatterElements"
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
            #pragma multi_compile _ ScatterInt
            #pragma multi_compile ReduceNone ReduceAdd ReduceMul
            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            #ifdef ScatterInt
            #define DTYPE4 int4
            DECLARE_TENSOR(X, int);
            DECLARE_TENSOR(W, int);
            #else
            #define DTYPE4 float4
            DECLARE_TENSOR(X, float);
            DECLARE_TENSOR(W, float);
            #endif
            DECLARE_TENSOR(B, int);
            DECLARE_TENSOR_BLOCK_STRIDE(X)
            DECLARE_TENSOR_BLOCK_STRIDE(B)
            DECLARE_TENSOR_BLOCK_STRIDE(W)
            DECLARE_TENSOR_BLOCK_STRIDE_O;

            uint DimAxis, StrideAxis, NumIndices;

            DTYPE4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                uint3 lowerAxisUpperO = Unravel(uint2(StrideAxis, DimAxis), blockIndexO);
                DTYPE4 v = SampleBlockX(blockIndexO);
                uint blockIndexI = lowerAxisUpperO.x + StrideAxis * NumIndices * lowerAxisUpperO.z;
                for (uint j = 0; j < NumIndices; j++)
                {
                    int4 indicesJ = SampleBlockB(blockIndexI);
                    int4 mask = (indicesJ == lowerAxisUpperO.y || (indicesJ + DimAxis) == lowerAxisUpperO.y) ? 1 : 0;
                    DTYPE4 updates = SampleBlockW(blockIndexI);

                    #ifdef ReduceNone
                    v = v * (1 - mask) + updates * mask;
                    #elif ReduceAdd
                    v = v + updates * mask;
                    #elif ReduceMul
                    v = v * ((1 - mask) + updates * mask);
                    #endif
                    blockIndexI += StrideAxis;
                }

                return v;
            }
            ENDCG
        }
    }
}
