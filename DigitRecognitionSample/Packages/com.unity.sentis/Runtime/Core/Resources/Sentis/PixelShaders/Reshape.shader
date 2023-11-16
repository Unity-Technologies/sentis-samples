Shader "Hidden/Sentis/Reshape"
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
            #pragma multi_compile _ BLOCKWISE
            #pragma multi_compile _ INT
            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            DECLARE_TENSOR_BLOCK_STRIDE_O;

            #ifdef INT
            #define DTYPE4 int4
            DECLARE_TENSOR(X, int);
            #else
            #define DTYPE4 float4
            DECLARE_TENSOR(X, float);
            #endif
            DECLARE_TENSOR_BLOCK_STRIDE(X);

            DTYPE4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                DTYPE4 v = 0;
                #ifdef BLOCKWISE
                uint blockIndexO = GetBlockIndexO(screenPos);
                v = SampleBlockX(blockIndexO);
                #else
                uint4 index4 = GetIndexO(screenPos);
                v = SampleElementsX(index4);
                #endif

                return v;
            }
            ENDCG
        }
    }
}
