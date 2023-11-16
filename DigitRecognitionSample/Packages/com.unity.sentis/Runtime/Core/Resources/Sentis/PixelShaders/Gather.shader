Shader "Hidden/Sentis/Gather"
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
            #pragma multi_compile _ GatherInt
            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            #ifdef GatherInt
            #define DTYPE4 int4
            DECLARE_TENSOR(X, int);
            #else
            #define DTYPE4 float4
            DECLARE_TENSOR(X, float);
            #endif
            DECLARE_TENSOR(B, int);
            DECLARE_TENSOR_BLOCK_STRIDE(X)
            DECLARE_TENSOR_BLOCK_STRIDE(B)
            DECLARE_TENSOR_BLOCK_STRIDE_O;

            uint endLength, indicesLength, axisDim;

            DTYPE4 frag(v2f j, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint4 indexO4 = GetIndexO(screenPos);
                int4 end4 = indexO4 % endLength;
                int4 indices4 = (indexO4 / endLength) % indicesLength;
                int4 start4 = indexO4 / (endLength * indicesLength);

                int4 index4 = SampleElementsB(indices4);
                index4 = index4 < 0 ? axisDim + index4 : index4;

                return SampleElementsX(start4 * endLength * axisDim + index4 * endLength + end4);
            }
            ENDCG
        }
    }
}
