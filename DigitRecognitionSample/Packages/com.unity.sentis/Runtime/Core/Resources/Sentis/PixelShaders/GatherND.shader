Shader "Hidden/Sentis/GatherND"
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
            DECLARE_TENSOR(B, int);
            DECLARE_TENSOR_BLOCK_STRIDE(X)
            DECLARE_TENSOR_BLOCK_STRIDE(B)
            DECLARE_TENSOR_BLOCK_STRIDE_O;

            uint ShapeO[8];
            uint ShapeX[8];
            uint ShapeB[8];
            uint StridesO[8];
            uint StridesX[8];
            uint StridesB[8];
            uint RankX, RankO, RankB;

            uint iStart, iEndIndices, iEndX, iEndMin, iStartB, iEndB;

            float4 frag(v2f j, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint4 indexO4 = GetIndexO(screenPos);
                uint4 itIndices = 0;
                uint4 itX = 0;
                uint i;

                // iterate up to point where i < iEndIndices and i < iEndX
                for (i = iStart; i < iEndMin; i++)
                {
                    uint4 itO = (indexO4 / StridesO[i]) % ShapeO[i];
                    itIndices += itO * StridesB[(RankO - RankB) + i];
                    itX += itO * StridesX[(RankO - RankX) + i];
                }

                // finish indices if iEndIndices > iEndX
                for (i = iEndMin; i < iEndIndices; i++)
                {
                    itIndices += ((indexO4 / StridesO[i]) % ShapeO[i]) * StridesB[(RankO - RankB) + i];
                }

                // finish X if iEndX > iEndIndices
                for (i = iEndMin; i < iEndX; i++)
                {
                    itX += ((indexO4 / StridesO[i]) % ShapeO[i]) * StridesX[(RankO - RankX) + i];
                }

                itIndices -= iStartB;

                for (i = iStartB; i < iEndB; i++)
                {
                    int4 index4 = SampleElementsB(itIndices + i);
                    index4 = index4 < 0 ? ShapeX[i] + index4 : index4;
                    itX += index4 * StridesX[i];
                }

                for (; i < 8; i++)
                {
                    itX += ((indexO4 / StridesO[i]) % ShapeO[i]) * StridesX[i];
                }

                return SampleElementsX(itX);
            }
            ENDCG
        }
    }
}
