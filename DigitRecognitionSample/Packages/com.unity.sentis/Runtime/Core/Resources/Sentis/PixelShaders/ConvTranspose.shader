Shader "Hidden/Sentis/ConvTranspose"
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
            #pragma multi_compile CONVTRANSPOSE1D CONVTRANSPOSE2D CONVTRANSPOSE3D
            #pragma multi_compile NONE RELU
            #pragma multi_compile _ USEBIAS

            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            #ifdef USEBIAS
            DECLARE_TENSOR(B, float);
            #endif
            DECLARE_TENSOR(K, float);
            DECLARE_TENSOR(X, float);

            uint O_width, O_height, O_depth, O_channelsDiv4;
            uint K_width, K_height, K_depth, K_mDivGroup;
            uint X_width, X_height, X_depth, X_channels, X_channelsDiv4;

            int StrideZ, StrideY, StrideX;
            int PadZ, PadY, PadX;

            float4 ApplyFusedActivation(float4 v)
            {
                #ifdef RELU
                return max(v, 0);
                #endif
                return v;
            }

            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                uint strideCK = 1;
                uint strideCX = 1;
                uint n = blockIndexO;
                int w = n % O_width;
                n /= O_width;
                strideCK *= K_width;
                strideCX *= X_width;
                const uint oxMin = w - PadX + StrideX - 1 < 0 ? 0 : (w - PadX + StrideX - 1) / (uint)StrideX;
                const uint oxMax = min(X_width, ceil((K_width + w - PadX + StrideX - 1) / StrideX));
                #if defined(CONVTRANSPOSE3D) | defined(CONVTRANSPOSE2D)
                int h = n % O_height;
                n /= O_height;
                strideCK *= K_height;
                strideCX *= X_height;
                const uint oyMin = h - PadY + StrideY - 1 < 0 ? 0 : (h - PadY + StrideY - 1) / (uint)StrideY;
                const uint oyMax = min(X_height, ceil((K_height + h - PadY + StrideY - 1) / StrideY));
                #endif
                #if defined(CONVTRANSPOSE3D)
                int d = n % O_depth;
                n /= O_depth;
                strideCK *= K_depth;
                strideCX *= X_depth;
                const uint ozMin = d - PadZ + StrideZ - 1 < 0 ? 0 : (d - PadZ + StrideZ - 1) / (uint)StrideZ;
                const uint ozMax = min(X_depth, ceil((K_depth + d - PadZ + StrideZ - 1) / StrideZ));
                #endif
                uint kDiv4 = n % O_channelsDiv4;
                n /= O_channelsDiv4;
                const uint4 k4Offset = strideCK * UnblockAxis(kDiv4);

                const uint xDelta = strideCX;
                const uint kDelta = strideCK * K_mDivGroup;

                #ifdef USEBIAS
                float4 acc4 = SampleBlockB(kDiv4);
                #else
                float4 acc4 = 0;
                #endif

                uint4 indexX = strideCX * X_channelsDiv4 * n;
                uint4 indexK = 0;

                #if defined(CONVTRANSPOSE3D)
                for (uint oz = ozMin, dz = K_depth - 1 - (ozMin * StrideZ - d + PadZ); oz < ozMax; oz++, dz -= StrideZ)
                {
                    indexX[2] = indexX[3] + oz * (X_width * X_height);
                    indexK[2] = indexK[3] + dz * (K_width * K_height);
                #endif
                #if defined(CONVTRANSPOSE3D) | defined(CONVTRANSPOSE2D)
                for (uint oy = oyMin, dy = K_height - 1 - (oyMin * StrideY - h + PadY); oy < oyMax; oy++, dy -= StrideY)
                {
                    indexX[1] = indexX[2] + oy * X_width;
                    indexK[1] = indexK[2] + dy * K_width;
                #endif
                for (uint ox = oxMin, dx = K_width - 1 - (oxMin * StrideX - w + PadX); ox < oxMax; ox++, dx -= StrideX)
                {
                    indexX[0] = indexX[1] + ox;
                    indexK[0] = indexK[1] + dx;
                    for (uint cDiv4 = 0; cDiv4 < X_channelsDiv4; ++cDiv4)
                    {
                        uint blockIndexX = indexX[0] + cDiv4 * xDelta;
                        uint4 blockIndexK = k4Offset + indexK[0] + cDiv4 * kDelta;
                        float4 v = SampleBlockX(blockIndexX);
                        v *= (UnblockAxis(cDiv4) < X_channels ? 1.0f : 0.0f);

                        float4 k0 = SampleBlockK(blockIndexK.x);
                        float4 k1 = SampleBlockK(blockIndexK.y);
                        float4 k2 = SampleBlockK(blockIndexK.z);
                        float4 k3 = SampleBlockK(blockIndexK.w);

                        acc4 += mul(float4x4(k0, k1, k2, k3), v);
                    }
                }
                #if defined(CONVTRANSPOSE3D) | defined(CONVTRANSPOSE2D)
                }
                #endif
                #if defined(CONVTRANSPOSE3D)
                }
                #endif
                return ApplyFusedActivation(acc4);
            }
            ENDCG
        }
    }
}
