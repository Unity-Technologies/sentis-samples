Shader "Hidden/Sentis/Conv"
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
            #pragma multi_compile CONV1D CONV2D CONV3D
            #pragma multi_compile _ USEBIAS
            #pragma multi_compile NONE RELU

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
            uint K_width, K_height, K_depth;
            uint X_width, X_height, X_depth, X_channels, X_channelsDiv4;

            uint StrideZ, StrideY, StrideX;
            uint PadZ, PadY, PadX;
            uint DilationZ, DilationY, DilationX;
            uint Groups;

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
                uint w = n % O_width;
                n /= O_width;
                strideCK *= K_width;
                strideCX *= X_width;
                #if defined(CONV3D) | defined(CONV2D)
                uint h = n % O_height;
                n /= O_height;
                strideCK *= K_height;
                strideCX *= X_height;
                #endif
                #if defined(CONV3D)
                uint d = n % O_depth;
                n /= O_depth;
                strideCK *= K_depth;
                strideCX *= X_depth;
                #endif
                uint kDiv4 = n % O_channelsDiv4;
                n /= O_channelsDiv4;
                uint4 k4Offset = (strideCK * X_channelsDiv4) * UnblockAxis(kDiv4);

                #ifdef USEBIAS
                float4 acc4 = SampleBlockB(kDiv4);
                #else
                float4 acc4 = 0;
                #endif

                uint4 indexX = strideCX * X_channelsDiv4 * n;
                uint4 indexK = 0;

                #if defined(CONV3D)
                for (uint dz = 0; dz < K_depth; ++dz)
                {
                    uint oz = (d * StrideZ + DilationZ * dz) - PadZ;
                    if (oz >= X_depth)
                        continue;
                    indexX[2] = indexX[3] + oz * (X_width * X_height);
                    indexK[2] = indexK[3] + dz * (K_width * K_height);
                #endif
                #if defined(CONV3D) | defined(CONV2D)
                for (uint dy = 0; dy < K_height; ++dy)
                {
                    uint oy = (h * StrideY + DilationY * dy) - PadY;
                    if (oy >= X_height)
                        continue;
                    indexX[1] = indexX[2] + oy * X_width;
                    indexK[1] = indexK[2] + dy * K_width;
                #endif
                for (uint dx = 0; dx < K_width; ++dx)
                {
                    uint ox = (w * StrideX + DilationX * dx) - PadX;
                    if (ox >= X_width)
                        continue;
                    indexX[0] = indexX[1] + ox;
                    indexK[0] = indexK[1] + dx;

                    for (uint cDiv4 = 0; cDiv4 < X_channelsDiv4; ++cDiv4)
                    {
                        uint blockIndexX = indexX[0] + cDiv4 * strideCX;
                        uint4 blockIndexK = k4Offset + (indexK[0] + cDiv4 * strideCK);
                        float4 v = SampleBlockX(blockIndexX);
                        v *= UnblockAxis(cDiv4) < X_channels ? 1.0f : 0.0f;

                        uint4 kIndex4 = blockIndexK;
                        float4 k0 = SampleBlockK(kIndex4.x);
                        float4 k1 = SampleBlockK(kIndex4.y);
                        float4 k2 = SampleBlockK(kIndex4.z);
                        float4 k3 = SampleBlockK(kIndex4.w);

                        acc4 += mul(float4x4(k0, k1, k2, k3), v);
                    }
                }
                #if defined(CONV3D) | defined(CONV2D)
                }
                #endif
                #if defined(CONV3D)
                }
                #endif

                return ApplyFusedActivation(acc4);
            }
            ENDCG
        }
    }
}
