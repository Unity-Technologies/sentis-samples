Shader "Hidden/Sentis/Upsample"
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
            #pragma multi_compile UPSAMPLE1D UPSAMPLE2D UPSAMPLE3D
            #pragma multi_compile LINEAR NEAREST_FLOOR NEAREST_CEIL

            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            DECLARE_TENSOR(X, float);

            uint O_width, O_height, O_depth, O_channelsDiv4;
            uint X_width, X_height, X_depth, X_channelsDiv4;

            float4 Scale;
            float4 Bias;

            float4 BilinearInterpolation(float fracSrcPosX, float fracSrcPosY, float4 p00, float4 p01, float4 p10, float4 p11)
            {
                float4 v = p00 * (1 - fracSrcPosX) * (1 - fracSrcPosY) +
                           p01 * (1 - fracSrcPosX) * fracSrcPosY +
                           p10 * fracSrcPosX       * (1 - fracSrcPosY) +
                           p11 * fracSrcPosX       * fracSrcPosY;
                return v;
            }

            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                uint n = blockIndexO;
                uint w = n % O_width;
                n /= O_width;
                #if defined(UPSAMPLE2D) | defined(UPSAMPLE3D)
                uint h = n % O_height;
                n /= O_height;
                #ifdef UPSAMPLE3D
                uint d = n % O_depth;
                n /= O_depth;
                #endif
                #endif
                uint cDiv4 = n % O_channelsDiv4;
                n /= O_channelsDiv4;

                #ifdef UPSAMPLE1D
                float srcPosX = w * Scale[0] + Bias[0];
                int offset = X_width * (cDiv4 + X_channelsDiv4 * n);
                #endif
                #ifdef UPSAMPLE2D
                float srcPosY = h * Scale[0] + Bias[0];
                float srcPosX = w * Scale[1] + Bias[1];
                int offset = X_width * X_height * (cDiv4 + X_channelsDiv4 * n);
                #endif
                #ifdef UPSAMPLE3D
                float srcPosZ = d * Scale[0] + Bias[0];
                float srcPosY = h * Scale[1] + Bias[1];
                float srcPosX = w * Scale[2] + Bias[2];
                int offset = X_width * X_height * X_depth * (cDiv4 + X_channelsDiv4 * n);
                #endif

                float4 v = 0;
                #if defined(LINEAR)
                    float floorSrcPosX = floor(srcPosX);
                    float fracSrcPosX = srcPosX - floorSrcPosX;
                    int xLower = clamp((int)floorSrcPosX + 0, 0, (int)X_width - 1);
                    int xUpper = clamp((int)floorSrcPosX + 1, 0, (int)X_width - 1);

                    #ifdef UPSAMPLE1D
                    //from https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/interpolation/trilinear-interpolation
                    float4 p0 = SampleBlockX(xLower + offset);
                    float4 p1 = SampleBlockX(xUpper + offset);
                    v = p0 * (1 - fracSrcPosX) + p1 * fracSrcPosX;
                    #else
                    float floorSrcPosY = floor(srcPosY);
                    float fracSrcPosY = srcPosY - floorSrcPosY;
                    int yLower = clamp((int)floorSrcPosY + 0, 0, (int)X_height - 1);
                    int yUpper = clamp((int)floorSrcPosY + 1, 0, (int)X_height - 1);
                    #ifdef UPSAMPLE2D
                    //from https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/interpolation/trilinear-interpolation
                    float4 p00 = SampleBlockX(xLower + X_width * yLower + offset);
                    float4 p01 = SampleBlockX(xLower + X_width * yUpper + offset);
                    float4 p10 = SampleBlockX(xUpper + X_width * yLower + offset);
                    float4 p11 = SampleBlockX(xUpper + X_width * yUpper + offset);
                    v = BilinearInterpolation(fracSrcPosX, fracSrcPosY, p00, p01, p10, p11);
                    #endif
                    #ifdef UPSAMPLE3D
                    float floorSrcPosZ = floor(srcPosZ);
                    float fracSrcPosZ = srcPosZ - floorSrcPosZ;
                    int zLower = clamp((int)floorSrcPosZ + 0, 0, (int)X_depth - 1);
                    int zUpper = clamp((int)floorSrcPosZ + 1, 0, (int)X_depth - 1);

                    //from https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/interpolation/trilinear-interpolation
                    float4 p000 = SampleBlockX(xLower + X_width * (yLower + X_height * zLower) + offset);
                    float4 p001 = SampleBlockX(xLower + X_width * (yLower + X_height * zUpper) + offset);
                    float4 p010 = SampleBlockX(xLower + X_width * (yUpper + X_height * zLower) + offset);
                    float4 p011 = SampleBlockX(xLower + X_width * (yUpper + X_height * zUpper) + offset);
                    float4 p100 = SampleBlockX(xUpper + X_width * (yLower + X_height * zLower) + offset);
                    float4 p101 = SampleBlockX(xUpper + X_width * (yLower + X_height * zUpper) + offset);
                    float4 p110 = SampleBlockX(xUpper + X_width * (yUpper + X_height * zLower) + offset);
                    float4 p111 = SampleBlockX(xUpper + X_width * (yUpper + X_height * zUpper) + offset);
                    float4 e = BilinearInterpolation(fracSrcPosX, fracSrcPosY, p000, p010, p100, p110);
                    float4 f = BilinearInterpolation(fracSrcPosX, fracSrcPosY, p001, p011, p101, p111);
                    v = e * (1 - fracSrcPosZ) + f * fracSrcPosZ;
                    #endif
                    #endif
                #else
                    #if defined(NEAREST_FLOOR)
                        int ox = clamp((int)floor(srcPosX), 0, (int)X_width - 1);
                        #if defined(UPSAMPLE2D) | defined(UPSAMPLE3D)
                        int oy = clamp((int)floor(srcPosY), 0, (int)X_height - 1);
                        #if defined(UPSAMPLE3D)
                        int oz = clamp((int)floor(srcPosZ), 0, (int)X_depth - 1);
                        #endif
                        #endif
                    #else // defined(NEAREST_CEIL)
                        int ox = clamp((int)ceil(srcPosX), 0, (int)X_width - 1);
                        #if defined(UPSAMPLE2D) | defined(UPSAMPLE3D)
                        int oy = clamp((int)ceil(srcPosY), 0, (int)X_height - 1);
                        #if defined(UPSAMPLE3D)
                        int oz = clamp((int)ceil(srcPosZ), 0, (int)X_depth - 1);
                        #endif
                        #endif
                    #endif
                    #ifdef UPSAMPLE1D
                    v = SampleBlockX(ox + offset);
                    #endif
                    #ifdef UPSAMPLE2D
                    v = SampleBlockX(ox + X_width * oy + offset);
                    #endif
                    #ifdef UPSAMPLE3D
                    v = SampleBlockX(ox + X_width * (oy + X_height * oz) + offset);
                    #endif
                #endif

                return v;
            }
            ENDCG
        }
    }
}
