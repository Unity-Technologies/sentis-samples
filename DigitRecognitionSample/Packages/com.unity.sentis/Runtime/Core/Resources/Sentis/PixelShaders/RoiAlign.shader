Shader "Hidden/Sentis/RoiAlign"
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
            #pragma multi_compile RoiAlignAvg RoiAlignMax

            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            DECLARE_TENSOR(B, int);
            DECLARE_TENSOR_BLOCK_STRIDE(B);
            DECLARE_TENSOR(X, float);
            DECLARE_TENSOR(S, float);

            float spatialScale;
            uint numRois;
            float normalizeOHeight;
            float normalizeOWidth;
            int samplingRatio;

            uint O_width, O_height, O_channelsDiv4;
            uint X_width, X_height, X_channelsDiv4;

            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                uint n = blockIndexO;
                uint w = n % O_width;
                n /= O_width;
                uint h = n % O_height;
                n /= O_height;
                uint cDiv4 = n % O_channelsDiv4;
                n /= O_channelsDiv4;

                uint batchIdx = SampleElementB(n);
                uint xOffset = X_width * X_height * (cDiv4 + X_channelsDiv4 * batchIdx);

                // https://github.com/pytorch/vision/blob/7dc5e5bd60b55eb4e6ea5c1265d6dc7b17d2e917/torchvision/csrc/ops/cpu/roi_align_kernel.cpp
                // https://github.com/pytorch/vision/blob/7947fc8fb38b1d3a2aca03f22a2e6a3caa63f2a0/torchvision/csrc/ops/cpu/roi_align_common.h
                float4 roi = SampleBlockS(n) * spatialScale;
                float roiStartW = roi.x;
                float roiStartH = roi.y;
                float roiEndW = roi.z;
                float roiEndH = roi.w;

                float roiWidth = roiEndW - roiStartW;
                float roiHeight = roiEndH - roiStartH;

                roiWidth = max(roiWidth, 1.0f);
                roiHeight = max(roiHeight, 1.0f);

                float binSizeH = roiHeight / ((float)O_height);
                float binSizeW = roiWidth / ((float)O_width);

                int roiBinGridH = (samplingRatio > 0) ? samplingRatio : ceil(roiHeight * normalizeOHeight);
                int roiBinGridW = (samplingRatio > 0) ? samplingRatio : ceil(roiWidth * normalizeOWidth);

                float startH = roiStartH + h * binSizeH;
                float startW = roiStartW + w * binSizeW;

                float4 v = 0.0f;
                for (uint iy = 0; iy < (uint)roiBinGridH; iy++)
                {
                    float y = startH + (iy + 0.5f) * binSizeH / ((float)roiBinGridH);

                    for (uint ix = 0; ix < (uint)roiBinGridW; ix++)
                    {
                        float x = startW + (ix + 0.5f) * binSizeW / ((float)roiBinGridW);

                        if (y >= (float)X_height || y < -1.0 || x >= (float)X_width || x < -1.0)
                            continue;

                        y = max(y, 0.0f);
                        x = max(x, 0.0f);

                        uint yLow = (uint)y;
                        uint xLow = (uint)x;
                        uint yHigh;
                        uint xHigh;

                        if (yLow >= X_height - 1)
                        {
                            yHigh = yLow = X_height - 1;
                            y = (float)yLow;
                        }
                        else
                        {
                            yHigh = yLow + 1;
                        }

                        if (xLow >= X_width - 1)
                        {
                            xHigh = xLow = X_width - 1;
                            x = (float)xLow;
                        }
                        else
                        {
                            xHigh = xLow + 1;
                        }

                        float ly = y - yLow;
                        float lx = x - xLow;
                        float hy = 1.0f - ly;
                        float hx = 1.0f - lx;
                        float w0 = hy * hx;
                        float w1 = hy * lx;
                        float w2 = ly * hx;
                        float w3 = ly * lx;

                        float4 x0 = w0 * SampleBlockX(xOffset + xLow + X_width * yLow);
                        float4 x1 = w1 * SampleBlockX(xOffset + xHigh + X_width * yLow);
                        float4 x2 = w2 * SampleBlockX(xOffset + xLow + X_width * yHigh);
                        float4 x3 = w3 * SampleBlockX(xOffset + xHigh + X_width * yHigh);

                        #ifdef RoiAlignAvg
                        v = v + x0 + x1 + x2 + x3;
                        #endif
                        #ifdef RoiAlignMax
                        v = max(v, max(x0, max(x1, max(x2, x3))));
                        #endif
                    }
                }

                #ifdef RoiAlignAvg
                v /= max(roiBinGridH * roiBinGridW, 1);
                #endif
                return v;
            }
            ENDCG
        }
    }
}
