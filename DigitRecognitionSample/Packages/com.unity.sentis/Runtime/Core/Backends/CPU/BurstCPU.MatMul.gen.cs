// This is auto-generated -- do not modify directly
using UnityEngine;
using System;
using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Burst.CompilerServices;
using Unity.Burst.Intrinsics;
using static Unity.Burst.Intrinsics.X86.Avx;
using static Unity.Burst.Intrinsics.X86.Avx2;
using static Unity.Burst.Intrinsics.X86.Fma;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;
using Unity.Mathematics;

namespace Unity.Sentis {
public partial class CPUBackend
{
#if UNITY_WEBGL || WEBGL_MATMUL_OVERRIDE
    // Note: This is a copy of an older version of the kernel that did not use v256. The inline expansion
    // of VectorUtils.MulAdd causes the compiler to emit code that is twice as a kernel that pastes the
    // body of VectorUtils.MulAdd directly into the kernel. While this problem is investigated, WASM builds
    // will use this older version of the kernel.
    static unsafe void MultiplyBlockUnroll2x16Wasm(
        [NoAlias] float* Ap, int strideA,
        [NoAlias] float* Bp, int strideB,
        [NoAlias] float* Cp, int strideC,
        int M, int N, int K, bool accumulateC)
    {
        // Avoid generating compiled code for the case where M/N/K are negative or zero.
        Hint.Assume(M > 0);
        Hint.Assume(N > 0);
        Hint.Assume(K > 0);

        // Help the compiler produce better code by using the strides as pointer-width values.
        // For x64, the inner loop is then able to use scaled indexing modes to access the rows
        // of matrix A, instead of wasting extra registers to store the offsets.
        var nstrideA = new System.IntPtr(strideA);
        var nstrideB = new System.IntPtr(strideB);
        var nstrideC = new System.IntPtr(strideC);

        while (M >= 2)
        {
            float *Bnp = Bp;
            float *Cnp = Cp;
            var n = (uint)N;

            while (n >= 16)
            {
                float accum0_0 = 0.0f;
                float accum1_0 = 0.0f;
                float accum2_0 = 0.0f;
                float accum3_0 = 0.0f;
                float accum4_0 = 0.0f;
                float accum5_0 = 0.0f;
                float accum6_0 = 0.0f;
                float accum7_0 = 0.0f;
                float accum8_0 = 0.0f;
                float accum9_0 = 0.0f;
                float accum10_0 = 0.0f;
                float accum11_0 = 0.0f;
                float accum12_0 = 0.0f;
                float accum13_0 = 0.0f;
                float accum14_0 = 0.0f;
                float accum15_0 = 0.0f;

                float accum0_1 = 0.0f;
                float accum1_1 = 0.0f;
                float accum2_1 = 0.0f;
                float accum3_1 = 0.0f;
                float accum4_1 = 0.0f;
                float accum5_1 = 0.0f;
                float accum6_1 = 0.0f;
                float accum7_1 = 0.0f;
                float accum8_1 = 0.0f;
                float accum9_1 = 0.0f;
                float accum10_1 = 0.0f;
                float accum11_1 = 0.0f;
                float accum12_1 = 0.0f;
                float accum13_1 = 0.0f;
                float accum14_1 = 0.0f;
                float accum15_1 = 0.0f;

                if (accumulateC)
                {
                    accum0_0 = StrideAddress(Cnp, nstrideC, 0)[0];
                    accum1_0 = StrideAddress(Cnp, nstrideC, 0)[1];
                    accum2_0 = StrideAddress(Cnp, nstrideC, 0)[2];
                    accum3_0 = StrideAddress(Cnp, nstrideC, 0)[3];
                    accum4_0 = StrideAddress(Cnp, nstrideC, 0)[4];
                    accum5_0 = StrideAddress(Cnp, nstrideC, 0)[5];
                    accum6_0 = StrideAddress(Cnp, nstrideC, 0)[6];
                    accum7_0 = StrideAddress(Cnp, nstrideC, 0)[7];
                    accum8_0 = StrideAddress(Cnp, nstrideC, 0)[8];
                    accum9_0 = StrideAddress(Cnp, nstrideC, 0)[9];
                    accum10_0 = StrideAddress(Cnp, nstrideC, 0)[10];
                    accum11_0 = StrideAddress(Cnp, nstrideC, 0)[11];
                    accum12_0 = StrideAddress(Cnp, nstrideC, 0)[12];
                    accum13_0 = StrideAddress(Cnp, nstrideC, 0)[13];
                    accum14_0 = StrideAddress(Cnp, nstrideC, 0)[14];
                    accum15_0 = StrideAddress(Cnp, nstrideC, 0)[15];

                    accum0_1 = StrideAddress(Cnp, nstrideC, 1)[0];
                    accum1_1 = StrideAddress(Cnp, nstrideC, 1)[1];
                    accum2_1 = StrideAddress(Cnp, nstrideC, 1)[2];
                    accum3_1 = StrideAddress(Cnp, nstrideC, 1)[3];
                    accum4_1 = StrideAddress(Cnp, nstrideC, 1)[4];
                    accum5_1 = StrideAddress(Cnp, nstrideC, 1)[5];
                    accum6_1 = StrideAddress(Cnp, nstrideC, 1)[6];
                    accum7_1 = StrideAddress(Cnp, nstrideC, 1)[7];
                    accum8_1 = StrideAddress(Cnp, nstrideC, 1)[8];
                    accum9_1 = StrideAddress(Cnp, nstrideC, 1)[9];
                    accum10_1 = StrideAddress(Cnp, nstrideC, 1)[10];
                    accum11_1 = StrideAddress(Cnp, nstrideC, 1)[11];
                    accum12_1 = StrideAddress(Cnp, nstrideC, 1)[12];
                    accum13_1 = StrideAddress(Cnp, nstrideC, 1)[13];
                    accum14_1 = StrideAddress(Cnp, nstrideC, 1)[14];
                    accum15_1 = StrideAddress(Cnp, nstrideC, 1)[15];
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                var k = (uint)K;

                for (; k > 0; k -= 1)
                {
                    float B_0 = Bkp[0];
                    float B_1 = Bkp[1];
                    float B_2 = Bkp[2];
                    float B_3 = Bkp[3];
                    float B_4 = Bkp[4];
                    float B_5 = Bkp[5];
                    float B_6 = Bkp[6];
                    float B_7 = Bkp[7];
                    float B_8 = Bkp[8];
                    float B_9 = Bkp[9];
                    float B_10 = Bkp[10];
                    float B_11 = Bkp[11];
                    float B_12 = Bkp[12];
                    float B_13 = Bkp[13];
                    float B_14 = Bkp[14];
                    float B_15 = Bkp[15];

                    float A_0 = StrideAddress(Akp, nstrideA, 0)[0];
                    accum0_0 += A_0 * B_0;
                    accum1_0 += A_0 * B_1;
                    accum2_0 += A_0 * B_2;
                    accum3_0 += A_0 * B_3;
                    accum4_0 += A_0 * B_4;
                    accum5_0 += A_0 * B_5;
                    accum6_0 += A_0 * B_6;
                    accum7_0 += A_0 * B_7;
                    accum8_0 += A_0 * B_8;
                    accum9_0 += A_0 * B_9;
                    accum10_0 += A_0 * B_10;
                    accum11_0 += A_0 * B_11;
                    accum12_0 += A_0 * B_12;
                    accum13_0 += A_0 * B_13;
                    accum14_0 += A_0 * B_14;
                    accum15_0 += A_0 * B_15;

                    float A_1 = StrideAddress(Akp, nstrideA, 1)[0];
                    accum0_1 += A_1 * B_0;
                    accum1_1 += A_1 * B_1;
                    accum2_1 += A_1 * B_2;
                    accum3_1 += A_1 * B_3;
                    accum4_1 += A_1 * B_4;
                    accum5_1 += A_1 * B_5;
                    accum6_1 += A_1 * B_6;
                    accum7_1 += A_1 * B_7;
                    accum8_1 += A_1 * B_8;
                    accum9_1 += A_1 * B_9;
                    accum10_1 += A_1 * B_10;
                    accum11_1 += A_1 * B_11;
                    accum12_1 += A_1 * B_12;
                    accum13_1 += A_1 * B_13;
                    accum14_1 += A_1 * B_14;
                    accum15_1 += A_1 * B_15;

                    Akp += 1;
                    Bkp += nstrideB.ToInt64();
                }

                StrideAddress(Cnp, nstrideC, 0)[0] = accum0_0;
                StrideAddress(Cnp, nstrideC, 0)[1] = accum1_0;
                StrideAddress(Cnp, nstrideC, 0)[2] = accum2_0;
                StrideAddress(Cnp, nstrideC, 0)[3] = accum3_0;
                StrideAddress(Cnp, nstrideC, 0)[4] = accum4_0;
                StrideAddress(Cnp, nstrideC, 0)[5] = accum5_0;
                StrideAddress(Cnp, nstrideC, 0)[6] = accum6_0;
                StrideAddress(Cnp, nstrideC, 0)[7] = accum7_0;
                StrideAddress(Cnp, nstrideC, 0)[8] = accum8_0;
                StrideAddress(Cnp, nstrideC, 0)[9] = accum9_0;
                StrideAddress(Cnp, nstrideC, 0)[10] = accum10_0;
                StrideAddress(Cnp, nstrideC, 0)[11] = accum11_0;
                StrideAddress(Cnp, nstrideC, 0)[12] = accum12_0;
                StrideAddress(Cnp, nstrideC, 0)[13] = accum13_0;
                StrideAddress(Cnp, nstrideC, 0)[14] = accum14_0;
                StrideAddress(Cnp, nstrideC, 0)[15] = accum15_0;
                
                StrideAddress(Cnp, nstrideC, 1)[0] = accum0_1;
                StrideAddress(Cnp, nstrideC, 1)[1] = accum1_1;
                StrideAddress(Cnp, nstrideC, 1)[2] = accum2_1;
                StrideAddress(Cnp, nstrideC, 1)[3] = accum3_1;
                StrideAddress(Cnp, nstrideC, 1)[4] = accum4_1;
                StrideAddress(Cnp, nstrideC, 1)[5] = accum5_1;
                StrideAddress(Cnp, nstrideC, 1)[6] = accum6_1;
                StrideAddress(Cnp, nstrideC, 1)[7] = accum7_1;
                StrideAddress(Cnp, nstrideC, 1)[8] = accum8_1;
                StrideAddress(Cnp, nstrideC, 1)[9] = accum9_1;
                StrideAddress(Cnp, nstrideC, 1)[10] = accum10_1;
                StrideAddress(Cnp, nstrideC, 1)[11] = accum11_1;
                StrideAddress(Cnp, nstrideC, 1)[12] = accum12_1;
                StrideAddress(Cnp, nstrideC, 1)[13] = accum13_1;
                StrideAddress(Cnp, nstrideC, 1)[14] = accum14_1;
                StrideAddress(Cnp, nstrideC, 1)[15] = accum15_1;
                
                n -= 16;
                Bnp += 16;
                Cnp += 16;
            }
            if (n >= 8)
            {
                float accum0_0 = 0.0f;
                float accum1_0 = 0.0f;
                float accum2_0 = 0.0f;
                float accum3_0 = 0.0f;
                float accum4_0 = 0.0f;
                float accum5_0 = 0.0f;
                float accum6_0 = 0.0f;
                float accum7_0 = 0.0f;

                float accum0_1 = 0.0f;
                float accum1_1 = 0.0f;
                float accum2_1 = 0.0f;
                float accum3_1 = 0.0f;
                float accum4_1 = 0.0f;
                float accum5_1 = 0.0f;
                float accum6_1 = 0.0f;
                float accum7_1 = 0.0f;

                if (accumulateC)
                {
                    accum0_0 = StrideAddress(Cnp, nstrideC, 0)[0];
                    accum1_0 = StrideAddress(Cnp, nstrideC, 0)[1];
                    accum2_0 = StrideAddress(Cnp, nstrideC, 0)[2];
                    accum3_0 = StrideAddress(Cnp, nstrideC, 0)[3];
                    accum4_0 = StrideAddress(Cnp, nstrideC, 0)[4];
                    accum5_0 = StrideAddress(Cnp, nstrideC, 0)[5];
                    accum6_0 = StrideAddress(Cnp, nstrideC, 0)[6];
                    accum7_0 = StrideAddress(Cnp, nstrideC, 0)[7];

                    accum0_1 = StrideAddress(Cnp, nstrideC, 1)[0];
                    accum1_1 = StrideAddress(Cnp, nstrideC, 1)[1];
                    accum2_1 = StrideAddress(Cnp, nstrideC, 1)[2];
                    accum3_1 = StrideAddress(Cnp, nstrideC, 1)[3];
                    accum4_1 = StrideAddress(Cnp, nstrideC, 1)[4];
                    accum5_1 = StrideAddress(Cnp, nstrideC, 1)[5];
                    accum6_1 = StrideAddress(Cnp, nstrideC, 1)[6];
                    accum7_1 = StrideAddress(Cnp, nstrideC, 1)[7];
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                var k = (uint)K;

                for (; k > 0; k -= 1)
                {
                    float B_0 = Bkp[0];
                    float B_1 = Bkp[1];
                    float B_2 = Bkp[2];
                    float B_3 = Bkp[3];
                    float B_4 = Bkp[4];
                    float B_5 = Bkp[5];
                    float B_6 = Bkp[6];
                    float B_7 = Bkp[7];

                    float A_0 = StrideAddress(Akp, nstrideA, 0)[0];
                    accum0_0 += A_0 * B_0;
                    accum1_0 += A_0 * B_1;
                    accum2_0 += A_0 * B_2;
                    accum3_0 += A_0 * B_3;
                    accum4_0 += A_0 * B_4;
                    accum5_0 += A_0 * B_5;
                    accum6_0 += A_0 * B_6;
                    accum7_0 += A_0 * B_7;

                    float A_1 = StrideAddress(Akp, nstrideA, 1)[0];
                    accum0_1 += A_1 * B_0;
                    accum1_1 += A_1 * B_1;
                    accum2_1 += A_1 * B_2;
                    accum3_1 += A_1 * B_3;
                    accum4_1 += A_1 * B_4;
                    accum5_1 += A_1 * B_5;
                    accum6_1 += A_1 * B_6;
                    accum7_1 += A_1 * B_7;

                    Akp += 1;
                    Bkp += nstrideB.ToInt64();
                }

                StrideAddress(Cnp, nstrideC, 0)[0] = accum0_0;
                StrideAddress(Cnp, nstrideC, 0)[1] = accum1_0;
                StrideAddress(Cnp, nstrideC, 0)[2] = accum2_0;
                StrideAddress(Cnp, nstrideC, 0)[3] = accum3_0;
                StrideAddress(Cnp, nstrideC, 0)[4] = accum4_0;
                StrideAddress(Cnp, nstrideC, 0)[5] = accum5_0;
                StrideAddress(Cnp, nstrideC, 0)[6] = accum6_0;
                StrideAddress(Cnp, nstrideC, 0)[7] = accum7_0;
                
                StrideAddress(Cnp, nstrideC, 1)[0] = accum0_1;
                StrideAddress(Cnp, nstrideC, 1)[1] = accum1_1;
                StrideAddress(Cnp, nstrideC, 1)[2] = accum2_1;
                StrideAddress(Cnp, nstrideC, 1)[3] = accum3_1;
                StrideAddress(Cnp, nstrideC, 1)[4] = accum4_1;
                StrideAddress(Cnp, nstrideC, 1)[5] = accum5_1;
                StrideAddress(Cnp, nstrideC, 1)[6] = accum6_1;
                StrideAddress(Cnp, nstrideC, 1)[7] = accum7_1;
                
                n -= 8;
                Bnp += 8;
                Cnp += 8;
            }

            Ap = StrideAddress(Ap, nstrideA, 2);
            Cp = StrideAddress(Cp, nstrideC, 2);
            M -= 2;
        }
        if (M >= 1)
        {
            float *Bnp = Bp;
            float *Cnp = Cp;
            var n = (uint)N;

            while (n >= 16)
            {
                float accum0_0 = 0.0f;
                float accum1_0 = 0.0f;
                float accum2_0 = 0.0f;
                float accum3_0 = 0.0f;
                float accum4_0 = 0.0f;
                float accum5_0 = 0.0f;
                float accum6_0 = 0.0f;
                float accum7_0 = 0.0f;
                float accum8_0 = 0.0f;
                float accum9_0 = 0.0f;
                float accum10_0 = 0.0f;
                float accum11_0 = 0.0f;
                float accum12_0 = 0.0f;
                float accum13_0 = 0.0f;
                float accum14_0 = 0.0f;
                float accum15_0 = 0.0f;

                if (accumulateC)
                {
                    accum0_0 = StrideAddress(Cnp, nstrideC, 0)[0];
                    accum1_0 = StrideAddress(Cnp, nstrideC, 0)[1];
                    accum2_0 = StrideAddress(Cnp, nstrideC, 0)[2];
                    accum3_0 = StrideAddress(Cnp, nstrideC, 0)[3];
                    accum4_0 = StrideAddress(Cnp, nstrideC, 0)[4];
                    accum5_0 = StrideAddress(Cnp, nstrideC, 0)[5];
                    accum6_0 = StrideAddress(Cnp, nstrideC, 0)[6];
                    accum7_0 = StrideAddress(Cnp, nstrideC, 0)[7];
                    accum8_0 = StrideAddress(Cnp, nstrideC, 0)[8];
                    accum9_0 = StrideAddress(Cnp, nstrideC, 0)[9];
                    accum10_0 = StrideAddress(Cnp, nstrideC, 0)[10];
                    accum11_0 = StrideAddress(Cnp, nstrideC, 0)[11];
                    accum12_0 = StrideAddress(Cnp, nstrideC, 0)[12];
                    accum13_0 = StrideAddress(Cnp, nstrideC, 0)[13];
                    accum14_0 = StrideAddress(Cnp, nstrideC, 0)[14];
                    accum15_0 = StrideAddress(Cnp, nstrideC, 0)[15];
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                var k = (uint)K;

                for (; k > 0; k -= 1)
                {
                    float B_0 = Bkp[0];
                    float B_1 = Bkp[1];
                    float B_2 = Bkp[2];
                    float B_3 = Bkp[3];
                    float B_4 = Bkp[4];
                    float B_5 = Bkp[5];
                    float B_6 = Bkp[6];
                    float B_7 = Bkp[7];
                    float B_8 = Bkp[8];
                    float B_9 = Bkp[9];
                    float B_10 = Bkp[10];
                    float B_11 = Bkp[11];
                    float B_12 = Bkp[12];
                    float B_13 = Bkp[13];
                    float B_14 = Bkp[14];
                    float B_15 = Bkp[15];

                    float A_0 = StrideAddress(Akp, nstrideA, 0)[0];
                    accum0_0 += A_0 * B_0;
                    accum1_0 += A_0 * B_1;
                    accum2_0 += A_0 * B_2;
                    accum3_0 += A_0 * B_3;
                    accum4_0 += A_0 * B_4;
                    accum5_0 += A_0 * B_5;
                    accum6_0 += A_0 * B_6;
                    accum7_0 += A_0 * B_7;
                    accum8_0 += A_0 * B_8;
                    accum9_0 += A_0 * B_9;
                    accum10_0 += A_0 * B_10;
                    accum11_0 += A_0 * B_11;
                    accum12_0 += A_0 * B_12;
                    accum13_0 += A_0 * B_13;
                    accum14_0 += A_0 * B_14;
                    accum15_0 += A_0 * B_15;

                    Akp += 1;
                    Bkp += nstrideB.ToInt64();
                }

                StrideAddress(Cnp, nstrideC, 0)[0] = accum0_0;
                StrideAddress(Cnp, nstrideC, 0)[1] = accum1_0;
                StrideAddress(Cnp, nstrideC, 0)[2] = accum2_0;
                StrideAddress(Cnp, nstrideC, 0)[3] = accum3_0;
                StrideAddress(Cnp, nstrideC, 0)[4] = accum4_0;
                StrideAddress(Cnp, nstrideC, 0)[5] = accum5_0;
                StrideAddress(Cnp, nstrideC, 0)[6] = accum6_0;
                StrideAddress(Cnp, nstrideC, 0)[7] = accum7_0;
                StrideAddress(Cnp, nstrideC, 0)[8] = accum8_0;
                StrideAddress(Cnp, nstrideC, 0)[9] = accum9_0;
                StrideAddress(Cnp, nstrideC, 0)[10] = accum10_0;
                StrideAddress(Cnp, nstrideC, 0)[11] = accum11_0;
                StrideAddress(Cnp, nstrideC, 0)[12] = accum12_0;
                StrideAddress(Cnp, nstrideC, 0)[13] = accum13_0;
                StrideAddress(Cnp, nstrideC, 0)[14] = accum14_0;
                StrideAddress(Cnp, nstrideC, 0)[15] = accum15_0;
                
                n -= 16;
                Bnp += 16;
                Cnp += 16;
            }
            if (n >= 8)
            {
                float accum0_0 = 0.0f;
                float accum1_0 = 0.0f;
                float accum2_0 = 0.0f;
                float accum3_0 = 0.0f;
                float accum4_0 = 0.0f;
                float accum5_0 = 0.0f;
                float accum6_0 = 0.0f;
                float accum7_0 = 0.0f;

                if (accumulateC)
                {
                    accum0_0 = StrideAddress(Cnp, nstrideC, 0)[0];
                    accum1_0 = StrideAddress(Cnp, nstrideC, 0)[1];
                    accum2_0 = StrideAddress(Cnp, nstrideC, 0)[2];
                    accum3_0 = StrideAddress(Cnp, nstrideC, 0)[3];
                    accum4_0 = StrideAddress(Cnp, nstrideC, 0)[4];
                    accum5_0 = StrideAddress(Cnp, nstrideC, 0)[5];
                    accum6_0 = StrideAddress(Cnp, nstrideC, 0)[6];
                    accum7_0 = StrideAddress(Cnp, nstrideC, 0)[7];
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                var k = (uint)K;

                for (; k > 0; k -= 1)
                {
                    float B_0 = Bkp[0];
                    float B_1 = Bkp[1];
                    float B_2 = Bkp[2];
                    float B_3 = Bkp[3];
                    float B_4 = Bkp[4];
                    float B_5 = Bkp[5];
                    float B_6 = Bkp[6];
                    float B_7 = Bkp[7];

                    float A_0 = StrideAddress(Akp, nstrideA, 0)[0];
                    accum0_0 += A_0 * B_0;
                    accum1_0 += A_0 * B_1;
                    accum2_0 += A_0 * B_2;
                    accum3_0 += A_0 * B_3;
                    accum4_0 += A_0 * B_4;
                    accum5_0 += A_0 * B_5;
                    accum6_0 += A_0 * B_6;
                    accum7_0 += A_0 * B_7;

                    Akp += 1;
                    Bkp += nstrideB.ToInt64();
                }

                StrideAddress(Cnp, nstrideC, 0)[0] = accum0_0;
                StrideAddress(Cnp, nstrideC, 0)[1] = accum1_0;
                StrideAddress(Cnp, nstrideC, 0)[2] = accum2_0;
                StrideAddress(Cnp, nstrideC, 0)[3] = accum3_0;
                StrideAddress(Cnp, nstrideC, 0)[4] = accum4_0;
                StrideAddress(Cnp, nstrideC, 0)[5] = accum5_0;
                StrideAddress(Cnp, nstrideC, 0)[6] = accum6_0;
                StrideAddress(Cnp, nstrideC, 0)[7] = accum7_0;
                
                n -= 8;
                Bnp += 8;
                Cnp += 8;
            }

            Ap = StrideAddress(Ap, nstrideA, 1);
            Cp = StrideAddress(Cp, nstrideC, 1);
            M -= 1;
        }
    }
#else
    static unsafe void MultiplyBlockUnroll1x16(
        [NoAlias] float* Ap, int strideA,
        [NoAlias] float* Bp, int strideB,
        [NoAlias] float* Cp, int strideC,
        int M, int N, int K, bool accumulateC)
    {
        // Avoid generating compiled code for the case where M/N/K are negative or zero.
        Hint.Assume(M > 0);
        Hint.Assume(N > 0);
        Hint.Assume(K > 0);

        // Help the compiler produce better code by using the strides as pointer-width values.
        // For x64, the inner loop is then able to use scaled indexing modes to access the rows
        // of matrix A, instead of wasting extra registers to store the offsets.
        var nstrideA = new System.IntPtr(strideA);
        var nstrideB = new System.IntPtr(strideB);
        var nstrideC = new System.IntPtr(strideC);

        while (M >= 1)
        {
            float *Bnp = Bp;
            float *Cnp = Cp;
            var n = (uint)N;

            while (n >= 16)
            {
                v256 accum0_0 = new v256();
                v256 accum1_0 = new v256();

                if (accumulateC)
                {
                    float* Cnp_0 = Cnp;
                    accum0_0 = *(v256*)(Cnp_0 + 0);
                    accum1_0 = *(v256*)(Cnp_0 + 8);
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                float *AkpEnd = Akp + (uint)K;

                do
                {
                    v256 B_0 = *(v256*)&Bkp[0];
                    v256 B_1 = *(v256*)&Bkp[8];

                    v256 A_0 = new v256(*Akp);
                    accum0_0 = VectorUtils.MulAdd(A_0, B_0, accum0_0);
                    accum1_0 = VectorUtils.MulAdd(A_0, B_1, accum1_0);

                    Akp += 1;
                    Bkp = (float*)StrideAddress(Bkp, nstrideB, 1);
                }
                while (Akp < AkpEnd);

                {
                    float* Cnp_0 = Cnp;
                    *(v256*)(Cnp_0 + 0) = accum0_0;
                    *(v256*)(Cnp_0 + 8) = accum1_0;
                }

                n -= 16;
                Bnp += 16;
                Cnp += 16;
            }
            if (n >= 8)
            {
                v256 accum0_0 = new v256();

                if (accumulateC)
                {
                    float* Cnp_0 = Cnp;
                    accum0_0 = *(v256*)(Cnp_0 + 0);
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                float *AkpEnd = Akp + (uint)K;

                do
                {
                    v256 B_0 = *(v256*)&Bkp[0];

                    v256 A_0 = new v256(*Akp);
                    accum0_0 = VectorUtils.MulAdd(A_0, B_0, accum0_0);

                    Akp += 1;
                    Bkp = (float*)StrideAddress(Bkp, nstrideB, 1);
                }
                while (Akp < AkpEnd);

                {
                    float* Cnp_0 = Cnp;
                    *(v256*)(Cnp_0 + 0) = accum0_0;
                }

                n -= 8;
                Bnp += 8;
                Cnp += 8;
            }

            Ap = StrideAddress(Ap, nstrideA, 1);
            Cp = StrideAddress(Cp, nstrideC, 1);
            M -= 1;
        }
    }
    static unsafe void MultiplyBlockUnroll2x16(
        [NoAlias] float* Ap, int strideA,
        [NoAlias] float* Bp, int strideB,
        [NoAlias] float* Cp, int strideC,
        int M, int N, int K, bool accumulateC)
    {
        // Avoid generating compiled code for the case where M/N/K are negative or zero.
        Hint.Assume(M > 0);
        Hint.Assume(N > 0);
        Hint.Assume(K > 0);

        // Help the compiler produce better code by using the strides as pointer-width values.
        // For x64, the inner loop is then able to use scaled indexing modes to access the rows
        // of matrix A, instead of wasting extra registers to store the offsets.
        var nstrideA = new System.IntPtr(strideA);
        var nstrideB = new System.IntPtr(strideB);
        var nstrideC = new System.IntPtr(strideC);

        while (M >= 2)
        {
            float *Bnp = Bp;
            float *Cnp = Cp;
            var n = (uint)N;

            while (n >= 16)
            {
                v256 accum0_0 = new v256();
                v256 accum1_0 = new v256();
                v256 accum0_1 = new v256();
                v256 accum1_1 = new v256();

                if (accumulateC)
                {
                    float* Cnp_0 = Cnp;
                    accum0_0 = *(v256*)(Cnp_0 + 0);
                    accum1_0 = *(v256*)(Cnp_0 + 8);
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    accum0_1 = *(v256*)(Cnp_1 + 0);
                    accum1_1 = *(v256*)(Cnp_1 + 8);
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                float *AkpEnd = Akp + (uint)K;

                do
                {
                    v256 B_0 = *(v256*)&Bkp[0];
                    v256 B_1 = *(v256*)&Bkp[8];

                    v256 A_0 = new v256(*Akp);
                    accum0_0 = VectorUtils.MulAdd(A_0, B_0, accum0_0);
                    accum1_0 = VectorUtils.MulAdd(A_0, B_1, accum1_0);
                    v256 A_1 = new v256(*StrideAddress(Akp, nstrideA, 1));
                    accum0_1 = VectorUtils.MulAdd(A_1, B_0, accum0_1);
                    accum1_1 = VectorUtils.MulAdd(A_1, B_1, accum1_1);

                    Akp += 1;
                    Bkp = (float*)StrideAddress(Bkp, nstrideB, 1);
                }
                while (Akp < AkpEnd);

                {
                    float* Cnp_0 = Cnp;
                    *(v256*)(Cnp_0 + 0) = accum0_0;
                    *(v256*)(Cnp_0 + 8) = accum1_0;
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    *(v256*)(Cnp_1 + 0) = accum0_1;
                    *(v256*)(Cnp_1 + 8) = accum1_1;
                }

                n -= 16;
                Bnp += 16;
                Cnp += 16;
            }
            if (n >= 8)
            {
                v256 accum0_0 = new v256();
                v256 accum0_1 = new v256();

                if (accumulateC)
                {
                    float* Cnp_0 = Cnp;
                    accum0_0 = *(v256*)(Cnp_0 + 0);
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    accum0_1 = *(v256*)(Cnp_1 + 0);
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                float *AkpEnd = Akp + (uint)K;

                do
                {
                    v256 B_0 = *(v256*)&Bkp[0];

                    v256 A_0 = new v256(*Akp);
                    accum0_0 = VectorUtils.MulAdd(A_0, B_0, accum0_0);
                    v256 A_1 = new v256(*StrideAddress(Akp, nstrideA, 1));
                    accum0_1 = VectorUtils.MulAdd(A_1, B_0, accum0_1);

                    Akp += 1;
                    Bkp = (float*)StrideAddress(Bkp, nstrideB, 1);
                }
                while (Akp < AkpEnd);

                {
                    float* Cnp_0 = Cnp;
                    *(v256*)(Cnp_0 + 0) = accum0_0;
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    *(v256*)(Cnp_1 + 0) = accum0_1;
                }

                n -= 8;
                Bnp += 8;
                Cnp += 8;
            }

            Ap = StrideAddress(Ap, nstrideA, 2);
            Cp = StrideAddress(Cp, nstrideC, 2);
            M -= 2;
        }
        if (M >= 1)
        {
            float *Bnp = Bp;
            float *Cnp = Cp;
            var n = (uint)N;

            while (n >= 16)
            {
                v256 accum0_0 = new v256();
                v256 accum1_0 = new v256();

                if (accumulateC)
                {
                    float* Cnp_0 = Cnp;
                    accum0_0 = *(v256*)(Cnp_0 + 0);
                    accum1_0 = *(v256*)(Cnp_0 + 8);
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                float *AkpEnd = Akp + (uint)K;

                do
                {
                    v256 B_0 = *(v256*)&Bkp[0];
                    v256 B_1 = *(v256*)&Bkp[8];

                    v256 A_0 = new v256(*Akp);
                    accum0_0 = VectorUtils.MulAdd(A_0, B_0, accum0_0);
                    accum1_0 = VectorUtils.MulAdd(A_0, B_1, accum1_0);

                    Akp += 1;
                    Bkp = (float*)StrideAddress(Bkp, nstrideB, 1);
                }
                while (Akp < AkpEnd);

                {
                    float* Cnp_0 = Cnp;
                    *(v256*)(Cnp_0 + 0) = accum0_0;
                    *(v256*)(Cnp_0 + 8) = accum1_0;
                }

                n -= 16;
                Bnp += 16;
                Cnp += 16;
            }
            if (n >= 8)
            {
                v256 accum0_0 = new v256();

                if (accumulateC)
                {
                    float* Cnp_0 = Cnp;
                    accum0_0 = *(v256*)(Cnp_0 + 0);
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                float *AkpEnd = Akp + (uint)K;

                do
                {
                    v256 B_0 = *(v256*)&Bkp[0];

                    v256 A_0 = new v256(*Akp);
                    accum0_0 = VectorUtils.MulAdd(A_0, B_0, accum0_0);

                    Akp += 1;
                    Bkp = (float*)StrideAddress(Bkp, nstrideB, 1);
                }
                while (Akp < AkpEnd);

                {
                    float* Cnp_0 = Cnp;
                    *(v256*)(Cnp_0 + 0) = accum0_0;
                }

                n -= 8;
                Bnp += 8;
                Cnp += 8;
            }

            Ap = StrideAddress(Ap, nstrideA, 1);
            Cp = StrideAddress(Cp, nstrideC, 1);
            M -= 1;
        }
    }
    static unsafe void MultiplyBlockUnroll4x16(
        [NoAlias] float* Ap, int strideA,
        [NoAlias] float* Bp, int strideB,
        [NoAlias] float* Cp, int strideC,
        int M, int N, int K, bool accumulateC)
    {
        // Avoid generating compiled code for the case where M/N/K are negative or zero.
        Hint.Assume(M > 0);
        Hint.Assume(N > 0);
        Hint.Assume(K > 0);

        // Help the compiler produce better code by using the strides as pointer-width values.
        // For x64, the inner loop is then able to use scaled indexing modes to access the rows
        // of matrix A, instead of wasting extra registers to store the offsets.
        var nstrideA = new System.IntPtr(strideA);
        var nstrideB = new System.IntPtr(strideB);
        var nstrideC = new System.IntPtr(strideC);

        while (M >= 4)
        {
            float *Bnp = Bp;
            float *Cnp = Cp;
            var n = (uint)N;

            while (n >= 16)
            {
                v256 accum0_0 = new v256();
                v256 accum1_0 = new v256();
                v256 accum0_1 = new v256();
                v256 accum1_1 = new v256();
                v256 accum0_2 = new v256();
                v256 accum1_2 = new v256();
                v256 accum0_3 = new v256();
                v256 accum1_3 = new v256();

                if (accumulateC)
                {
                    float* Cnp_0 = Cnp;
                    accum0_0 = *(v256*)(Cnp_0 + 0);
                    accum1_0 = *(v256*)(Cnp_0 + 8);
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    accum0_1 = *(v256*)(Cnp_1 + 0);
                    accum1_1 = *(v256*)(Cnp_1 + 8);
                    float* Cnp_2 = StrideAddress(Cnp, nstrideC, 2);
                    accum0_2 = *(v256*)(Cnp_2 + 0);
                    accum1_2 = *(v256*)(Cnp_2 + 8);
                    float* Cnp_3 = StrideAddress(Cnp, nstrideC, 3);
                    accum0_3 = *(v256*)(Cnp_3 + 0);
                    accum1_3 = *(v256*)(Cnp_3 + 8);
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                float *AkpEnd = Akp + (uint)K;

                do
                {
                    v256 B_0 = *(v256*)&Bkp[0];
                    v256 B_1 = *(v256*)&Bkp[8];

                    v256 A_0 = new v256(*Akp);
                    accum0_0 = VectorUtils.MulAdd(A_0, B_0, accum0_0);
                    accum1_0 = VectorUtils.MulAdd(A_0, B_1, accum1_0);
                    v256 A_1 = new v256(*StrideAddress(Akp, nstrideA, 1));
                    accum0_1 = VectorUtils.MulAdd(A_1, B_0, accum0_1);
                    accum1_1 = VectorUtils.MulAdd(A_1, B_1, accum1_1);
                    v256 A_2 = new v256(*StrideAddress(Akp, nstrideA, 2));
                    accum0_2 = VectorUtils.MulAdd(A_2, B_0, accum0_2);
                    accum1_2 = VectorUtils.MulAdd(A_2, B_1, accum1_2);
                    v256 A_3 = new v256(*StrideAddress(Akp, nstrideA, 3));
                    accum0_3 = VectorUtils.MulAdd(A_3, B_0, accum0_3);
                    accum1_3 = VectorUtils.MulAdd(A_3, B_1, accum1_3);

                    Akp += 1;
                    Bkp = (float*)StrideAddress(Bkp, nstrideB, 1);
                }
                while (Akp < AkpEnd);

                {
                    float* Cnp_0 = Cnp;
                    *(v256*)(Cnp_0 + 0) = accum0_0;
                    *(v256*)(Cnp_0 + 8) = accum1_0;
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    *(v256*)(Cnp_1 + 0) = accum0_1;
                    *(v256*)(Cnp_1 + 8) = accum1_1;
                    float* Cnp_2 = StrideAddress(Cnp, nstrideC, 2);
                    *(v256*)(Cnp_2 + 0) = accum0_2;
                    *(v256*)(Cnp_2 + 8) = accum1_2;
                    float* Cnp_3 = StrideAddress(Cnp, nstrideC, 3);
                    *(v256*)(Cnp_3 + 0) = accum0_3;
                    *(v256*)(Cnp_3 + 8) = accum1_3;
                }

                n -= 16;
                Bnp += 16;
                Cnp += 16;
            }
            if (n >= 8)
            {
                v256 accum0_0 = new v256();
                v256 accum0_1 = new v256();
                v256 accum0_2 = new v256();
                v256 accum0_3 = new v256();

                if (accumulateC)
                {
                    float* Cnp_0 = Cnp;
                    accum0_0 = *(v256*)(Cnp_0 + 0);
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    accum0_1 = *(v256*)(Cnp_1 + 0);
                    float* Cnp_2 = StrideAddress(Cnp, nstrideC, 2);
                    accum0_2 = *(v256*)(Cnp_2 + 0);
                    float* Cnp_3 = StrideAddress(Cnp, nstrideC, 3);
                    accum0_3 = *(v256*)(Cnp_3 + 0);
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                float *AkpEnd = Akp + (uint)K;

                do
                {
                    v256 B_0 = *(v256*)&Bkp[0];

                    v256 A_0 = new v256(*Akp);
                    accum0_0 = VectorUtils.MulAdd(A_0, B_0, accum0_0);
                    v256 A_1 = new v256(*StrideAddress(Akp, nstrideA, 1));
                    accum0_1 = VectorUtils.MulAdd(A_1, B_0, accum0_1);
                    v256 A_2 = new v256(*StrideAddress(Akp, nstrideA, 2));
                    accum0_2 = VectorUtils.MulAdd(A_2, B_0, accum0_2);
                    v256 A_3 = new v256(*StrideAddress(Akp, nstrideA, 3));
                    accum0_3 = VectorUtils.MulAdd(A_3, B_0, accum0_3);

                    Akp += 1;
                    Bkp = (float*)StrideAddress(Bkp, nstrideB, 1);
                }
                while (Akp < AkpEnd);

                {
                    float* Cnp_0 = Cnp;
                    *(v256*)(Cnp_0 + 0) = accum0_0;
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    *(v256*)(Cnp_1 + 0) = accum0_1;
                    float* Cnp_2 = StrideAddress(Cnp, nstrideC, 2);
                    *(v256*)(Cnp_2 + 0) = accum0_2;
                    float* Cnp_3 = StrideAddress(Cnp, nstrideC, 3);
                    *(v256*)(Cnp_3 + 0) = accum0_3;
                }

                n -= 8;
                Bnp += 8;
                Cnp += 8;
            }

            Ap = StrideAddress(Ap, nstrideA, 4);
            Cp = StrideAddress(Cp, nstrideC, 4);
            M -= 4;
        }
        if (M >= 3)
        {
            float *Bnp = Bp;
            float *Cnp = Cp;
            var n = (uint)N;

            while (n >= 16)
            {
                v256 accum0_0 = new v256();
                v256 accum1_0 = new v256();
                v256 accum0_1 = new v256();
                v256 accum1_1 = new v256();
                v256 accum0_2 = new v256();
                v256 accum1_2 = new v256();

                if (accumulateC)
                {
                    float* Cnp_0 = Cnp;
                    accum0_0 = *(v256*)(Cnp_0 + 0);
                    accum1_0 = *(v256*)(Cnp_0 + 8);
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    accum0_1 = *(v256*)(Cnp_1 + 0);
                    accum1_1 = *(v256*)(Cnp_1 + 8);
                    float* Cnp_2 = StrideAddress(Cnp, nstrideC, 2);
                    accum0_2 = *(v256*)(Cnp_2 + 0);
                    accum1_2 = *(v256*)(Cnp_2 + 8);
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                float *AkpEnd = Akp + (uint)K;

                do
                {
                    v256 B_0 = *(v256*)&Bkp[0];
                    v256 B_1 = *(v256*)&Bkp[8];

                    v256 A_0 = new v256(*Akp);
                    accum0_0 = VectorUtils.MulAdd(A_0, B_0, accum0_0);
                    accum1_0 = VectorUtils.MulAdd(A_0, B_1, accum1_0);
                    v256 A_1 = new v256(*StrideAddress(Akp, nstrideA, 1));
                    accum0_1 = VectorUtils.MulAdd(A_1, B_0, accum0_1);
                    accum1_1 = VectorUtils.MulAdd(A_1, B_1, accum1_1);
                    v256 A_2 = new v256(*StrideAddress(Akp, nstrideA, 2));
                    accum0_2 = VectorUtils.MulAdd(A_2, B_0, accum0_2);
                    accum1_2 = VectorUtils.MulAdd(A_2, B_1, accum1_2);

                    Akp += 1;
                    Bkp = (float*)StrideAddress(Bkp, nstrideB, 1);
                }
                while (Akp < AkpEnd);

                {
                    float* Cnp_0 = Cnp;
                    *(v256*)(Cnp_0 + 0) = accum0_0;
                    *(v256*)(Cnp_0 + 8) = accum1_0;
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    *(v256*)(Cnp_1 + 0) = accum0_1;
                    *(v256*)(Cnp_1 + 8) = accum1_1;
                    float* Cnp_2 = StrideAddress(Cnp, nstrideC, 2);
                    *(v256*)(Cnp_2 + 0) = accum0_2;
                    *(v256*)(Cnp_2 + 8) = accum1_2;
                }

                n -= 16;
                Bnp += 16;
                Cnp += 16;
            }
            if (n >= 8)
            {
                v256 accum0_0 = new v256();
                v256 accum0_1 = new v256();
                v256 accum0_2 = new v256();

                if (accumulateC)
                {
                    float* Cnp_0 = Cnp;
                    accum0_0 = *(v256*)(Cnp_0 + 0);
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    accum0_1 = *(v256*)(Cnp_1 + 0);
                    float* Cnp_2 = StrideAddress(Cnp, nstrideC, 2);
                    accum0_2 = *(v256*)(Cnp_2 + 0);
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                float *AkpEnd = Akp + (uint)K;

                do
                {
                    v256 B_0 = *(v256*)&Bkp[0];

                    v256 A_0 = new v256(*Akp);
                    accum0_0 = VectorUtils.MulAdd(A_0, B_0, accum0_0);
                    v256 A_1 = new v256(*StrideAddress(Akp, nstrideA, 1));
                    accum0_1 = VectorUtils.MulAdd(A_1, B_0, accum0_1);
                    v256 A_2 = new v256(*StrideAddress(Akp, nstrideA, 2));
                    accum0_2 = VectorUtils.MulAdd(A_2, B_0, accum0_2);

                    Akp += 1;
                    Bkp = (float*)StrideAddress(Bkp, nstrideB, 1);
                }
                while (Akp < AkpEnd);

                {
                    float* Cnp_0 = Cnp;
                    *(v256*)(Cnp_0 + 0) = accum0_0;
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    *(v256*)(Cnp_1 + 0) = accum0_1;
                    float* Cnp_2 = StrideAddress(Cnp, nstrideC, 2);
                    *(v256*)(Cnp_2 + 0) = accum0_2;
                }

                n -= 8;
                Bnp += 8;
                Cnp += 8;
            }

            Ap = StrideAddress(Ap, nstrideA, 3);
            Cp = StrideAddress(Cp, nstrideC, 3);
            M -= 3;
        }
        if (M >= 2)
        {
            float *Bnp = Bp;
            float *Cnp = Cp;
            var n = (uint)N;

            while (n >= 16)
            {
                v256 accum0_0 = new v256();
                v256 accum1_0 = new v256();
                v256 accum0_1 = new v256();
                v256 accum1_1 = new v256();

                if (accumulateC)
                {
                    float* Cnp_0 = Cnp;
                    accum0_0 = *(v256*)(Cnp_0 + 0);
                    accum1_0 = *(v256*)(Cnp_0 + 8);
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    accum0_1 = *(v256*)(Cnp_1 + 0);
                    accum1_1 = *(v256*)(Cnp_1 + 8);
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                float *AkpEnd = Akp + (uint)K;

                do
                {
                    v256 B_0 = *(v256*)&Bkp[0];
                    v256 B_1 = *(v256*)&Bkp[8];

                    v256 A_0 = new v256(*Akp);
                    accum0_0 = VectorUtils.MulAdd(A_0, B_0, accum0_0);
                    accum1_0 = VectorUtils.MulAdd(A_0, B_1, accum1_0);
                    v256 A_1 = new v256(*StrideAddress(Akp, nstrideA, 1));
                    accum0_1 = VectorUtils.MulAdd(A_1, B_0, accum0_1);
                    accum1_1 = VectorUtils.MulAdd(A_1, B_1, accum1_1);

                    Akp += 1;
                    Bkp = (float*)StrideAddress(Bkp, nstrideB, 1);
                }
                while (Akp < AkpEnd);

                {
                    float* Cnp_0 = Cnp;
                    *(v256*)(Cnp_0 + 0) = accum0_0;
                    *(v256*)(Cnp_0 + 8) = accum1_0;
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    *(v256*)(Cnp_1 + 0) = accum0_1;
                    *(v256*)(Cnp_1 + 8) = accum1_1;
                }

                n -= 16;
                Bnp += 16;
                Cnp += 16;
            }
            if (n >= 8)
            {
                v256 accum0_0 = new v256();
                v256 accum0_1 = new v256();

                if (accumulateC)
                {
                    float* Cnp_0 = Cnp;
                    accum0_0 = *(v256*)(Cnp_0 + 0);
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    accum0_1 = *(v256*)(Cnp_1 + 0);
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                float *AkpEnd = Akp + (uint)K;

                do
                {
                    v256 B_0 = *(v256*)&Bkp[0];

                    v256 A_0 = new v256(*Akp);
                    accum0_0 = VectorUtils.MulAdd(A_0, B_0, accum0_0);
                    v256 A_1 = new v256(*StrideAddress(Akp, nstrideA, 1));
                    accum0_1 = VectorUtils.MulAdd(A_1, B_0, accum0_1);

                    Akp += 1;
                    Bkp = (float*)StrideAddress(Bkp, nstrideB, 1);
                }
                while (Akp < AkpEnd);

                {
                    float* Cnp_0 = Cnp;
                    *(v256*)(Cnp_0 + 0) = accum0_0;
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    *(v256*)(Cnp_1 + 0) = accum0_1;
                }

                n -= 8;
                Bnp += 8;
                Cnp += 8;
            }

            Ap = StrideAddress(Ap, nstrideA, 2);
            Cp = StrideAddress(Cp, nstrideC, 2);
            M -= 2;
        }
        if (M >= 1)
        {
            float *Bnp = Bp;
            float *Cnp = Cp;
            var n = (uint)N;

            while (n >= 16)
            {
                v256 accum0_0 = new v256();
                v256 accum1_0 = new v256();

                if (accumulateC)
                {
                    float* Cnp_0 = Cnp;
                    accum0_0 = *(v256*)(Cnp_0 + 0);
                    accum1_0 = *(v256*)(Cnp_0 + 8);
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                float *AkpEnd = Akp + (uint)K;

                do
                {
                    v256 B_0 = *(v256*)&Bkp[0];
                    v256 B_1 = *(v256*)&Bkp[8];

                    v256 A_0 = new v256(*Akp);
                    accum0_0 = VectorUtils.MulAdd(A_0, B_0, accum0_0);
                    accum1_0 = VectorUtils.MulAdd(A_0, B_1, accum1_0);

                    Akp += 1;
                    Bkp = (float*)StrideAddress(Bkp, nstrideB, 1);
                }
                while (Akp < AkpEnd);

                {
                    float* Cnp_0 = Cnp;
                    *(v256*)(Cnp_0 + 0) = accum0_0;
                    *(v256*)(Cnp_0 + 8) = accum1_0;
                }

                n -= 16;
                Bnp += 16;
                Cnp += 16;
            }
            if (n >= 8)
            {
                v256 accum0_0 = new v256();

                if (accumulateC)
                {
                    float* Cnp_0 = Cnp;
                    accum0_0 = *(v256*)(Cnp_0 + 0);
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                float *AkpEnd = Akp + (uint)K;

                do
                {
                    v256 B_0 = *(v256*)&Bkp[0];

                    v256 A_0 = new v256(*Akp);
                    accum0_0 = VectorUtils.MulAdd(A_0, B_0, accum0_0);

                    Akp += 1;
                    Bkp = (float*)StrideAddress(Bkp, nstrideB, 1);
                }
                while (Akp < AkpEnd);

                {
                    float* Cnp_0 = Cnp;
                    *(v256*)(Cnp_0 + 0) = accum0_0;
                }

                n -= 8;
                Bnp += 8;
                Cnp += 8;
            }

            Ap = StrideAddress(Ap, nstrideA, 1);
            Cp = StrideAddress(Cp, nstrideC, 1);
            M -= 1;
        }
    }
    static unsafe void MultiplyBlockUnroll3x24(
        [NoAlias] float* Ap, int strideA,
        [NoAlias] float* Bp, int strideB,
        [NoAlias] float* Cp, int strideC,
        int M, int N, int K, bool accumulateC)
    {
        // Avoid generating compiled code for the case where M/N/K are negative or zero.
        Hint.Assume(M > 0);
        Hint.Assume(N > 0);
        Hint.Assume(K > 0);

        // Help the compiler produce better code by using the strides as pointer-width values.
        // For x64, the inner loop is then able to use scaled indexing modes to access the rows
        // of matrix A, instead of wasting extra registers to store the offsets.
        var nstrideA = new System.IntPtr(strideA);
        var nstrideB = new System.IntPtr(strideB);
        var nstrideC = new System.IntPtr(strideC);

        while (M >= 3)
        {
            float *Bnp = Bp;
            float *Cnp = Cp;
            var n = (uint)N;

            while (n >= 24)
            {
                v256 accum0_0 = new v256();
                v256 accum1_0 = new v256();
                v256 accum2_0 = new v256();
                v256 accum0_1 = new v256();
                v256 accum1_1 = new v256();
                v256 accum2_1 = new v256();
                v256 accum0_2 = new v256();
                v256 accum1_2 = new v256();
                v256 accum2_2 = new v256();

                if (accumulateC)
                {
                    float* Cnp_0 = Cnp;
                    accum0_0 = *(v256*)(Cnp_0 + 0);
                    accum1_0 = *(v256*)(Cnp_0 + 8);
                    accum2_0 = *(v256*)(Cnp_0 + 16);
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    accum0_1 = *(v256*)(Cnp_1 + 0);
                    accum1_1 = *(v256*)(Cnp_1 + 8);
                    accum2_1 = *(v256*)(Cnp_1 + 16);
                    float* Cnp_2 = StrideAddress(Cnp, nstrideC, 2);
                    accum0_2 = *(v256*)(Cnp_2 + 0);
                    accum1_2 = *(v256*)(Cnp_2 + 8);
                    accum2_2 = *(v256*)(Cnp_2 + 16);
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                float *AkpEnd = Akp + (uint)K;

                do
                {
                    v256 B_0 = *(v256*)&Bkp[0];
                    v256 B_1 = *(v256*)&Bkp[8];
                    v256 B_2 = *(v256*)&Bkp[16];

                    v256 A_0 = new v256(*Akp);
                    accum0_0 = VectorUtils.MulAdd(A_0, B_0, accum0_0);
                    accum1_0 = VectorUtils.MulAdd(A_0, B_1, accum1_0);
                    accum2_0 = VectorUtils.MulAdd(A_0, B_2, accum2_0);
                    v256 A_1 = new v256(*StrideAddress(Akp, nstrideA, 1));
                    accum0_1 = VectorUtils.MulAdd(A_1, B_0, accum0_1);
                    accum1_1 = VectorUtils.MulAdd(A_1, B_1, accum1_1);
                    accum2_1 = VectorUtils.MulAdd(A_1, B_2, accum2_1);
                    v256 A_2 = new v256(*StrideAddress(Akp, nstrideA, 2));
                    accum0_2 = VectorUtils.MulAdd(A_2, B_0, accum0_2);
                    accum1_2 = VectorUtils.MulAdd(A_2, B_1, accum1_2);
                    accum2_2 = VectorUtils.MulAdd(A_2, B_2, accum2_2);

                    Akp += 1;
                    Bkp = (float*)StrideAddress(Bkp, nstrideB, 1);
                }
                while (Akp < AkpEnd);

                {
                    float* Cnp_0 = Cnp;
                    *(v256*)(Cnp_0 + 0) = accum0_0;
                    *(v256*)(Cnp_0 + 8) = accum1_0;
                    *(v256*)(Cnp_0 + 16) = accum2_0;
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    *(v256*)(Cnp_1 + 0) = accum0_1;
                    *(v256*)(Cnp_1 + 8) = accum1_1;
                    *(v256*)(Cnp_1 + 16) = accum2_1;
                    float* Cnp_2 = StrideAddress(Cnp, nstrideC, 2);
                    *(v256*)(Cnp_2 + 0) = accum0_2;
                    *(v256*)(Cnp_2 + 8) = accum1_2;
                    *(v256*)(Cnp_2 + 16) = accum2_2;
                }

                n -= 24;
                Bnp += 24;
                Cnp += 24;
            }
            if (n >= 16)
            {
                v256 accum0_0 = new v256();
                v256 accum1_0 = new v256();
                v256 accum0_1 = new v256();
                v256 accum1_1 = new v256();
                v256 accum0_2 = new v256();
                v256 accum1_2 = new v256();

                if (accumulateC)
                {
                    float* Cnp_0 = Cnp;
                    accum0_0 = *(v256*)(Cnp_0 + 0);
                    accum1_0 = *(v256*)(Cnp_0 + 8);
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    accum0_1 = *(v256*)(Cnp_1 + 0);
                    accum1_1 = *(v256*)(Cnp_1 + 8);
                    float* Cnp_2 = StrideAddress(Cnp, nstrideC, 2);
                    accum0_2 = *(v256*)(Cnp_2 + 0);
                    accum1_2 = *(v256*)(Cnp_2 + 8);
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                float *AkpEnd = Akp + (uint)K;

                do
                {
                    v256 B_0 = *(v256*)&Bkp[0];
                    v256 B_1 = *(v256*)&Bkp[8];

                    v256 A_0 = new v256(*Akp);
                    accum0_0 = VectorUtils.MulAdd(A_0, B_0, accum0_0);
                    accum1_0 = VectorUtils.MulAdd(A_0, B_1, accum1_0);
                    v256 A_1 = new v256(*StrideAddress(Akp, nstrideA, 1));
                    accum0_1 = VectorUtils.MulAdd(A_1, B_0, accum0_1);
                    accum1_1 = VectorUtils.MulAdd(A_1, B_1, accum1_1);
                    v256 A_2 = new v256(*StrideAddress(Akp, nstrideA, 2));
                    accum0_2 = VectorUtils.MulAdd(A_2, B_0, accum0_2);
                    accum1_2 = VectorUtils.MulAdd(A_2, B_1, accum1_2);

                    Akp += 1;
                    Bkp = (float*)StrideAddress(Bkp, nstrideB, 1);
                }
                while (Akp < AkpEnd);

                {
                    float* Cnp_0 = Cnp;
                    *(v256*)(Cnp_0 + 0) = accum0_0;
                    *(v256*)(Cnp_0 + 8) = accum1_0;
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    *(v256*)(Cnp_1 + 0) = accum0_1;
                    *(v256*)(Cnp_1 + 8) = accum1_1;
                    float* Cnp_2 = StrideAddress(Cnp, nstrideC, 2);
                    *(v256*)(Cnp_2 + 0) = accum0_2;
                    *(v256*)(Cnp_2 + 8) = accum1_2;
                }

                n -= 16;
                Bnp += 16;
                Cnp += 16;
            }
            if (n >= 8)
            {
                v256 accum0_0 = new v256();
                v256 accum0_1 = new v256();
                v256 accum0_2 = new v256();

                if (accumulateC)
                {
                    float* Cnp_0 = Cnp;
                    accum0_0 = *(v256*)(Cnp_0 + 0);
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    accum0_1 = *(v256*)(Cnp_1 + 0);
                    float* Cnp_2 = StrideAddress(Cnp, nstrideC, 2);
                    accum0_2 = *(v256*)(Cnp_2 + 0);
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                float *AkpEnd = Akp + (uint)K;

                do
                {
                    v256 B_0 = *(v256*)&Bkp[0];

                    v256 A_0 = new v256(*Akp);
                    accum0_0 = VectorUtils.MulAdd(A_0, B_0, accum0_0);
                    v256 A_1 = new v256(*StrideAddress(Akp, nstrideA, 1));
                    accum0_1 = VectorUtils.MulAdd(A_1, B_0, accum0_1);
                    v256 A_2 = new v256(*StrideAddress(Akp, nstrideA, 2));
                    accum0_2 = VectorUtils.MulAdd(A_2, B_0, accum0_2);

                    Akp += 1;
                    Bkp = (float*)StrideAddress(Bkp, nstrideB, 1);
                }
                while (Akp < AkpEnd);

                {
                    float* Cnp_0 = Cnp;
                    *(v256*)(Cnp_0 + 0) = accum0_0;
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    *(v256*)(Cnp_1 + 0) = accum0_1;
                    float* Cnp_2 = StrideAddress(Cnp, nstrideC, 2);
                    *(v256*)(Cnp_2 + 0) = accum0_2;
                }

                n -= 8;
                Bnp += 8;
                Cnp += 8;
            }

            Ap = StrideAddress(Ap, nstrideA, 3);
            Cp = StrideAddress(Cp, nstrideC, 3);
            M -= 3;
        }
        if (M >= 2)
        {
            float *Bnp = Bp;
            float *Cnp = Cp;
            var n = (uint)N;

            while (n >= 24)
            {
                v256 accum0_0 = new v256();
                v256 accum1_0 = new v256();
                v256 accum2_0 = new v256();
                v256 accum0_1 = new v256();
                v256 accum1_1 = new v256();
                v256 accum2_1 = new v256();

                if (accumulateC)
                {
                    float* Cnp_0 = Cnp;
                    accum0_0 = *(v256*)(Cnp_0 + 0);
                    accum1_0 = *(v256*)(Cnp_0 + 8);
                    accum2_0 = *(v256*)(Cnp_0 + 16);
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    accum0_1 = *(v256*)(Cnp_1 + 0);
                    accum1_1 = *(v256*)(Cnp_1 + 8);
                    accum2_1 = *(v256*)(Cnp_1 + 16);
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                float *AkpEnd = Akp + (uint)K;

                do
                {
                    v256 B_0 = *(v256*)&Bkp[0];
                    v256 B_1 = *(v256*)&Bkp[8];
                    v256 B_2 = *(v256*)&Bkp[16];

                    v256 A_0 = new v256(*Akp);
                    accum0_0 = VectorUtils.MulAdd(A_0, B_0, accum0_0);
                    accum1_0 = VectorUtils.MulAdd(A_0, B_1, accum1_0);
                    accum2_0 = VectorUtils.MulAdd(A_0, B_2, accum2_0);
                    v256 A_1 = new v256(*StrideAddress(Akp, nstrideA, 1));
                    accum0_1 = VectorUtils.MulAdd(A_1, B_0, accum0_1);
                    accum1_1 = VectorUtils.MulAdd(A_1, B_1, accum1_1);
                    accum2_1 = VectorUtils.MulAdd(A_1, B_2, accum2_1);

                    Akp += 1;
                    Bkp = (float*)StrideAddress(Bkp, nstrideB, 1);
                }
                while (Akp < AkpEnd);

                {
                    float* Cnp_0 = Cnp;
                    *(v256*)(Cnp_0 + 0) = accum0_0;
                    *(v256*)(Cnp_0 + 8) = accum1_0;
                    *(v256*)(Cnp_0 + 16) = accum2_0;
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    *(v256*)(Cnp_1 + 0) = accum0_1;
                    *(v256*)(Cnp_1 + 8) = accum1_1;
                    *(v256*)(Cnp_1 + 16) = accum2_1;
                }

                n -= 24;
                Bnp += 24;
                Cnp += 24;
            }
            if (n >= 16)
            {
                v256 accum0_0 = new v256();
                v256 accum1_0 = new v256();
                v256 accum0_1 = new v256();
                v256 accum1_1 = new v256();

                if (accumulateC)
                {
                    float* Cnp_0 = Cnp;
                    accum0_0 = *(v256*)(Cnp_0 + 0);
                    accum1_0 = *(v256*)(Cnp_0 + 8);
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    accum0_1 = *(v256*)(Cnp_1 + 0);
                    accum1_1 = *(v256*)(Cnp_1 + 8);
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                float *AkpEnd = Akp + (uint)K;

                do
                {
                    v256 B_0 = *(v256*)&Bkp[0];
                    v256 B_1 = *(v256*)&Bkp[8];

                    v256 A_0 = new v256(*Akp);
                    accum0_0 = VectorUtils.MulAdd(A_0, B_0, accum0_0);
                    accum1_0 = VectorUtils.MulAdd(A_0, B_1, accum1_0);
                    v256 A_1 = new v256(*StrideAddress(Akp, nstrideA, 1));
                    accum0_1 = VectorUtils.MulAdd(A_1, B_0, accum0_1);
                    accum1_1 = VectorUtils.MulAdd(A_1, B_1, accum1_1);

                    Akp += 1;
                    Bkp = (float*)StrideAddress(Bkp, nstrideB, 1);
                }
                while (Akp < AkpEnd);

                {
                    float* Cnp_0 = Cnp;
                    *(v256*)(Cnp_0 + 0) = accum0_0;
                    *(v256*)(Cnp_0 + 8) = accum1_0;
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    *(v256*)(Cnp_1 + 0) = accum0_1;
                    *(v256*)(Cnp_1 + 8) = accum1_1;
                }

                n -= 16;
                Bnp += 16;
                Cnp += 16;
            }
            if (n >= 8)
            {
                v256 accum0_0 = new v256();
                v256 accum0_1 = new v256();

                if (accumulateC)
                {
                    float* Cnp_0 = Cnp;
                    accum0_0 = *(v256*)(Cnp_0 + 0);
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    accum0_1 = *(v256*)(Cnp_1 + 0);
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                float *AkpEnd = Akp + (uint)K;

                do
                {
                    v256 B_0 = *(v256*)&Bkp[0];

                    v256 A_0 = new v256(*Akp);
                    accum0_0 = VectorUtils.MulAdd(A_0, B_0, accum0_0);
                    v256 A_1 = new v256(*StrideAddress(Akp, nstrideA, 1));
                    accum0_1 = VectorUtils.MulAdd(A_1, B_0, accum0_1);

                    Akp += 1;
                    Bkp = (float*)StrideAddress(Bkp, nstrideB, 1);
                }
                while (Akp < AkpEnd);

                {
                    float* Cnp_0 = Cnp;
                    *(v256*)(Cnp_0 + 0) = accum0_0;
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    *(v256*)(Cnp_1 + 0) = accum0_1;
                }

                n -= 8;
                Bnp += 8;
                Cnp += 8;
            }

            Ap = StrideAddress(Ap, nstrideA, 2);
            Cp = StrideAddress(Cp, nstrideC, 2);
            M -= 2;
        }
        if (M >= 1)
        {
            float *Bnp = Bp;
            float *Cnp = Cp;
            var n = (uint)N;

            while (n >= 24)
            {
                v256 accum0_0 = new v256();
                v256 accum1_0 = new v256();
                v256 accum2_0 = new v256();

                if (accumulateC)
                {
                    float* Cnp_0 = Cnp;
                    accum0_0 = *(v256*)(Cnp_0 + 0);
                    accum1_0 = *(v256*)(Cnp_0 + 8);
                    accum2_0 = *(v256*)(Cnp_0 + 16);
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                float *AkpEnd = Akp + (uint)K;

                do
                {
                    v256 B_0 = *(v256*)&Bkp[0];
                    v256 B_1 = *(v256*)&Bkp[8];
                    v256 B_2 = *(v256*)&Bkp[16];

                    v256 A_0 = new v256(*Akp);
                    accum0_0 = VectorUtils.MulAdd(A_0, B_0, accum0_0);
                    accum1_0 = VectorUtils.MulAdd(A_0, B_1, accum1_0);
                    accum2_0 = VectorUtils.MulAdd(A_0, B_2, accum2_0);

                    Akp += 1;
                    Bkp = (float*)StrideAddress(Bkp, nstrideB, 1);
                }
                while (Akp < AkpEnd);

                {
                    float* Cnp_0 = Cnp;
                    *(v256*)(Cnp_0 + 0) = accum0_0;
                    *(v256*)(Cnp_0 + 8) = accum1_0;
                    *(v256*)(Cnp_0 + 16) = accum2_0;
                }

                n -= 24;
                Bnp += 24;
                Cnp += 24;
            }
            if (n >= 16)
            {
                v256 accum0_0 = new v256();
                v256 accum1_0 = new v256();

                if (accumulateC)
                {
                    float* Cnp_0 = Cnp;
                    accum0_0 = *(v256*)(Cnp_0 + 0);
                    accum1_0 = *(v256*)(Cnp_0 + 8);
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                float *AkpEnd = Akp + (uint)K;

                do
                {
                    v256 B_0 = *(v256*)&Bkp[0];
                    v256 B_1 = *(v256*)&Bkp[8];

                    v256 A_0 = new v256(*Akp);
                    accum0_0 = VectorUtils.MulAdd(A_0, B_0, accum0_0);
                    accum1_0 = VectorUtils.MulAdd(A_0, B_1, accum1_0);

                    Akp += 1;
                    Bkp = (float*)StrideAddress(Bkp, nstrideB, 1);
                }
                while (Akp < AkpEnd);

                {
                    float* Cnp_0 = Cnp;
                    *(v256*)(Cnp_0 + 0) = accum0_0;
                    *(v256*)(Cnp_0 + 8) = accum1_0;
                }

                n -= 16;
                Bnp += 16;
                Cnp += 16;
            }
            if (n >= 8)
            {
                v256 accum0_0 = new v256();

                if (accumulateC)
                {
                    float* Cnp_0 = Cnp;
                    accum0_0 = *(v256*)(Cnp_0 + 0);
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                float *AkpEnd = Akp + (uint)K;

                do
                {
                    v256 B_0 = *(v256*)&Bkp[0];

                    v256 A_0 = new v256(*Akp);
                    accum0_0 = VectorUtils.MulAdd(A_0, B_0, accum0_0);

                    Akp += 1;
                    Bkp = (float*)StrideAddress(Bkp, nstrideB, 1);
                }
                while (Akp < AkpEnd);

                {
                    float* Cnp_0 = Cnp;
                    *(v256*)(Cnp_0 + 0) = accum0_0;
                }

                n -= 8;
                Bnp += 8;
                Cnp += 8;
            }

            Ap = StrideAddress(Ap, nstrideA, 1);
            Cp = StrideAddress(Cp, nstrideC, 1);
            M -= 1;
        }
    }
    static unsafe void MultiplyBlockUnroll4x24(
        [NoAlias] float* Ap, int strideA,
        [NoAlias] float* Bp, int strideB,
        [NoAlias] float* Cp, int strideC,
        int M, int N, int K, bool accumulateC)
    {
        // Avoid generating compiled code for the case where M/N/K are negative or zero.
        Hint.Assume(M > 0);
        Hint.Assume(N > 0);
        Hint.Assume(K > 0);

        // Help the compiler produce better code by using the strides as pointer-width values.
        // For x64, the inner loop is then able to use scaled indexing modes to access the rows
        // of matrix A, instead of wasting extra registers to store the offsets.
        var nstrideA = new System.IntPtr(strideA);
        var nstrideB = new System.IntPtr(strideB);
        var nstrideC = new System.IntPtr(strideC);

        while (M >= 4)
        {
            float *Bnp = Bp;
            float *Cnp = Cp;
            var n = (uint)N;

            while (n >= 24)
            {
                v256 accum0_0 = new v256();
                v256 accum1_0 = new v256();
                v256 accum2_0 = new v256();
                v256 accum0_1 = new v256();
                v256 accum1_1 = new v256();
                v256 accum2_1 = new v256();
                v256 accum0_2 = new v256();
                v256 accum1_2 = new v256();
                v256 accum2_2 = new v256();
                v256 accum0_3 = new v256();
                v256 accum1_3 = new v256();
                v256 accum2_3 = new v256();

                if (accumulateC)
                {
                    float* Cnp_0 = Cnp;
                    accum0_0 = *(v256*)(Cnp_0 + 0);
                    accum1_0 = *(v256*)(Cnp_0 + 8);
                    accum2_0 = *(v256*)(Cnp_0 + 16);
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    accum0_1 = *(v256*)(Cnp_1 + 0);
                    accum1_1 = *(v256*)(Cnp_1 + 8);
                    accum2_1 = *(v256*)(Cnp_1 + 16);
                    float* Cnp_2 = StrideAddress(Cnp, nstrideC, 2);
                    accum0_2 = *(v256*)(Cnp_2 + 0);
                    accum1_2 = *(v256*)(Cnp_2 + 8);
                    accum2_2 = *(v256*)(Cnp_2 + 16);
                    float* Cnp_3 = StrideAddress(Cnp, nstrideC, 3);
                    accum0_3 = *(v256*)(Cnp_3 + 0);
                    accum1_3 = *(v256*)(Cnp_3 + 8);
                    accum2_3 = *(v256*)(Cnp_3 + 16);
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                float *AkpEnd = Akp + (uint)K;

                do
                {
                    v256 B_0 = *(v256*)&Bkp[0];
                    v256 B_1 = *(v256*)&Bkp[8];
                    v256 B_2 = *(v256*)&Bkp[16];

                    v256 A_0 = new v256(*Akp);
                    accum0_0 = VectorUtils.MulAdd(A_0, B_0, accum0_0);
                    accum1_0 = VectorUtils.MulAdd(A_0, B_1, accum1_0);
                    accum2_0 = VectorUtils.MulAdd(A_0, B_2, accum2_0);
                    v256 A_1 = new v256(*StrideAddress(Akp, nstrideA, 1));
                    accum0_1 = VectorUtils.MulAdd(A_1, B_0, accum0_1);
                    accum1_1 = VectorUtils.MulAdd(A_1, B_1, accum1_1);
                    accum2_1 = VectorUtils.MulAdd(A_1, B_2, accum2_1);
                    v256 A_2 = new v256(*StrideAddress(Akp, nstrideA, 2));
                    accum0_2 = VectorUtils.MulAdd(A_2, B_0, accum0_2);
                    accum1_2 = VectorUtils.MulAdd(A_2, B_1, accum1_2);
                    accum2_2 = VectorUtils.MulAdd(A_2, B_2, accum2_2);
                    v256 A_3 = new v256(*StrideAddress(Akp, nstrideA, 3));
                    accum0_3 = VectorUtils.MulAdd(A_3, B_0, accum0_3);
                    accum1_3 = VectorUtils.MulAdd(A_3, B_1, accum1_3);
                    accum2_3 = VectorUtils.MulAdd(A_3, B_2, accum2_3);

                    Akp += 1;
                    Bkp = (float*)StrideAddress(Bkp, nstrideB, 1);
                }
                while (Akp < AkpEnd);

                {
                    float* Cnp_0 = Cnp;
                    *(v256*)(Cnp_0 + 0) = accum0_0;
                    *(v256*)(Cnp_0 + 8) = accum1_0;
                    *(v256*)(Cnp_0 + 16) = accum2_0;
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    *(v256*)(Cnp_1 + 0) = accum0_1;
                    *(v256*)(Cnp_1 + 8) = accum1_1;
                    *(v256*)(Cnp_1 + 16) = accum2_1;
                    float* Cnp_2 = StrideAddress(Cnp, nstrideC, 2);
                    *(v256*)(Cnp_2 + 0) = accum0_2;
                    *(v256*)(Cnp_2 + 8) = accum1_2;
                    *(v256*)(Cnp_2 + 16) = accum2_2;
                    float* Cnp_3 = StrideAddress(Cnp, nstrideC, 3);
                    *(v256*)(Cnp_3 + 0) = accum0_3;
                    *(v256*)(Cnp_3 + 8) = accum1_3;
                    *(v256*)(Cnp_3 + 16) = accum2_3;
                }

                n -= 24;
                Bnp += 24;
                Cnp += 24;
            }
            if (n >= 16)
            {
                v256 accum0_0 = new v256();
                v256 accum1_0 = new v256();
                v256 accum0_1 = new v256();
                v256 accum1_1 = new v256();
                v256 accum0_2 = new v256();
                v256 accum1_2 = new v256();
                v256 accum0_3 = new v256();
                v256 accum1_3 = new v256();

                if (accumulateC)
                {
                    float* Cnp_0 = Cnp;
                    accum0_0 = *(v256*)(Cnp_0 + 0);
                    accum1_0 = *(v256*)(Cnp_0 + 8);
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    accum0_1 = *(v256*)(Cnp_1 + 0);
                    accum1_1 = *(v256*)(Cnp_1 + 8);
                    float* Cnp_2 = StrideAddress(Cnp, nstrideC, 2);
                    accum0_2 = *(v256*)(Cnp_2 + 0);
                    accum1_2 = *(v256*)(Cnp_2 + 8);
                    float* Cnp_3 = StrideAddress(Cnp, nstrideC, 3);
                    accum0_3 = *(v256*)(Cnp_3 + 0);
                    accum1_3 = *(v256*)(Cnp_3 + 8);
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                float *AkpEnd = Akp + (uint)K;

                do
                {
                    v256 B_0 = *(v256*)&Bkp[0];
                    v256 B_1 = *(v256*)&Bkp[8];

                    v256 A_0 = new v256(*Akp);
                    accum0_0 = VectorUtils.MulAdd(A_0, B_0, accum0_0);
                    accum1_0 = VectorUtils.MulAdd(A_0, B_1, accum1_0);
                    v256 A_1 = new v256(*StrideAddress(Akp, nstrideA, 1));
                    accum0_1 = VectorUtils.MulAdd(A_1, B_0, accum0_1);
                    accum1_1 = VectorUtils.MulAdd(A_1, B_1, accum1_1);
                    v256 A_2 = new v256(*StrideAddress(Akp, nstrideA, 2));
                    accum0_2 = VectorUtils.MulAdd(A_2, B_0, accum0_2);
                    accum1_2 = VectorUtils.MulAdd(A_2, B_1, accum1_2);
                    v256 A_3 = new v256(*StrideAddress(Akp, nstrideA, 3));
                    accum0_3 = VectorUtils.MulAdd(A_3, B_0, accum0_3);
                    accum1_3 = VectorUtils.MulAdd(A_3, B_1, accum1_3);

                    Akp += 1;
                    Bkp = (float*)StrideAddress(Bkp, nstrideB, 1);
                }
                while (Akp < AkpEnd);

                {
                    float* Cnp_0 = Cnp;
                    *(v256*)(Cnp_0 + 0) = accum0_0;
                    *(v256*)(Cnp_0 + 8) = accum1_0;
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    *(v256*)(Cnp_1 + 0) = accum0_1;
                    *(v256*)(Cnp_1 + 8) = accum1_1;
                    float* Cnp_2 = StrideAddress(Cnp, nstrideC, 2);
                    *(v256*)(Cnp_2 + 0) = accum0_2;
                    *(v256*)(Cnp_2 + 8) = accum1_2;
                    float* Cnp_3 = StrideAddress(Cnp, nstrideC, 3);
                    *(v256*)(Cnp_3 + 0) = accum0_3;
                    *(v256*)(Cnp_3 + 8) = accum1_3;
                }

                n -= 16;
                Bnp += 16;
                Cnp += 16;
            }
            if (n >= 8)
            {
                v256 accum0_0 = new v256();
                v256 accum0_1 = new v256();
                v256 accum0_2 = new v256();
                v256 accum0_3 = new v256();

                if (accumulateC)
                {
                    float* Cnp_0 = Cnp;
                    accum0_0 = *(v256*)(Cnp_0 + 0);
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    accum0_1 = *(v256*)(Cnp_1 + 0);
                    float* Cnp_2 = StrideAddress(Cnp, nstrideC, 2);
                    accum0_2 = *(v256*)(Cnp_2 + 0);
                    float* Cnp_3 = StrideAddress(Cnp, nstrideC, 3);
                    accum0_3 = *(v256*)(Cnp_3 + 0);
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                float *AkpEnd = Akp + (uint)K;

                do
                {
                    v256 B_0 = *(v256*)&Bkp[0];

                    v256 A_0 = new v256(*Akp);
                    accum0_0 = VectorUtils.MulAdd(A_0, B_0, accum0_0);
                    v256 A_1 = new v256(*StrideAddress(Akp, nstrideA, 1));
                    accum0_1 = VectorUtils.MulAdd(A_1, B_0, accum0_1);
                    v256 A_2 = new v256(*StrideAddress(Akp, nstrideA, 2));
                    accum0_2 = VectorUtils.MulAdd(A_2, B_0, accum0_2);
                    v256 A_3 = new v256(*StrideAddress(Akp, nstrideA, 3));
                    accum0_3 = VectorUtils.MulAdd(A_3, B_0, accum0_3);

                    Akp += 1;
                    Bkp = (float*)StrideAddress(Bkp, nstrideB, 1);
                }
                while (Akp < AkpEnd);

                {
                    float* Cnp_0 = Cnp;
                    *(v256*)(Cnp_0 + 0) = accum0_0;
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    *(v256*)(Cnp_1 + 0) = accum0_1;
                    float* Cnp_2 = StrideAddress(Cnp, nstrideC, 2);
                    *(v256*)(Cnp_2 + 0) = accum0_2;
                    float* Cnp_3 = StrideAddress(Cnp, nstrideC, 3);
                    *(v256*)(Cnp_3 + 0) = accum0_3;
                }

                n -= 8;
                Bnp += 8;
                Cnp += 8;
            }

            Ap = StrideAddress(Ap, nstrideA, 4);
            Cp = StrideAddress(Cp, nstrideC, 4);
            M -= 4;
        }
        if (M >= 3)
        {
            float *Bnp = Bp;
            float *Cnp = Cp;
            var n = (uint)N;

            while (n >= 24)
            {
                v256 accum0_0 = new v256();
                v256 accum1_0 = new v256();
                v256 accum2_0 = new v256();
                v256 accum0_1 = new v256();
                v256 accum1_1 = new v256();
                v256 accum2_1 = new v256();
                v256 accum0_2 = new v256();
                v256 accum1_2 = new v256();
                v256 accum2_2 = new v256();

                if (accumulateC)
                {
                    float* Cnp_0 = Cnp;
                    accum0_0 = *(v256*)(Cnp_0 + 0);
                    accum1_0 = *(v256*)(Cnp_0 + 8);
                    accum2_0 = *(v256*)(Cnp_0 + 16);
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    accum0_1 = *(v256*)(Cnp_1 + 0);
                    accum1_1 = *(v256*)(Cnp_1 + 8);
                    accum2_1 = *(v256*)(Cnp_1 + 16);
                    float* Cnp_2 = StrideAddress(Cnp, nstrideC, 2);
                    accum0_2 = *(v256*)(Cnp_2 + 0);
                    accum1_2 = *(v256*)(Cnp_2 + 8);
                    accum2_2 = *(v256*)(Cnp_2 + 16);
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                float *AkpEnd = Akp + (uint)K;

                do
                {
                    v256 B_0 = *(v256*)&Bkp[0];
                    v256 B_1 = *(v256*)&Bkp[8];
                    v256 B_2 = *(v256*)&Bkp[16];

                    v256 A_0 = new v256(*Akp);
                    accum0_0 = VectorUtils.MulAdd(A_0, B_0, accum0_0);
                    accum1_0 = VectorUtils.MulAdd(A_0, B_1, accum1_0);
                    accum2_0 = VectorUtils.MulAdd(A_0, B_2, accum2_0);
                    v256 A_1 = new v256(*StrideAddress(Akp, nstrideA, 1));
                    accum0_1 = VectorUtils.MulAdd(A_1, B_0, accum0_1);
                    accum1_1 = VectorUtils.MulAdd(A_1, B_1, accum1_1);
                    accum2_1 = VectorUtils.MulAdd(A_1, B_2, accum2_1);
                    v256 A_2 = new v256(*StrideAddress(Akp, nstrideA, 2));
                    accum0_2 = VectorUtils.MulAdd(A_2, B_0, accum0_2);
                    accum1_2 = VectorUtils.MulAdd(A_2, B_1, accum1_2);
                    accum2_2 = VectorUtils.MulAdd(A_2, B_2, accum2_2);

                    Akp += 1;
                    Bkp = (float*)StrideAddress(Bkp, nstrideB, 1);
                }
                while (Akp < AkpEnd);

                {
                    float* Cnp_0 = Cnp;
                    *(v256*)(Cnp_0 + 0) = accum0_0;
                    *(v256*)(Cnp_0 + 8) = accum1_0;
                    *(v256*)(Cnp_0 + 16) = accum2_0;
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    *(v256*)(Cnp_1 + 0) = accum0_1;
                    *(v256*)(Cnp_1 + 8) = accum1_1;
                    *(v256*)(Cnp_1 + 16) = accum2_1;
                    float* Cnp_2 = StrideAddress(Cnp, nstrideC, 2);
                    *(v256*)(Cnp_2 + 0) = accum0_2;
                    *(v256*)(Cnp_2 + 8) = accum1_2;
                    *(v256*)(Cnp_2 + 16) = accum2_2;
                }

                n -= 24;
                Bnp += 24;
                Cnp += 24;
            }
            if (n >= 16)
            {
                v256 accum0_0 = new v256();
                v256 accum1_0 = new v256();
                v256 accum0_1 = new v256();
                v256 accum1_1 = new v256();
                v256 accum0_2 = new v256();
                v256 accum1_2 = new v256();

                if (accumulateC)
                {
                    float* Cnp_0 = Cnp;
                    accum0_0 = *(v256*)(Cnp_0 + 0);
                    accum1_0 = *(v256*)(Cnp_0 + 8);
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    accum0_1 = *(v256*)(Cnp_1 + 0);
                    accum1_1 = *(v256*)(Cnp_1 + 8);
                    float* Cnp_2 = StrideAddress(Cnp, nstrideC, 2);
                    accum0_2 = *(v256*)(Cnp_2 + 0);
                    accum1_2 = *(v256*)(Cnp_2 + 8);
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                float *AkpEnd = Akp + (uint)K;

                do
                {
                    v256 B_0 = *(v256*)&Bkp[0];
                    v256 B_1 = *(v256*)&Bkp[8];

                    v256 A_0 = new v256(*Akp);
                    accum0_0 = VectorUtils.MulAdd(A_0, B_0, accum0_0);
                    accum1_0 = VectorUtils.MulAdd(A_0, B_1, accum1_0);
                    v256 A_1 = new v256(*StrideAddress(Akp, nstrideA, 1));
                    accum0_1 = VectorUtils.MulAdd(A_1, B_0, accum0_1);
                    accum1_1 = VectorUtils.MulAdd(A_1, B_1, accum1_1);
                    v256 A_2 = new v256(*StrideAddress(Akp, nstrideA, 2));
                    accum0_2 = VectorUtils.MulAdd(A_2, B_0, accum0_2);
                    accum1_2 = VectorUtils.MulAdd(A_2, B_1, accum1_2);

                    Akp += 1;
                    Bkp = (float*)StrideAddress(Bkp, nstrideB, 1);
                }
                while (Akp < AkpEnd);

                {
                    float* Cnp_0 = Cnp;
                    *(v256*)(Cnp_0 + 0) = accum0_0;
                    *(v256*)(Cnp_0 + 8) = accum1_0;
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    *(v256*)(Cnp_1 + 0) = accum0_1;
                    *(v256*)(Cnp_1 + 8) = accum1_1;
                    float* Cnp_2 = StrideAddress(Cnp, nstrideC, 2);
                    *(v256*)(Cnp_2 + 0) = accum0_2;
                    *(v256*)(Cnp_2 + 8) = accum1_2;
                }

                n -= 16;
                Bnp += 16;
                Cnp += 16;
            }
            if (n >= 8)
            {
                v256 accum0_0 = new v256();
                v256 accum0_1 = new v256();
                v256 accum0_2 = new v256();

                if (accumulateC)
                {
                    float* Cnp_0 = Cnp;
                    accum0_0 = *(v256*)(Cnp_0 + 0);
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    accum0_1 = *(v256*)(Cnp_1 + 0);
                    float* Cnp_2 = StrideAddress(Cnp, nstrideC, 2);
                    accum0_2 = *(v256*)(Cnp_2 + 0);
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                float *AkpEnd = Akp + (uint)K;

                do
                {
                    v256 B_0 = *(v256*)&Bkp[0];

                    v256 A_0 = new v256(*Akp);
                    accum0_0 = VectorUtils.MulAdd(A_0, B_0, accum0_0);
                    v256 A_1 = new v256(*StrideAddress(Akp, nstrideA, 1));
                    accum0_1 = VectorUtils.MulAdd(A_1, B_0, accum0_1);
                    v256 A_2 = new v256(*StrideAddress(Akp, nstrideA, 2));
                    accum0_2 = VectorUtils.MulAdd(A_2, B_0, accum0_2);

                    Akp += 1;
                    Bkp = (float*)StrideAddress(Bkp, nstrideB, 1);
                }
                while (Akp < AkpEnd);

                {
                    float* Cnp_0 = Cnp;
                    *(v256*)(Cnp_0 + 0) = accum0_0;
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    *(v256*)(Cnp_1 + 0) = accum0_1;
                    float* Cnp_2 = StrideAddress(Cnp, nstrideC, 2);
                    *(v256*)(Cnp_2 + 0) = accum0_2;
                }

                n -= 8;
                Bnp += 8;
                Cnp += 8;
            }

            Ap = StrideAddress(Ap, nstrideA, 3);
            Cp = StrideAddress(Cp, nstrideC, 3);
            M -= 3;
        }
        if (M >= 2)
        {
            float *Bnp = Bp;
            float *Cnp = Cp;
            var n = (uint)N;

            while (n >= 24)
            {
                v256 accum0_0 = new v256();
                v256 accum1_0 = new v256();
                v256 accum2_0 = new v256();
                v256 accum0_1 = new v256();
                v256 accum1_1 = new v256();
                v256 accum2_1 = new v256();

                if (accumulateC)
                {
                    float* Cnp_0 = Cnp;
                    accum0_0 = *(v256*)(Cnp_0 + 0);
                    accum1_0 = *(v256*)(Cnp_0 + 8);
                    accum2_0 = *(v256*)(Cnp_0 + 16);
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    accum0_1 = *(v256*)(Cnp_1 + 0);
                    accum1_1 = *(v256*)(Cnp_1 + 8);
                    accum2_1 = *(v256*)(Cnp_1 + 16);
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                float *AkpEnd = Akp + (uint)K;

                do
                {
                    v256 B_0 = *(v256*)&Bkp[0];
                    v256 B_1 = *(v256*)&Bkp[8];
                    v256 B_2 = *(v256*)&Bkp[16];

                    v256 A_0 = new v256(*Akp);
                    accum0_0 = VectorUtils.MulAdd(A_0, B_0, accum0_0);
                    accum1_0 = VectorUtils.MulAdd(A_0, B_1, accum1_0);
                    accum2_0 = VectorUtils.MulAdd(A_0, B_2, accum2_0);
                    v256 A_1 = new v256(*StrideAddress(Akp, nstrideA, 1));
                    accum0_1 = VectorUtils.MulAdd(A_1, B_0, accum0_1);
                    accum1_1 = VectorUtils.MulAdd(A_1, B_1, accum1_1);
                    accum2_1 = VectorUtils.MulAdd(A_1, B_2, accum2_1);

                    Akp += 1;
                    Bkp = (float*)StrideAddress(Bkp, nstrideB, 1);
                }
                while (Akp < AkpEnd);

                {
                    float* Cnp_0 = Cnp;
                    *(v256*)(Cnp_0 + 0) = accum0_0;
                    *(v256*)(Cnp_0 + 8) = accum1_0;
                    *(v256*)(Cnp_0 + 16) = accum2_0;
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    *(v256*)(Cnp_1 + 0) = accum0_1;
                    *(v256*)(Cnp_1 + 8) = accum1_1;
                    *(v256*)(Cnp_1 + 16) = accum2_1;
                }

                n -= 24;
                Bnp += 24;
                Cnp += 24;
            }
            if (n >= 16)
            {
                v256 accum0_0 = new v256();
                v256 accum1_0 = new v256();
                v256 accum0_1 = new v256();
                v256 accum1_1 = new v256();

                if (accumulateC)
                {
                    float* Cnp_0 = Cnp;
                    accum0_0 = *(v256*)(Cnp_0 + 0);
                    accum1_0 = *(v256*)(Cnp_0 + 8);
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    accum0_1 = *(v256*)(Cnp_1 + 0);
                    accum1_1 = *(v256*)(Cnp_1 + 8);
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                float *AkpEnd = Akp + (uint)K;

                do
                {
                    v256 B_0 = *(v256*)&Bkp[0];
                    v256 B_1 = *(v256*)&Bkp[8];

                    v256 A_0 = new v256(*Akp);
                    accum0_0 = VectorUtils.MulAdd(A_0, B_0, accum0_0);
                    accum1_0 = VectorUtils.MulAdd(A_0, B_1, accum1_0);
                    v256 A_1 = new v256(*StrideAddress(Akp, nstrideA, 1));
                    accum0_1 = VectorUtils.MulAdd(A_1, B_0, accum0_1);
                    accum1_1 = VectorUtils.MulAdd(A_1, B_1, accum1_1);

                    Akp += 1;
                    Bkp = (float*)StrideAddress(Bkp, nstrideB, 1);
                }
                while (Akp < AkpEnd);

                {
                    float* Cnp_0 = Cnp;
                    *(v256*)(Cnp_0 + 0) = accum0_0;
                    *(v256*)(Cnp_0 + 8) = accum1_0;
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    *(v256*)(Cnp_1 + 0) = accum0_1;
                    *(v256*)(Cnp_1 + 8) = accum1_1;
                }

                n -= 16;
                Bnp += 16;
                Cnp += 16;
            }
            if (n >= 8)
            {
                v256 accum0_0 = new v256();
                v256 accum0_1 = new v256();

                if (accumulateC)
                {
                    float* Cnp_0 = Cnp;
                    accum0_0 = *(v256*)(Cnp_0 + 0);
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    accum0_1 = *(v256*)(Cnp_1 + 0);
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                float *AkpEnd = Akp + (uint)K;

                do
                {
                    v256 B_0 = *(v256*)&Bkp[0];

                    v256 A_0 = new v256(*Akp);
                    accum0_0 = VectorUtils.MulAdd(A_0, B_0, accum0_0);
                    v256 A_1 = new v256(*StrideAddress(Akp, nstrideA, 1));
                    accum0_1 = VectorUtils.MulAdd(A_1, B_0, accum0_1);

                    Akp += 1;
                    Bkp = (float*)StrideAddress(Bkp, nstrideB, 1);
                }
                while (Akp < AkpEnd);

                {
                    float* Cnp_0 = Cnp;
                    *(v256*)(Cnp_0 + 0) = accum0_0;
                    float* Cnp_1 = StrideAddress(Cnp, nstrideC, 1);
                    *(v256*)(Cnp_1 + 0) = accum0_1;
                }

                n -= 8;
                Bnp += 8;
                Cnp += 8;
            }

            Ap = StrideAddress(Ap, nstrideA, 2);
            Cp = StrideAddress(Cp, nstrideC, 2);
            M -= 2;
        }
        if (M >= 1)
        {
            float *Bnp = Bp;
            float *Cnp = Cp;
            var n = (uint)N;

            while (n >= 24)
            {
                v256 accum0_0 = new v256();
                v256 accum1_0 = new v256();
                v256 accum2_0 = new v256();

                if (accumulateC)
                {
                    float* Cnp_0 = Cnp;
                    accum0_0 = *(v256*)(Cnp_0 + 0);
                    accum1_0 = *(v256*)(Cnp_0 + 8);
                    accum2_0 = *(v256*)(Cnp_0 + 16);
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                float *AkpEnd = Akp + (uint)K;

                do
                {
                    v256 B_0 = *(v256*)&Bkp[0];
                    v256 B_1 = *(v256*)&Bkp[8];
                    v256 B_2 = *(v256*)&Bkp[16];

                    v256 A_0 = new v256(*Akp);
                    accum0_0 = VectorUtils.MulAdd(A_0, B_0, accum0_0);
                    accum1_0 = VectorUtils.MulAdd(A_0, B_1, accum1_0);
                    accum2_0 = VectorUtils.MulAdd(A_0, B_2, accum2_0);

                    Akp += 1;
                    Bkp = (float*)StrideAddress(Bkp, nstrideB, 1);
                }
                while (Akp < AkpEnd);

                {
                    float* Cnp_0 = Cnp;
                    *(v256*)(Cnp_0 + 0) = accum0_0;
                    *(v256*)(Cnp_0 + 8) = accum1_0;
                    *(v256*)(Cnp_0 + 16) = accum2_0;
                }

                n -= 24;
                Bnp += 24;
                Cnp += 24;
            }
            if (n >= 16)
            {
                v256 accum0_0 = new v256();
                v256 accum1_0 = new v256();

                if (accumulateC)
                {
                    float* Cnp_0 = Cnp;
                    accum0_0 = *(v256*)(Cnp_0 + 0);
                    accum1_0 = *(v256*)(Cnp_0 + 8);
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                float *AkpEnd = Akp + (uint)K;

                do
                {
                    v256 B_0 = *(v256*)&Bkp[0];
                    v256 B_1 = *(v256*)&Bkp[8];

                    v256 A_0 = new v256(*Akp);
                    accum0_0 = VectorUtils.MulAdd(A_0, B_0, accum0_0);
                    accum1_0 = VectorUtils.MulAdd(A_0, B_1, accum1_0);

                    Akp += 1;
                    Bkp = (float*)StrideAddress(Bkp, nstrideB, 1);
                }
                while (Akp < AkpEnd);

                {
                    float* Cnp_0 = Cnp;
                    *(v256*)(Cnp_0 + 0) = accum0_0;
                    *(v256*)(Cnp_0 + 8) = accum1_0;
                }

                n -= 16;
                Bnp += 16;
                Cnp += 16;
            }
            if (n >= 8)
            {
                v256 accum0_0 = new v256();

                if (accumulateC)
                {
                    float* Cnp_0 = Cnp;
                    accum0_0 = *(v256*)(Cnp_0 + 0);
                }

                float *Akp = Ap;
                float *Bkp = Bnp;
                float *AkpEnd = Akp + (uint)K;

                do
                {
                    v256 B_0 = *(v256*)&Bkp[0];

                    v256 A_0 = new v256(*Akp);
                    accum0_0 = VectorUtils.MulAdd(A_0, B_0, accum0_0);

                    Akp += 1;
                    Bkp = (float*)StrideAddress(Bkp, nstrideB, 1);
                }
                while (Akp < AkpEnd);

                {
                    float* Cnp_0 = Cnp;
                    *(v256*)(Cnp_0 + 0) = accum0_0;
                }

                n -= 8;
                Bnp += 8;
                Cnp += 8;
            }

            Ap = StrideAddress(Ap, nstrideA, 1);
            Cp = StrideAddress(Cp, nstrideC, 1);
            M -= 1;
        }
    }
#endif

    const int multiplyBlockWidthN = 8;      // required strideB/strideC alignment for MultiplyBlockUnroll

    static unsafe void MultiplyBlockUnroll(
        [NoAlias] float* Ap, int strideA,
        [NoAlias] float* Bp, int strideB,
        [NoAlias] float* Cp, int strideC,
        int M, int N, int K, bool accumulateC)
    {
#if UNITY_WEBGL || WEBGL_MATMUL_OVERRIDE
        MultiplyBlockUnroll2x16Wasm(Ap, strideA, Bp, strideB, Cp, strideC, M, N, K, accumulateC);
#else
        if (System.IntPtr.Size == 4)
        {
            // Note: Current versions of Burst do not set IsNeonSupported for 32-bit targets,
            // so assume NEON as the default fallback.
            if (Unity.Burst.Intrinsics.X86.Sse.IsSseSupported)
                MultiplyBlockUnroll1x16(Ap, strideA, Bp, strideB, Cp, strideC, M, N, K, accumulateC);
            else
                MultiplyBlockUnroll2x16(Ap, strideA, Bp, strideB, Cp, strideC, M, N, K, accumulateC);
        }
        else
        {
            if (Unity.Burst.Intrinsics.X86.Avx2.IsAvx2Supported)
                MultiplyBlockUnroll4x24(Ap, strideA, Bp, strideB, Cp, strideC, M, N, K, accumulateC);
            else if (Unity.Burst.Intrinsics.X86.Avx.IsAvxSupported)
                MultiplyBlockUnroll3x24(Ap, strideA, Bp, strideB, Cp, strideC, M, N, K, accumulateC);
            else if (Unity.Burst.Intrinsics.X86.Sse.IsSseSupported)
                MultiplyBlockUnroll2x16(Ap, strideA, Bp, strideB, Cp, strideC, M, N, K, accumulateC);
            else
                MultiplyBlockUnroll4x16(Ap, strideA, Bp, strideB, Cp, strideC, M, N, K, accumulateC);
        }
#endif
    }
}
}
