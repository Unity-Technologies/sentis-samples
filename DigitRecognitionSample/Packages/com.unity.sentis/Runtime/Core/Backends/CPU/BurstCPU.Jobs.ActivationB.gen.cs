// This is auto-generated -- do not modify directly

using UnityEngine;
using System;
using Unity.Burst;
using Unity.Burst.Intrinsics;
using static Unity.Burst.Intrinsics.X86.Avx;
using static Unity.Burst.Intrinsics.X86.Fma;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;
using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace Unity.Sentis
{
public partial class CPUBackend
{


[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct HardSigmoidJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public float alpha;
    public float beta;
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    public void Execute(int threadIdx)
    {
        float v = Xptr[threadIdx];
        Optr[threadIdx] = max(0.0f, min(1.0f, alpha * v + beta));
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct SquareJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public float alpha;
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    public void Execute(int threadIdx)
    {
        float v = Xptr[threadIdx];
        Optr[threadIdx] = v * v;
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct GeluJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public float alpha;
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    public void Execute(int threadIdx)
    {
        float vx = Xptr[threadIdx];

        float v = vx / sqrt(2);

        // Abramowitz/Stegun approximations
        // erf(x) = -erf(-x)
        float x = abs(v);

        float p = 0.3275911f;
        float a1 = 0.254829592f; float a2 = -0.284496736f; float a3 = 1.421413741f;
        float a4 = -1.453152027f; float a5 = 1.061405429f;

        float t = 1.0f / (1.0f + p * x);
        float t2 = t * t;
        float t3 = t2 * t;
        float t4 = t3 * t;
        float t5 = t4 * t;

        float erf = sign(v) * (1 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * exp(-x * x));

        Optr[threadIdx] = (erf + 1) * vx * 0.5f;

    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct ErfJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public float alpha;
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    public void Execute(int threadIdx)
    {
        float v = Xptr[threadIdx];

        // Abramowitz/Stegun approximations
        // erf(x) = -erf(-x)
        float x = abs(v);

        float p = 0.3275911f;
        float a1 = 0.254829592f; float a2 = -0.284496736f; float a3 = 1.421413741f;
        float a4 = -1.453152027f; float a5 = 1.061405429f;

        float t = 1.0f / (1.0f + p * x);
        float t2 = t * t;
        float t3 = t2 * t;
        float t4 = t3 * t;
        float t5 = t4 * t;

        Optr[threadIdx] = sign(v) * (1 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * exp(-x * x));
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct CeluJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public float alpha;
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    public void Execute(int threadIdx)
    {
        // f(x) = max(0,x) + min(0,alpha*(exp(x/alpha)-1))
        // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Celu
        float v = Xptr[threadIdx];
        Optr[threadIdx] = max(0.0f, v) + min(0.0f, alpha * (exp(v / alpha) - 1.0f));
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct ShrinkJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public float bias;
    public float lambd;
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    public void Execute(int threadIdx)
    {
        // If x < -lambd, y = x + bias; If x > lambd, y = x - bias; Otherwise, y = 0.
        // https://github.com/onnx/onnx/blob/main/docs/Operators.md#shrink
        float v = Xptr[threadIdx];
        float y = 0.0f;
        if (v < -lambd)
            y = v + bias;
        else if (v > lambd)
            y = v - bias;
        Optr[threadIdx] = y;
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct ThresholdedReluJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public float alpha;
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    public void Execute(int threadIdx)
    {
        // y = x for x > alpha, y = 0 otherwise
        // https://github.com/onnx/onnx/blob/main/docs/Operators.md#thresholdedrelu
        float v = Xptr[threadIdx];
        float y = 0.0f;
        if (v > alpha)
            y = v;
        Optr[threadIdx] = y;
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct EluJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public float alpha;
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    public void Execute(int threadIdx)
    {
        // f(x) = alpha * (exp(x) - 1.) for x < 0, f(x) = x for x >= 0
        // "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)", DA Clevert, 2015
        // https://arxiv.org/abs/1511.07289
        float v = Xptr[threadIdx];
        if (v <= 0.0f)
            v = alpha * (exp(v) - 1.0f);
        Optr[threadIdx] = v;
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct SeluJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public float alpha;
    public float gamma;
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    public void Execute(int threadIdx)
    {
        // f(x) = gamma * (alpha * e^x - alpha) for x <= 0, f(x) = gamma * x for x > 0
        float v = Xptr[threadIdx];
        if (v <= 0.0f)
            v = gamma * (alpha * exp(v) - alpha);
        else
            v = gamma * v;
        Optr[threadIdx] = v;
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct SoftplusJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    float Apply(float v) { return log(1 + exp(-abs(v))) + max(v, 0); }
    public void Execute(int threadIdx)
    {
        float v = Xptr[threadIdx];
        Optr[threadIdx] = Apply(v);
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct CeilJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    float Apply(float v) { return ceil(v); }
    public void Execute(int threadIdx)
    {
        float v = Xptr[threadIdx];
        Optr[threadIdx] = Apply(v);
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct FloorJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    float Apply(float v) { return floor(v); }
    public void Execute(int threadIdx)
    {
        float v = Xptr[threadIdx];
        Optr[threadIdx] = Apply(v);
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct RoundJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    float Apply(float v) { return round(v); }
    public void Execute(int threadIdx)
    {
        float v = Xptr[threadIdx];
        Optr[threadIdx] = Apply(v);
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct ReciprocalJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    float Apply(float v) { return 1.0f / v; }
    public void Execute(int threadIdx)
    {
        float v = Xptr[threadIdx];
        Optr[threadIdx] = Apply(v);
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct ExpJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    float Apply(float v) { return exp(v); }
    public void Execute(int threadIdx)
    {
        float v = Xptr[threadIdx];
        Optr[threadIdx] = Apply(v);
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct LogJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    float Apply(float v) { return log(v); }
    public void Execute(int threadIdx)
    {
        float v = Xptr[threadIdx];
        Optr[threadIdx] = Apply(v);
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct SqrtJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    float Apply(float v) { return sqrt(v); }
    public void Execute(int threadIdx)
    {
        float v = Xptr[threadIdx];
        Optr[threadIdx] = Apply(v);
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct AcosJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    float Apply(float v) { return acos(v); }
    public void Execute(int threadIdx)
    {
        float v = Xptr[threadIdx];
        Optr[threadIdx] = Apply(v);
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct AcoshJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    float Apply(float v) { return log( v + sqrt(v*v - 1.0f)); }
    public void Execute(int threadIdx)
    {
        float v = Xptr[threadIdx];
        Optr[threadIdx] = Apply(v);
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct AsinJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    float Apply(float v) { return asin(v); }
    public void Execute(int threadIdx)
    {
        float v = Xptr[threadIdx];
        Optr[threadIdx] = Apply(v);
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct AsinhJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    float Apply(float v) { return log( v + sqrt(v*v + 1.0f)); }
    public void Execute(int threadIdx)
    {
        float v = Xptr[threadIdx];
        Optr[threadIdx] = Apply(v);
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct AtanJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    float Apply(float v) { return atan(v); }
    public void Execute(int threadIdx)
    {
        float v = Xptr[threadIdx];
        Optr[threadIdx] = Apply(v);
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct AtanhJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    float Apply(float v) { return 0.5f * log((1.0f + v)/(1.0f - v)); }
    public void Execute(int threadIdx)
    {
        float v = Xptr[threadIdx];
        Optr[threadIdx] = Apply(v);
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct CosJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    float Apply(float v) { return cos(v); }
    public void Execute(int threadIdx)
    {
        float v = Xptr[threadIdx];
        Optr[threadIdx] = Apply(v);
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct CoshJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    float Apply(float v) { return 0.5f * (exp(v) + exp(-v)); }
    public void Execute(int threadIdx)
    {
        float v = Xptr[threadIdx];
        Optr[threadIdx] = Apply(v);
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct SinJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    float Apply(float v) { return sin(v); }
    public void Execute(int threadIdx)
    {
        float v = Xptr[threadIdx];
        Optr[threadIdx] = Apply(v);
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct SinhJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    float Apply(float v) { return 0.5f * (exp(v) - exp(-v)); }
    public void Execute(int threadIdx)
    {
        float v = Xptr[threadIdx];
        Optr[threadIdx] = Apply(v);
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct TanJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    float Apply(float v) { return tan(v); }
    public void Execute(int threadIdx)
    {
        float v = Xptr[threadIdx];
        Optr[threadIdx] = Apply(v);
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct SoftsignJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    float Apply(float v) {
        return v / (1.0f + abs(v));
    }
    public void Execute(int threadIdx)
    {
        float v = Xptr[threadIdx];
        Optr[threadIdx] = Apply(v);
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct HardSwishJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    float Apply(float v) {
        return v * max(0, min(1, 0.16666667f * v + 0.5f));
    }
    public void Execute(int threadIdx)
    {
        float v = Xptr[threadIdx];
        Optr[threadIdx] = Apply(v);
    }
}

}
}
