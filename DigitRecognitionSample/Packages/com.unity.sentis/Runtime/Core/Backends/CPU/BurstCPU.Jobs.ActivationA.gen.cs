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
internal unsafe struct LeakyReluJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public float alpha;
    public float f1;
    public float f2;
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    public void Execute(int threadIdx)
    {
        float v = Xptr[threadIdx];
        // from Theano impl
        // https://github.com/Theano/theano/blob/d395439aec5a6ddde8ef5c266fd976412a5c5695/theano/tensor/nnet/nnet.py#L2209-L2251
        Optr[threadIdx] = f1 * v + f2 * abs(v);
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct SwishJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public float alpha;
    public float gamma;
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    public void Execute(int threadIdx)
    {
        // f(x) = sigmoid(x) * x = x / (1 + exp(-x))
        // "Searching for Activation Functions". P Ramachandran, 2017
        // https://arxiv.org/abs/1710.05941
        float v = Xptr[threadIdx];
        v = v / (1.0f + exp(-v));
        Optr[threadIdx] = v;
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct ClipJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public float maxV;
    public float minV;
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    public void Execute(int threadIdx)
    {
        float v = Xptr[threadIdx];
        // math.clamp => if the minimum value is is greater than the maximum value, the method returns the minimum value.
        // this is not the expected behavior so changing it to minmax
        Optr[threadIdx] = min(maxV, max(v, minV));
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct ReluJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    float Apply(float v) { return 0.5f * (v + abs(v)); }
    public void Execute(int threadIdx)
    {
        float v = Xptr[threadIdx];
        Optr[threadIdx] = Apply(v);
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct Relu6Job : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    float Apply(float v) { return 0.5f * (-abs(v - 6.0f) + abs(v) + 6.0f); }
    public void Execute(int threadIdx)
    {
        float v = Xptr[threadIdx];
        Optr[threadIdx] = Apply(v);
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct TanhJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    float Apply(float v) { return tanh(clamp(v,-16.0f,16.0f));/*clamp to avoid NaNs for large values*/ }
    public void Execute(int threadIdx)
    {
        float v = Xptr[threadIdx];
        Optr[threadIdx] = Apply(v);
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct SigmoidJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    float Apply(float v) { return 1.0f / (1.0f + exp(-v)); }
    public void Execute(int threadIdx)
    {
        float v = Xptr[threadIdx];
        Optr[threadIdx] = Apply(v);
    }
}

}
}
