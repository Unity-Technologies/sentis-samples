using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Profiling;

using System.Runtime.CompilerServices;
using Unity.Sentis.Compiler.Analyser;

[assembly: InternalsVisibleTo("Unity.Sentis.PerformanceTests")]
[assembly: InternalsVisibleTo("Unity.Sentis.Tests")]

namespace Unity.Sentis
{

/// <summary>
/// Represents a generic implementation of an <see cref="IWorker"/>.
/// </summary>
public class GenericWorker : IWorker
{
    Model m_Model;
    string m_DefaultInputName;
    string m_DefaultOutputName;
    Dictionary<string, TensorShape> m_InputShapes = new Dictionary<string, TensorShape>();
    IModelCompiler m_ModelCompiler;

    IBackend m_Backend;
    IVars m_Vars;
    CPUBackend m_FallbackBackend;
    HashSet<string> m_LayerCPUFallback;

    bool m_AllocatorIsStale = false;
    bool m_AllocatorIsOccupied = false;
    bool m_Verbose;
    bool m_TakeoverWeights;
    float m_Progress = 0f;

    Tensor m_SyncTensor;

    /// <summary>
    /// Initializes and returns an instance of `GenericWorker` for the specified `model` and `ops`.
    /// </summary>
    /// <param name="model">The model to execute.</param>
    /// <param name="backend">The backend to use for execution.</param>
    /// <param name="vars">The stored tensor variables to use for execution.</param>
    /// <param name="verbose">Whether to enable verbose logging during execution.</param>
    /// <param name="takeoverWeights">Whether to allow the worker to take ownership of the model weights during execution.</param>
    public GenericWorker(Model model, IBackend backend, IVars vars, bool verbose = false, bool takeoverWeights = false)
    {
        m_Model = model;
        m_DefaultInputName = GraphLogicAnalysis.GetDefaultInputName(model);
        m_DefaultOutputName = GraphLogicAnalysis.GetDefaultOutputName(model);
        m_Vars = vars;
        m_Backend = backend;
        if (backend.GetType() == typeof(CPUBackend))
            m_FallbackBackend = (backend as CPUBackend);
        else
            m_FallbackBackend = new CPUBackend(m_Vars.GetAllocator());

        m_LayerCPUFallback = model.LayerCPUFallback;

        m_ModelCompiler = backend as IModelCompiler;
        m_Verbose = verbose;
        m_TakeoverWeights = takeoverWeights;

        m_AllocatorIsStale = true;
    }

    /// <summary>
    /// Finalizes the `GenericWorker`.
    /// </summary>
    ~GenericWorker()
    {
        Dispose();
    }

    /// <summary>
    /// Gets the backend used by the worker for execution.
    /// </summary>
    /// <returns>The backend used for execution.</returns>
    public IBackend GetBackend() { return m_Backend; }

    void OccupyAllocator()
    {
        m_AllocatorIsOccupied = true;
    }

    void ResetAllocatorIfStale()
    {
        if (m_AllocatorIsStale)
        {
            m_Backend.ResetAllocator();
            m_FallbackBackend.ResetAllocator();
            m_AllocatorIsStale = false;
            m_AllocatorIsOccupied = false;
        }
    }

    void ResetAllocatorIfStaleAndNotOccupied()
    {
        if (!m_AllocatorIsOccupied)
            ResetAllocatorIfStale();
    }

    /// <summary>
    /// Disposes of the worker and any associated memory.
    /// </summary>
    public virtual void Dispose()
    {
        m_Vars?.Dispose();
        m_Backend?.ResetAllocator(false); // clear allocator's memory
        m_FallbackBackend?.ResetAllocator(false); // clear allocator's memory
        m_InputShapes?.Clear();

        m_Vars = null;
        m_Backend = null;
        m_InputShapes = null;
        m_FallbackBackend = null;
    }

    /// <inheritdoc/>
    public virtual void PrepareForInput(IDictionary<string, TensorShape> inputShapes)
    {
        m_InputShapes.Clear();
        foreach (var input in inputShapes)
            m_InputShapes.Add(input.Key, input.Value);
        m_Vars.PrepareStorage(m_Model, m_Backend, m_InputShapes, m_TakeoverWeights);
    }

    /// <inheritdoc/>
    public virtual void SetInput(string name, Tensor x)
    {
        ResetAllocatorIfStale();
        OccupyAllocator();

        m_Vars.SetInput(name, x);

        // if single input network, then we have enough information to prepare network for execution
        if (m_Model.inputs.Count <= 1 && name == m_DefaultInputName)
        {
            PrepareForInput(new Dictionary<string, TensorShape> { { name, x.shape } }); // @TODO: get rid of allocation
        }

        m_InputShapes[name] = x.shape;
    }

    /// <inheritdoc/>
    public virtual void SetInput(Tensor x)
    {
        SetInput(m_DefaultInputName, x);
    }

    /// <inheritdoc/>
    public virtual IWorker Execute(IDictionary<string, Tensor> inputs)
    {
        foreach (var entry in inputs)
            SetInput(entry.Key, entry.Value);
        return Execute();
    }

    /// <inheritdoc/>
    public virtual IWorker Execute(Tensor input)
    {
        SetInput(input);
        return Execute();
    }

    /// <inheritdoc/>
    public virtual IWorker Execute()
    {
        Profiler.BeginSample ("Sentis.Execute");
        var enumerator = StartManualSchedule();
        while (enumerator.MoveNext()) {};
        Profiler.EndSample ();
        return this;
    }

    /// <inheritdoc/>
    public virtual void FlushSchedule(bool blocking)
    {
        // force execution of scheduled ops by requesting results of the intermediate tensor from the device
        m_SyncTensor.CompleteAllPendingOperations();
    }

    /// <inheritdoc/>
    public virtual IEnumerator StartManualSchedule(IDictionary<string, Tensor> inputs)
    {
        foreach (var entry in inputs)
            SetInput(entry.Key, entry.Value);
        return StartManualSchedule();
    }

    /// <inheritdoc/>
    public virtual IEnumerator StartManualSchedule(Tensor input)
    {
        SetInput(input);
        return StartManualSchedule();
    }

    /// <inheritdoc/>
    public virtual float scheduleProgress => m_Progress;

    /// <inheritdoc/>
    public virtual IEnumerator StartManualSchedule()
    {
        ResetAllocatorIfStaleAndNotOccupied();
        m_AllocatorIsStale = true;

        m_Vars.PrepareStorage(m_Model, m_Backend, m_InputShapes, m_TakeoverWeights);

        if (m_ModelCompiler != null)
            m_ModelCompiler.PrepareModel(m_Model, m_InputShapes, m_Vars);

        ExecutionContext ctx = new ExecutionContext();
        ctx.vars = m_Vars;

        int idx = 0;
        foreach (var l in m_Model.layers)
        {
            idx++;

            m_Progress = idx / (float)m_Model.layers.Count;

            Profiler.BeginSample(l.name);

            var inputs = m_Vars.GatherInputs(l);

            if (m_Verbose)
                D.Log(l);

            m_Vars.PrepareStorage(l);
            if (m_ModelCompiler != null)
                m_ModelCompiler.PreExecuteLayer(l, inputs);

            ctx.backend = m_Backend;
            if (m_LayerCPUFallback.Contains(l.name))
                ctx.backend = m_FallbackBackend;

            Profiler.BeginSample(l.profilerTag);
            Tensor X = l.Execute(inputs, ctx);
            Profiler.EndSample();

            m_Vars.DisposeAfterLayer(l);
            m_Vars.Store(l, X);
            m_SyncTensor = X;

            // layer.name
            Profiler.EndSample();

            yield return null;
        }

        // request ResetAllocator before next Execute() starts
        m_AllocatorIsOccupied = false;

        if (m_Verbose)
            D.Log(m_Vars.GetAllocator());
    }

    /// <inheritdoc/>
    public virtual Tensor PeekOutput()
    {
        Profiler.BeginSample("Sentis.PeekOutput");
        var X = m_Vars.PeekOutput(m_DefaultOutputName);
        Profiler.EndSample();

        return X;
    }

    /// <inheritdoc/>
    public virtual Tensor PeekOutput(string name)
    {
        Profiler.BeginSample("Sentis.PeekOutput");
        var X = m_Vars.PeekOutput(name);
        Profiler.EndSample();

        return X;
    }

    /// <summary>
    /// Returns a summary of the execution as a string.
    /// </summary>
    /// <returns>The string representation of the allocator and backend states.</returns>
    public virtual string Summary()
    {
        return m_Vars.GetAllocator().ToString() + "\n" + m_Backend.ToString();
    }
}

class GenericVars : IVars
{
    Dictionary<string, Tensor> m_TensorsByName = new Dictionary<string, Tensor>();
    HashSet<Tensor> m_ModelTensors = new HashSet<Tensor>();
    Dictionary<string, Tensor[]> m_InputTensorsByLayer = new Dictionary<string, Tensor[]>();
    Dictionary<string, int> m_LayerNameToId = new Dictionary<string, int>();
    Dictionary<string, List<int>> m_LayerNameToDisposeWhenDone = new Dictionary<string, List<int>>();
    Dictionary<int, Layers.Layer> m_LayerIdToLayer = new Dictionary<int, Layers.Layer>();
    ITensorAllocator m_Allocator;

    public GenericVars(bool forceCachingByShape = false)
    {
        if (forceCachingByShape)
            m_Allocator = new TensorCachingByShapeAllocator();
        else
            m_Allocator = new DefaultTensorAllocator();
    }

    ~GenericVars()
    {
        Dispose();
    }

    /// <summary>
    /// Disposes of the worker and any associated memory.
    /// </summary>
    public virtual void Dispose()
    {
        foreach (var t in m_ModelTensors)
            t.Dispose();
        m_ModelTensors.Clear();

        // don't dispose input/user-owned tensors
        foreach (var ts in m_InputTensorsByLayer.Values)
        {
            var isVisited = false;
            foreach (var t in ts)
            {
                if (t == null)
                    continue;
                isVisited = true;
                if (IsTensorOwnedByInternalAllocator(t))
                    t.Dispose();
            }
            // m_InputTensorsByLayer is null initialized to avoid gc, if layer isn't visited nothing to dispose
#if (UNITY_ASSERTIONS)
            if (!isVisited)
                D.Log("GenericWorker.GenericVars.Dispose: un-visited layer, execution most likely failed mid way");
#endif
        }
        m_InputTensorsByLayer.Clear();

        m_LayerNameToId.Clear();
        m_LayerNameToDisposeWhenDone.Clear();
        m_LayerIdToLayer.Clear();

        m_Allocator.Dispose();
    }

    /// <inheritdoc/>
    public virtual ITensorAllocator GetAllocator()
    {
        return m_Allocator;
    }

    protected virtual bool IsTensorOwnedByInternalAllocator(Tensor tensor)
    {
        return tensor.allocator == GetAllocator();
    }

    protected bool ValidateGlobalInputs(Model model, IDictionary<string, TensorShape> inputShapes)
    {
        var valid = true;
        foreach (var input in model.inputs)
        {
            if (m_TensorsByName.TryGetValue(input.name, out var tensor))
            {
                if (tensor.dataType != input.dataType)
                    D.LogWarning($"Given input data type: {tensor.dataType} does not match that of model input tensor: {input.dataType} for input: {input.name}");
                model.ValidateInputTensorShape(input, tensor.shape);
                continue;
            }

            if (inputShapes != null && inputShapes.TryGetValue(input.name, out var tensorShape))
            {
                model.ValidateInputTensorShape(input, tensorShape);
                continue;
            }

            D.LogWarning("Global input is missing: " + input.name);
            valid = false;
        }
        return valid;
    }

    protected virtual Tensor[] PrepareLayerInputTensors(Layers.Layer layer, IBackend backend)
    {
        return new Tensor[layer.inputs.Length];
    }

    /// <inheritdoc/>
    public virtual void SetInput(string name, Tensor X)
    {
        m_TensorsByName[name] = X;
    }

    /// <inheritdoc/>
    public virtual void PrepareStorage(Model model, IBackend backend, IDictionary<string, TensorShape> inputShapes, bool takeoverWeights)
    {
        ValidateGlobalInputs(model, inputShapes);

        m_LayerNameToId.Clear();
        m_LayerNameToDisposeWhenDone.Clear();
        m_LayerIdToLayer.Clear();

        for(var i = 0; i < model.constants.Count; i++)
        {
            var constant = model.constants[i];
            Tensor tensor = AllocatorUtils.NewTensor(constant.dataType, constant.shape, new SharedArrayTensorData(constant.shape, constant.weights, (int)constant.offset), null);

            m_TensorsByName[constant.name] = tensor;
            m_ModelTensors.Add(tensor);
            if (takeoverWeights)
                constant.weights = null;
        }

        for (var i = 0; i < model.layers.Count; i++)
        {
            var layer = model.layers[i];

            // prepare input placeholders and argument tensors only once per layer
            if (m_InputTensorsByLayer.ContainsKey(layer.name))
                continue;

            var tensors = PrepareLayerInputTensors(layer, backend);
            m_InputTensorsByLayer.Add(layer.name, tensors);
        }

        // For each layer we find the latest downstream layer that has said layer as input
        // ex:
        // 0 -> 1 -> 4 -> 5 -> 8
        //   -> 2 -> 3  /     |
        //   -> 7 ------------/
        // latestDownstreamLayer:
        //  0 -> 7, 1 -> 4, 2 -> 3, 4 -> 5, 5 -> 8, 7 -> 8
        Dictionary<string, int> latestDownstreamLayer = new Dictionary<string, int>();
        for (var i = 0; i < model.layers.Count; i++)
        {
            var forLayer = model.layers[i];
            m_LayerNameToId[forLayer.name] = i;
            m_LayerIdToLayer[i] = forLayer;

            foreach (var input in forLayer.inputs)
            {
                if (string.IsNullOrEmpty(input))
                    continue;
                if (latestDownstreamLayer.ContainsKey(input))
                    latestDownstreamLayer[input] = Math.Max(latestDownstreamLayer[input], i);
                else
                    latestDownstreamLayer[input] = i;
            }
        }

        // now that we have the latestDownstreamLayer, we inverse the map
        // and compute when we reach a layer, what layers can I delete
        // in this case
        // 3 -> [2], 4 -> [1], 5 -> [4,3] , 7 -> [0], 8 -> [5,7]

        // keep layer if output
        var preserve = new HashSet<string>(
            model.inputs.Select(i => i.name).Concat(
            model.outputs));

        foreach (var entry in latestDownstreamLayer)
        {
            if(preserve.Contains(entry.Key))
                continue;
            // input can be not specified
            if(!m_LayerNameToId.ContainsKey(entry.Key))
                continue;

            var forLayer = m_LayerIdToLayer[entry.Value];
            if (m_LayerNameToDisposeWhenDone.ContainsKey(forLayer.name))
                m_LayerNameToDisposeWhenDone[forLayer.name].Add(m_LayerNameToId[entry.Key]);
            else
                m_LayerNameToDisposeWhenDone[forLayer.name] = new List<int>() { m_LayerNameToId[entry.Key] };
        }
    }

    /// <inheritdoc/>
    public virtual Tensor[] GatherInputs(Layers.Layer forLayer)
    {
        var tensors = m_InputTensorsByLayer[forLayer.name];

        for (var i = 0; i < forLayer.inputs.Length; i++)
        {
            tensors[i] = string.IsNullOrEmpty(forLayer.inputs[i]) ? null : PeekOutput(forLayer.inputs[i]);
        }

        return tensors;
    }

    /// <inheritdoc/>
    public virtual void PrepareStorage(Layers.Layer forLayer) {}

    /// <inheritdoc/>
    public virtual void DisposeAfterLayer(Layers.Layer forLayer)
    {
        if(!m_LayerNameToDisposeWhenDone.ContainsKey(forLayer.name))
            return;

        foreach (var layerIdxToDispose in m_LayerNameToDisposeWhenDone[forLayer.name])
        {
            var l = m_LayerIdToLayer[layerIdxToDispose];
            var key = l.name;

            if (!(m_TensorsByName.ContainsKey(key) && !m_ModelTensors.Contains(m_TensorsByName[key])))
                continue;

            if (IsTensorOwnedByInternalAllocator(m_TensorsByName[key]))
                m_TensorsByName[key].Dispose();
            m_TensorsByName.Remove(key);
        }
    }

    /// <inheritdoc/>
    public virtual void Store(string fromLayer, Tensor result)
    {
        // @TODO: implement Disposal of the old tensor that is going to be overwritten with new one
        // NOTE: need to make IWorker.FinishExecutionAndDownloadOutput to do real copy before enabling code below
        // otherwise there is a risk of Disposing tensor that is already owned by the user, if one calls FinishExecutionAndDownloadOutput on m_TensorsByName
        // if (m_TensorsByName.ContainsKey(fromLayer.name))
        // {
        //     var oldTensor = m_TensorsByName[fromLayer.name];
        //     if (oldTensor != result && IsTensorOwnedByInternalAllocator(oldTensor))
        //         oldTensor.Dispose();
        // }

        m_TensorsByName[fromLayer] = result;
    }

    /// <inheritdoc/>
    public virtual void Store(Layers.Layer fromLayer, Tensor result)
    {
        Store(fromLayer.name, result);
    }

    /// <inheritdoc/>
    public virtual Tensor PeekOutput(string name)
    {
        if (!m_TensorsByName.ContainsKey(name))
            D.LogWarning("GenericVars missing variable: " + name);

        return m_TensorsByName[name];
    }
}

class GenericVarsWithReuse : GenericVars
{
    Model m_CachedModel;
    bool m_LayerRequiresStorage;
    HashSet<Layers.Layer> m_LayersWithStorage;
    Tensor m_Temporary;
    string m_TemporaryName;
    IDictionary<string, TensorShape> m_CachedInputShapes;

    internal bool layerRequiresStorage => m_LayerRequiresStorage;
    protected Tensor temporary => m_Temporary;

    public GenericVarsWithReuse(bool forceCachingByShape = false) : base(forceCachingByShape) { }

    protected void ReleaseTemporary()
    {
        m_TemporaryName = null;
        if (m_Temporary == null)
            return;

        if (IsTensorOwnedByInternalAllocator(m_Temporary))
            m_Temporary.Dispose();
        m_Temporary = null;
    }

    /// <inheritdoc/>
    public override void PrepareStorage(Model model, IBackend backend, IDictionary<string, TensorShape> inputShapes, bool takeoverWeights)
    {
        if(m_CachedInputShapes != inputShapes)
        {
            m_CachedInputShapes = inputShapes;
            base.PrepareStorage(model, backend, inputShapes, takeoverWeights);
        }

        ReleaseTemporary();

        if (m_CachedModel != model)
            m_LayersWithStorage = MemoryFootprintAnalysis.FindLayersThatRequireStorage(model);
        m_CachedModel = model;

        Assert.AreEqual(m_Temporary, null);
    }

    /// <inheritdoc/>
    public override void PrepareStorage(Layers.Layer forLayer)
    {
        base.PrepareStorage(forLayer);
        m_LayerRequiresStorage = m_LayersWithStorage.Contains(forLayer);
    }

    /// <inheritdoc/>
    public override void Store(Layers.Layer fromLayer, Tensor result)
    {
        if (result != m_Temporary)
            ReleaseTemporary();

        if (layerRequiresStorage)
        {
            Assert.IsNotNull(result);
            base.Store(fromLayer, result);

            m_Temporary = null;
            m_TemporaryName = null;
        }
        else
        {
            Assert.IsTrue(m_Temporary == null || m_Temporary.tensorOnDevice == result.tensorOnDevice);

            m_Temporary = result;
            m_TemporaryName = fromLayer.name;
        }
    }

    /// <inheritdoc/>
    public override Tensor PeekOutput(string name)
    {
        if (m_TemporaryName == name)
        {
            Assert.IsNotNull(m_Temporary);
            return m_Temporary;
        }

        return base.PeekOutput(name);
    }
}

class GenericVarsWithPreallocation : GenericVarsWithReuse, ITensorAllocator
{
    Model m_CachedModel;

    DefaultTensorAllocator m_InferenceScopedPingPongAllocator = new DefaultTensorAllocator();
    DefaultTensorAllocator m_InferenceScopedStorageAllocator = new DefaultTensorAllocator();
    TempTensorCachingAllocator m_LayerScopedAllocator = new TempTensorCachingAllocator();

    /// <inheritdoc/>
    public virtual void PostLayerCleanup()
    {
        m_InferenceScopedPingPongAllocator.PostLayerCleanup();
        m_InferenceScopedStorageAllocator.PostLayerCleanup();
        m_LayerScopedAllocator.PostLayerCleanup();
    }

    /// <inheritdoc/>
    public override void PrepareStorage(Model model, IBackend backend, IDictionary<string, TensorShape> inputShapes, bool takeoverWeights)
    {
        base.PrepareStorage(model, backend, inputShapes, takeoverWeights);

        m_CachedModel = model;

        m_InferenceScopedPingPongAllocator.PostLayerCleanup();//reset allocation count
    }

    /// <inheritdoc/>
    public override void DisposeAfterLayer(Layers.Layer forLayer)
    {
        PostLayerCleanup();

        base.DisposeAfterLayer(forLayer);
    }

    /// <inheritdoc/>
    public override ITensorAllocator GetAllocator()
    {
        return this;
    }

    /// <inheritdoc/>
    protected override bool IsTensorOwnedByInternalAllocator(Tensor tensor)
    {
        var allocator = tensor.allocator;
        return allocator == m_InferenceScopedPingPongAllocator ||
               allocator == m_InferenceScopedStorageAllocator ||
               allocator == m_LayerScopedAllocator;
    }

    /// <inheritdoc/>
    public virtual Tensor Alloc(TensorShape shape, DataType dataType, DeviceType deviceType, AllocScope scope)
    {
        if (scope == AllocScope.InternalToLayer)
            return m_LayerScopedAllocator.Alloc(shape, dataType, deviceType, scope);

        if (layerRequiresStorage)
            return m_InferenceScopedStorageAllocator.Alloc(shape, dataType, deviceType, scope);
        else
            return m_InferenceScopedPingPongAllocator.Alloc(shape, dataType, deviceType, scope);
    }

    /// <inheritdoc/>
    public virtual Tensor Alloc(TensorShape shape, DataType dataType, ITensorData buffer, AllocScope scope)
    {
        if (scope == AllocScope.InternalToLayer)
            return m_LayerScopedAllocator.Alloc(shape, dataType, buffer, scope);

        if (layerRequiresStorage)
            return m_InferenceScopedStorageAllocator.Alloc(shape, dataType, buffer, scope);
        else
            return m_InferenceScopedPingPongAllocator.Alloc(shape, dataType, buffer, scope);
    }

    /// <inheritdoc/>
    public virtual void MoveToDevice(Tensor X, ITensorData newBuffer, ITensorData oldBuffer, bool disposeDetachedBufferHint)
    {
        X.allocator.MoveToDevice(X, newBuffer, oldBuffer, disposeDetachedBufferHint);
    }

    /// <inheritdoc/>
    public virtual void Release(Tensor tensor, bool calledFromTensorDispose)
    {
        tensor.allocator.Release(tensor, calledFromTensorDispose);
    }

    /// <inheritdoc/>
    public virtual void WaiveOwnership(Tensor X)
    {
        X.allocator.WaiveOwnership(X);
    }

    /// <inheritdoc/>
    public virtual void Reset(bool keepCachedMemory)
    {
        m_InferenceScopedPingPongAllocator.Reset(keepCachedMemory);
        m_InferenceScopedStorageAllocator.Reset(keepCachedMemory);
        m_LayerScopedAllocator.Reset(keepCachedMemory);
    }

    /// <summary>
    /// Disposes of the worker and any associated memory.
    /// </summary>
    public override void Dispose()
    {
        base.Dispose();

        m_InferenceScopedPingPongAllocator.Dispose();
        m_InferenceScopedStorageAllocator.Dispose();
        m_LayerScopedAllocator.Dispose();
    }
}

class DefaultTensorAllocator : TensorCachingAllocator { }

class DefaultVars : GenericVarsWithPreallocation { }


} // namespace Unity.Sentis
