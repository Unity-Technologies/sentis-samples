using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine; // CustomYieldInstruction
using UnityEngine.Assertions;
using UnityEngine.Rendering;

namespace Unity.Sentis {

/// <summary>
/// Types of devices that Sentis uses to execute a neural network.
/// </summary>
public enum DeviceType
{
    /// <summary>
    /// Executes using the GPU.
    /// </summary>
    GPU = 1 << 8,

    /// <summary>
    /// Executes using the CPU.
    /// </summary>
    CPU = 1 << 9,
}

/// <summary>
/// Types of backend that Sentis uses to execute a neural network.
/// </summary>
public enum BackendType
{
    /// <summary>
    /// Executes using compute shaders on the GPU.
    /// </summary>
    GPUCompute = 0 | DeviceType.GPU,

    /// <summary>
    /// CommandBuffer implementation
    /// </summary>
    GPUCommandBuffer = 1 | DeviceType.GPU,

    /// <summary>
    /// Executes using pixel shaders on the GPU.
    /// </summary>
    GPUPixel = 2 | DeviceType.GPU,

    /// <summary>
    /// Executes using Burst on the CPU.
    /// </summary>
    CPU = 0 | DeviceType.CPU,
}

/// <summary>
/// An interface that allows you to execute neural networks (models).
///
/// `IWorker` abstracts implementation details on different hardware devices such as the CPU and the GPU. `IWorker` lets you do the following:
///
/// - Specify inputs.
/// - Schedule the work.
/// - Get outputs.
///
/// Internally, `IWorker` translates the neural network from a <see cref="Model"/> into a set of operations, then sends the operations to the hardware device for asynchronous execution.
///
/// Use `WorkerFactory.CreateWorker` or `Model.CreateWorker` to create a new instance of a worker.
/// </summary>
public interface IWorker : IDisposable
{
    /// <summary>
    /// Prepares the worker to execute the model using inputs of given shapes.
    /// </summary>
    /// <param name="inputShapes">A dictionary mapping input names to tensor shapes.</param>
    void PrepareForInput(IDictionary<string, TensorShape> inputShapes);

    /// <summary>
    /// Sets a tensor as the default input of the model. For models with more than one input this sets the first input.
    /// </summary>
    /// <param name="inputTensor">The tensor to set to the default input of the model.</param>
    void SetInput(Tensor inputTensor);

    /// <summary>
    /// Sets a tensor as a named input of the model.
    /// </summary>
    /// <param name="name">The name of the input to set.</param>
    /// <param name="inputTensor">The tensor to set as the input.</param>
    void SetInput(string name, Tensor inputTensor);

    /// <summary>
    /// Schedules the execution of the model on the worker. This is non-blocking.
    /// </summary>
    /// <returns>The `IWorker`.</returns>
    IWorker Execute();

    /// <summary>
    /// Sets a tensor as the default input of the model and schedules the execution of the model on the worker. This is non-blocking. For models with more than one input this sets the first input.
    /// </summary>
    /// <param name="inputTensor">The tensor to set to the default input of the model.</param>
    /// <returns>The `IWorker`.</returns>
    IWorker Execute(Tensor inputTensor);

    /// <summary>
    /// Sets multiple tensors as the inputs of the model and schedules execution of the model. This is non-blocking.
    /// </summary>
    /// <param name="inputTensors">The tensors to use as the inputs of the model as a dictionary mapping input names to tensors.</param>
    /// <returns>The `IWorker`.</returns>
    IWorker Execute(IDictionary<string, Tensor> inputTensors);

    /// <summary>
    /// Schedules the execution of the model one layer at a time. This is non-blocking.
    ///
    /// To schedule the execution of the next layer of the model, call `MoveNext` on the `IEnumerator` object this method returns.
    /// </summary>
    /// <returns>The `IEnumerator` for scheduling manual execution.</returns>
    IEnumerator StartManualSchedule();

    /// <summary>
    /// Sets a tensor as the default input of the model and schedules execution of the model one layer at a time. This is non-blocking. For models with more than one input this sets the first input.
    ///
    /// To schedule execution of the next layer of the model, call `MoveNext` on the `IEnumerator` object this method returns.
    /// </summary>
    /// <param name="inputTensor">The tensor to set to the default input of the model.</param>
    /// <returns>The `IEnumerator` for scheduling manual execution.</returns>
    IEnumerator StartManualSchedule(Tensor inputTensor);

    /// <summary>
    /// Sets multiple tensors as the inputs of the model and schedules execution of the model one layer at a time. This is non-blocking.
    ///
    /// To schedule execution of the next layer of the model, call `MoveNext` on the `IEnumerator` object this method returns.
    /// </summary>
    /// <param name="inputTensors">The tensors to use as the inputs of the model as a dictionary mapping input names to tensors.</param>
    /// <returns>The `IEnumerator` for scheduling manual execution.</returns>
    IEnumerator StartManualSchedule(IDictionary<string, Tensor> inputTensors);

    /// <summary>
    /// Schedules the execution of the part of the model that hasn't been scheduled yet. This is non-blocking.
    /// </summary>
    /// <param name="blocking">When the value is `true`, the method blocks further code until the model finishes executing.</param>
    void FlushSchedule(bool blocking = false);

    /// <summary>
    /// Returns the proportion of the model scheduled for execution since the last call to `StartManualSchedule`.
    ///
    /// Returns 0.0 after you call `StartManualSchedule`. Returns 1.0 when the model is fully scheduled.
    ///
    /// The value increases each time you iterate on the `IEnumerator` that `StartManualSchedule` returns.
    /// </summary>
    float scheduleProgress { get; }

    /// <summary>
    /// Returns a reference to the default output tensor. This is non-blocking.
    ///
    /// For models with more than one output this returns a reference to the first output tensor.
    ///
    /// The reference is valid only until you call `Execute()` or `Dispose()` on the worker.
    ///
    /// If you want to dispose of the worker but keep the tensor, use `FinishExecutionAndDownloadOutput()` instead, or use `TakeOwnership()` on the output tensor.
    /// </summary>
    /// <returns>The output tensor.</returns>
    Tensor PeekOutput();

    /// <summary>
    /// Returns a reference to an output tensor with a given `name`. This is non-blocking.
    ///
    /// The reference is valid only until you call `Execute()` or `Dispose()` on the worker.
    ///
    /// If you want to dispose of the worker but keep the tensor, use `FinishExecutionAndDownloadOutput()` instead, or use `TakeOwnership()` on the output tensor.
    /// </summary>
    /// <param name="name">The name of the output tensor to peek.</param>
    /// <returns>The output tensor.</returns>
    Tensor PeekOutput(string name);

    /// <summary>
    /// Returns a summary of the execution of the model.
    /// </summary>
    /// <returns>The summary of the execution.</returns>
    string Summary();

    /// <summary>
    /// Returns the backend used for execution.
    /// </summary>
    /// <returns>The `IBackend` used.</returns>
    IBackend GetBackend();
}

/// <summary>
/// Provides extension methods for the `IWorker` interface.
/// </summary>
public static class WorkerExtensions
{
    /// <summary>
    /// Returns a CPU copy of an output tensor. This is a blocking method, so the rest of your code waits until the model fully executes.
    /// </summary>
    /// <param name="worker">The worker to execute.</param>
    /// <param name="name">The optional name of the output tensor to copy. The first output tensor is used as a default.</param>
    /// <returns>A CPU copy of the output tensor.</returns>
    public static Tensor FinishExecutionAndDownloadOutput(this IWorker worker, string name = null)
    {
        var output = name == null ? worker.PeekOutput() : worker.PeekOutput(name);
        output.MakeReadable();
        return output;
    }

    /// <summary>
    /// Non-blocking API that schedules network execution on CommandBuffer in one go.
    /// </summary>
    /// <param name="cb">The command buffer to schedule execution on.</param>
    /// <param name="worker">The worker to use for execution.</param>
    /// <param name="inputs">A dictionary of input tensors.</param>
    public static void ExecuteWorker(this CommandBuffer cb, IWorker worker, Dictionary<string, Tensor> inputs)
    {
        var backend = worker.GetBackend();
        Assert.IsTrue(backend is GPUCommandBufferBackend);
        (backend as GPUCommandBufferBackend).cb = cb;
        worker.Execute(inputs);
    }

    /// <summary>
    /// Non-blocking API that schedules network execution on CommandBuffer in one go.
    /// </summary>
    /// <param name="cb">The command buffer to schedule execution on.</param>
    /// <param name="worker">The worker to use for execution.</param>
    /// <param name="input">The input tensor.</param>
    public static void ExecuteWorker(this CommandBuffer cb, IWorker worker, Tensor input)
    {
        var backend = worker.GetBackend();
        Assert.IsTrue(backend is GPUCommandBufferBackend);
        (backend as GPUCommandBufferBackend).cb = cb;
        worker.Execute(input);
    }
}

/// <summary>
/// Provides methods for instantiating workers and ops on given back ends.
/// </summary>
public class WorkerFactory
{
    /// <summary>
    /// Represents the configuration for a `WorkerFactory`.
    /// </summary>
    public struct WorkerConfiguration
    {
        /// <summary>
        /// Whether to log debug information about model execution to the Console window. The default is `false`.
        /// </summary>
        public bool verbose;

        /// <summary>
        /// If true the worker is allowed to take ownership of the weights memory from the model
        /// this is useful so worker to limit memory pressure when the worker need to copy those
        /// weight to a different device.
        /// </summary>
        public bool takeoverWeights;

        /// <summary>
        /// Initializes and returns an instance of `WorkerConfiguration`.
        /// </summary>
        /// <param name="verbose">Whether to use verbose logging.</param>
        /// <param name="takeoverWeights">Whether to allow the worker to take ownership of the model weights memory.</param>
        public WorkerConfiguration(bool verbose = false, bool takeoverWeights = false)
        {
            this.verbose = verbose;
            this.takeoverWeights = takeoverWeights;
        }
    }

    /// <summary>
    /// Initializes and returns an instance of `Ops` on a given back end.
    /// </summary>
    /// <param name="backendType">The type of backend to use.</param>
    /// <param name="allocator">The tensor allocator to use when allocating new tensors.</param>
    /// <returns>The created `Ops` instance.</returns>
    public static Ops CreateOps(BackendType backendType, ITensorAllocator allocator)
    {
        return BackendFactory.CreateOps(backendType, allocator, false);
    }

    /// <summary>
    /// Initializes and returns an instance of `IWorker` on a given back end with a `model` to execute and `workerConfiguration`.
    /// </summary>
    /// <param name="backendType">The type of backend to use.</param>
    /// <param name="model">The model to execute with this `IWorker`.</param>
    /// <param name="workerConfiguration">The worker configuration to use when executing.</param>
    /// <returns>The created `IWorker` instance.</returns>
    public static IWorker CreateWorker(BackendType backendType, Model model, WorkerConfiguration workerConfiguration)
    {
        return BackendFactory.CreateWorker(backendType, model, workerConfiguration);
    }

    /// <summary>
    /// Initializes and returns an instance of `IWorker` on a given back end with a `model` to execute.
    /// </summary>
    /// <param name="backendType">The type of backend to use.</param>
    /// <param name="model">The model to execute with this `IWorker`.</param>
    /// <param name="verbose">Whether to use verbose logging.</param>
    /// <returns>The created `IWorker` instance.</returns>
    public static IWorker CreateWorker(BackendType backendType, Model model, bool verbose = false)
    {
        var workerConfiguration = new WorkerConfiguration(verbose);
        return CreateWorker(backendType, model, workerConfiguration);
    }

    /// <summary>
    /// Initializes and returns an instance of `IWorker` on a given device with a `model` to execute. Sentis selects the best backend type available for `deviceType`.
    /// </summary>
    /// <param name="deviceType">The type of device to use. Sentis selects the best backend type available for `deviceType`.</param>
    /// <param name="model">The model to execute with this `IWorker`.</param>
    /// <param name="verbose">Whether to use verbose logging.</param>
    /// <returns>The created `IWorker` instance.</returns>
    public static IWorker CreateWorker(Model model, DeviceType deviceType, bool verbose = false)
    {
        var type = GetBestTypeForDevice(deviceType);
        var workerConfiguration = new WorkerConfiguration(verbose);
        return CreateWorker(type, model, workerConfiguration);
    }

    /// <summary>
    /// Checks if a backend type matches a device type. For example, `IsType(BackendType.GPUCompute, DeviceType.GPU)` returns `true`.
    /// </summary>
    /// <param name="backendType">The backend type to check.</param>
    /// <param name="deviceType">The device type to check.</param>
    /// <returns>Whether the backend type matches the device type.</returns>
    public static bool IsType(BackendType backendType, DeviceType deviceType)
    {
        return ((int)backendType & (int)deviceType) == (int)deviceType;
    }

    /// <summary>
    /// Returns the best backend type for the given `deviceType`.
    /// </summary>
    /// <param name="deviceType">The device type.</param>
    /// <returns>The selected backend type for the device type.</returns>
    public static BackendType GetBestTypeForDevice(DeviceType deviceType)
    {
        switch (deviceType)
        {
            case DeviceType.GPU:
                if (SystemInfo.supportsComputeShaders && ComputeInfo.supportsCompute)
                {
                    return BackendType.GPUCompute;
                }
                else
                {
                    return BackendType.GPUPixel;
                }
            default:
                return BackendType.CPU;
        }
    }
}

/// <summary>
/// Represents extensions for the `Model` class.
/// </summary>
public static class ModelExtensions
{
    /// <summary>
    /// Initializes and returns an instance of `IWorker` with a `model` to execute. Sentis selects the best backend type available for `deviceType`.
    ///
    /// This is a convenience method that internally calls `ModelLoader.Load` followed by `WorkerFactory.CreateWorker`.
    /// </summary>
    /// <param name="model">The model to execute.</param>
    /// <param name="deviceType">The preferred device for execution. For example `DeviceType.GPU` specifies the fast GPU path.</param>
    /// <param name="verbose">Whether to log scheduling of layers execution to the console.</param>
    /// <returns>The instantiated `IWorker`.</returns>
    public static IWorker CreateWorker(this Model model, DeviceType deviceType, bool verbose = false)
    {
        return WorkerFactory.CreateWorker(model, deviceType, verbose);
    }
}

/// <summary>
/// Represents extensions for the `ModelAsset` class.
/// </summary>
public static class ModelAssetExtensions
{
    /// <summary>
    /// Initializes and returns an instance of `IWorker` with a `modelAsset` to execute. Sentis selects the best backend type available for `device`.
    ///
    /// This is a convenience method that internally calls `ModelLoader.Load` followed by `WorkerFactory.CreateWorker`.
    /// </summary>
    /// <param name="modelAsset">The model asset to execute.</param>
    /// <param name="deviceType">The preferred device for execution. For example `DeviceType.GPU` specifies the fast GPU path</param>
    /// <param name="verbose">Whether to log scheduling of layers execution to the console.</param>
    /// <returns>The instantiated `IWorker`.</returns>
    public static IWorker CreateWorker(this ModelAsset modelAsset, DeviceType deviceType, bool verbose = false)
    {
        var model = ModelLoader.Load(modelAsset);
        return model.CreateWorker(deviceType, verbose);
    }
}

} // namespace Unity.Sentis
