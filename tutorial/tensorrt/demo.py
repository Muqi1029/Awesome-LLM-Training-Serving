import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

# STEP 1: create a network
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)

# STEP 2: define the dataflow and layers of the network
input_tensor = network.add_input("data", trt.float32, (1, 1, 4, 4))
w = np.ones((1, 1, 2, 2), dtype=np.float32)
b = np.zeros((1,), dtype=np.float32)
conv = network.add_convolution_nd(input_tensor, 1, (2, 2), w, b)
relu = network.add_activation(conv.get_output(0), trt.ActivationType.RELU)
network.mark_output(relu.get_output(0))

# STEP 3: build engine (serialize)
config = builder.create_builder_config()
print("Building Engine...")
serialized_engine = builder.build_serialized_network(
    network, config
)  # the most overhead
with open("simple_conv.engine", "wb") as f:
    f.write(serialized_engine)


# STEP 4: deserialize engine, create inference runtime
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(serialized_engine)
context = engine.create_execution_context()

# STEP 5: a demo to use the runtime
# 5.1: prepare input and output (cpu)
host_input = np.ones((1, 1, 4, 4), dtype=np.float32)
host_output = np.empty((1, 1, 3, 3), dtype=np.float32)  # 卷积后尺寸变 3x3

# 5.2 allocate input and output (device)
device_input = cuda.mem_alloc(host_input.nbytes)
device_output = cuda.mem_alloc(host_output.nbytes)

# 5.3 copy host to device
cuda.memcpy_htod(device_input, host_input)

# 5.4 execute the runtime
bindings = [int(device_input), int(device_output)]
context.execute_v2(bindings)

# 5.5 copy result back to output from device to host
cuda.memcpy_dtoh(host_output, device_output)

print("input data(4x4):")
print(host_input[0, 0])
print(type(host_input))
print("output(3x3):")
print(host_output[0, 0])
print(type(host_output))
