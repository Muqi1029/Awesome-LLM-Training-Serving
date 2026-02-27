import torch
from triton.runtime import driver

# get device properties
DEVICE = driver.active.get_active_torch_device()
properties = driver.active.utils.get_device_properties(DEVICE.index)
torch_prop = torch.cuda.get_device_properties(DEVICE.index)

# extract core parameters
NUM_SM = properties["multiprocessor_count"]
NUM_REGS_PER_SM = properties["max_num_regs"]
SIZE_SMEM_PER_SM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
L2_CACHE = torch_prop.L2_cache_size

# compute metrics
total_regs_mb = (NUM_REGS_PER_SM * NUM_SM * 4) / 1024 / 1024  # 4 bytes per register
total_smem_kb = (SIZE_SMEM_PER_SM * NUM_SM) / 1024
# 计算理论显存带宽 (GB/s)
# 频率单位通常是 kHz，位宽是 bit。公式: 频率 * 2 (DDR) * 位宽 / 8 / 1e6
bw_gb_s = properties["mem_clock_rate"] * 2 * (properties["mem_bus_width"] / 8) / 1e6

print("=" * 50)
print(f"🚀 GPU Hardware Profile: {torch_prop.name}")
print(f"📍 Compute Capability:  {torch_prop.major}.{torch_prop.minor}")
print("=" * 50)

print(f"⚙️ Compute Resources:")
print(f"  • Streaming Multiprocessors (SMs) : {NUM_SM}")
print(
    f"  • Clock Rate                    : {properties['sm_clock_rate'] / 1e3:.2f} MHz"
)
print(f"  • Warp Size                     : {WARP_SIZE} threads")
print(f"  • Total Registers (Global)      : {total_regs_mb:.2f} MB")

print(f"\n🧠 Memory Hierarchy:")
print(
    f"  • Registers (per SM)            : {NUM_REGS_PER_SM * 4 / 1024:.2f} KB ({NUM_REGS_PER_SM} regs)"
)
print(f"  • Shared Memory (per SM)        : {SIZE_SMEM_PER_SM / 1024:.2f} KB")
print(f"  • L2 Cache Size                 : {L2_CACHE / 1024 / 1024:.2f} MB")
print(f"  • VRAM Total Capacity           : {torch_prop.total_memory / 1024**3:.2f} GB")

print(f"\n⚡ Throughput:")
print(f"  • Memory Bus Width              : {properties['mem_bus_width']} bit")
print(f"  • Max Memory Bandwidth          : {bw_gb_s:.2f} GB/s")
print("=" * 50)


import torch
from triton.runtime import driver


def get_cores_per_sm(major, minor):
    """
    Return CUDA cores and Tensor Core per SM based on compute capability
    """
    cc = (major, minor)

    cores_dict = {(8, 9): (128, 4), (9, 0): (128, 4)}  # RTX40, L40
    assert cc in cores_dict

    return cores_dict[cc]


def get_tc_fma_per_cycle(major, minor):
    cc = (major, minor)
    d = {(8, 9): 4 * 4 * 4, (9, 0): 512, (12, 0): 1024}
    assert cc in d
    return d[cc]


def profile_gpu(device_id):
    torch.cuda.set_device(device_id)

    DEVICE = driver.active.get_active_torch_device()
    properties = driver.active.utils.get_device_properties(DEVICE.index)
    torch_prop = torch.cuda.get_device_properties(device_id)

    NUM_SM = properties["multiprocessor_count"]
    NUM_REGS_PER_SM = properties["max_num_regs"]
    SIZE_SMEM_PER_SM = properties["max_shared_mem"]
    WARP_SIZE = properties["warpSize"]
    L2_CACHE = torch_prop.L2_cache_size

    # ===============================
    # Memory Bandwidth
    # ===============================
    bw_gb_s = properties["mem_clock_rate"] * 2 * (properties["mem_bus_width"] / 8) / 1e6

    # ===============================
    # Compute Throughput
    # ===============================
    cuda_cores_per_sm, tensor_cores_per_sm = get_cores_per_sm(
        torch_prop.major, torch_prop.minor
    )

    # convert kHz to Hz
    clock_hz = properties["sm_clock_rate"] * 1e3

    # FP32
    fp32_flops = NUM_SM * cuda_cores_per_sm * 2 * clock_hz  # x 2 due to FMA
    fp32_tflops = fp32_flops / 1e12

    # Tensor Core FP16/BF16 (very rough theoretical peak)
    fp16_flops = (
        NUM_SM
        * tensor_cores_per_sm
        * get_tc_fma_per_cycle(torch_prop.major, torch_prop.minor)
        * 2
        * clock_hz
    )
    fp16_tflops = fp16_flops / 1e12

    # INT8 (Tensor Core, 2x FP16 throughput roughly)
    int8_tflops = fp16_tflops * 2
    fp8_tflops = fp16_tflops * 2

    # Compute intensity (Roofline upper bound)
    compute_intensity = fp32_tflops / bw_gb_s

    # ===============================
    # Print Report
    # ===============================
    print("=" * 70)
    print(f"🚀 GPU [{device_id}] : {torch_prop.name}")
    print(f"📍 Compute Capability : {torch_prop.major}.{torch_prop.minor}")
    print("=" * 70)

    print("⚙️ Compute Resources")
    print(f"  • SM Count                    : {NUM_SM}")
    print(f"  • CUDA Cores / SM             : {cuda_cores_per_sm}")
    print(f"  • Tensor Cores / SM           : {tensor_cores_per_sm}")
    print(
        f"  • SM Clock                    : {properties['sm_clock_rate']/1e3:.2f} MHz"
    )
    print(f"  • Warp Size                   : {WARP_SIZE} threads")

    print("\n🧠 Memory Hierarchy")
    print(f"  • Registers / SM              : {NUM_REGS_PER_SM * 4 / 1024:.2f} KB")
    print(f"  • Shared Memory / SM          : {SIZE_SMEM_PER_SM / 1024:.2f} KB")
    print(f"  • L2 Cache                    : {L2_CACHE / 1024 / 1024:.2f} MB")
    print(
        f"  • Total VRAM                  : {torch_prop.total_memory / 1024**3:.2f} GB"
    )

    print("\n⚡ Throughput (Theoretical Peak)")
    print(f"  • Memory Bandwidth            : {bw_gb_s:.2f} GB/s")
    print(f"  • FP32                        : {fp32_tflops:.2f} TFLOPs")
    print(f"  • FP16/BF16 (Tensor Core)     : {fp16_tflops:.2f} TFLOPs")
    print(f"  • INT8 (Tensor Core)          : {int8_tflops:.2f} TOPS")
    print(f"  • FP8 (Tensor Core)           : {fp8_tflops:.2f} TOPS")

    print("\n📈 Roofline Upper Bound")
    print(f"  • Compute Intensity (FP32/BW) : {compute_intensity:.2f} FLOPs/Byte")
    print("=" * 70)
    print()


def main():
    num_devices = torch.cuda.device_count()

    if num_devices == 0:
        print("No CUDA devices found.")
        return

    print(f"\nDetected {num_devices} CUDA device(s)\n")

    for i in range(num_devices):
        profile_gpu(i)


if __name__ == "__main__":
    main()
