import torch
import time
import pandas as pd
from collections import defaultdict

# 导入您的模型和配置类
# 假设这些文件位于 caduceus 目录中
from caduceus.modeling_caduceus import CaduceusForMaskedLM
from caduceus.configuration_caduceus import CaduceusConfig

def benchmark_model(config: CaduceusConfig, seq_len: int, batch_size: int, device: torch.device, warmup_steps: int = 10, benchmark_steps: int = 20):
    """
    对给定配置的模型进行基准测试。

    Args:
        config (CaduceusConfig): 模型配置。
        seq_len (int): 输入序列的长度。
        batch_size (int): 批次大小。
        device (torch.device): 运行模型的设备 (e.g., 'cuda:0')。
        warmup_steps (int): 预热步数。增加此值有助于系统达到稳定状态。
        benchmark_steps (int): 实际测试步数。增加此值可以获得更稳定的平均吞吐量。

    Returns:
        tuple: (吞吐量 (token/s), 峰值内存MB)
    """
    # 设置数据类型为 float16
    dtype = torch.float16

    # 设置为评估模式并禁用梯度计算, 并转换为指定数据类型
    model = CaduceusForMaskedLM(config).to(device=device, dtype=dtype).eval()
    print(model)
    
    # 创建虚拟输入数据
    dummy_input = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    # 清空CUDA缓存并重置内存统计
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    # 预热：运行几次前向传播以确保CUDA内核已编译并加载
    # 这可以防止一次性开销影响后续的精确测量
    print("    - Warming up...")
    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = model(dummy_input)
    
    # 等待所有CUDA核心完成
    torch.cuda.synchronize()

    # --- 精确测量 ---
    # 在预热后、实际测试前重置峰值内存统计
    # 这是关键步骤，确保我们只捕获基准测试循环中的内存峰值
    torch.cuda.reset_peak_memory_stats(device)
    
    # 开始基准测试
    print("    - Running benchmark...")
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(benchmark_steps):
            _ = model(dummy_input)
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    # 计算指标
    total_time = end_time - start_time
    # 计算每秒处理的token数
    tokens_per_second = (benchmark_steps * batch_size * seq_len) / total_time
    # torch.cuda.max_memory_allocated() 会返回自上次重置以来的峰值内存
    peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    # 清理模型以释放内存
    del model
    del dummy_input
    torch.cuda.empty_cache()

    return tokens_per_second, peak_memory_mb

def main():
    """
    主函数，运行基准测试并打印结果。
    """
    if not torch.cuda.is_available():
        print("Error: This benchmark requires CUDA support.")
        return

    device = torch.device("cuda:0")
    print(f"Using device: {torch.cuda.get_device_name(device)}")

    # --- 基准测试参数 ---
    SEQ_LENS = [1024, 4096, 8192, 32768, 65536, 131072]
    BATCH_SIZE = 1  # 使用 batch_size=1 专注于单个序列的处理性能
    D_MODEL = 64
    N_LAYER = 8 # 总层数

    results = defaultdict(list)

    for seq_len in SEQ_LENS:
        print(f"\n--- Testing sequence length: {seq_len} ---")
        
        # --- 配置 1: 带注意力层 (GatedMamba) ---
        print("Configuration 1: GatedMamba...")
        try:
            # 为注意力层定义配置字典
            attn_config = {
                "num_heads": 8,  # 例如，设置为8个头
                "head_dim": D_MODEL // 8, # 确保 head_dim * num_heads = d_model
                "use_flash_attn": torch.cuda.is_available() # 如果可用，自动使用Flash Attention# 如果可用，自动使用Flash Attention
            }
            config_with_attn = CaduceusConfig(
                d_model=D_MODEL,
                n_modules=1,
                layers_per_module=N_LAYER,
                attn_layer_idx=[3], # 在第4层插入注意力
                vocab_size=12, # 假设词汇表大小为12
                attn_cfg=attn_config # 传入注意力配置
            )
            throughput, memory = benchmark_model(config_with_attn, seq_len, BATCH_SIZE, device)
            results["Sequence Length"].append(seq_len)
            results["Model"].append("GatedMamba")
            results["Throughput (token/s)"].append(f"{throughput:.2f}")
            results["Peak Memory (MB)"].append(f"{memory:.2f}")
            print(f"  Done. Throughput: {throughput:.2f} token/s, Peak Memory: {memory:.2f} MB")
        except torch.cuda.OutOfMemoryError:
            results["Sequence Length"].append(seq_len)
            results["Model"].append("GatedMamba")
            results["Throughput (token/s)"].append("OOM")
            results["Peak Memory (MB)"].append("OOM")
            print("  CUDA out of memory error (OOM).")
        except Exception as e:
            results["Sequence Length"].append(seq_len)
            results["Model"].append("GatedMamba")
            results["Throughput (token/s)"].append(f"Error: {type(e).__name__}")
            results["Peak Memory (MB)"].append(f"Error: {type(e).__name__}")
            print(f"  An unknown error occurred: {e}")


        # --- 配置 2: 无注意力层 (Caduceus) ---
        print("Configuration 2: Caduceus...")
        try:
            config_no_attn = CaduceusConfig(
                d_model=D_MODEL,
                n_modules=1,
                layers_per_module=N_LAYER,
                attn_layer_idx=[], # 
                vocab_size=12, # 假设词汇表大小为12
                attn_cfg=attn_config # 传入注意力配置
            )
            throughput, memory = benchmark_model(config_no_attn, seq_len, BATCH_SIZE, device)
            results["Sequence Length"].append(seq_len)
            results["Model"].append("Caduceus")
            results["Throughput (token/s)"].append(f"{throughput:.2f}")
            results["Peak Memory (MB)"].append(f"{memory:.2f}")
            print(f"  Done. Throughput: {throughput:.2f} token/s, Peak Memory: {memory:.2f} MB")
        except torch.cuda.OutOfMemoryError:
            results["Sequence Length"].append(seq_len)
            results["Model"].append("Caduceus")
            results["Throughput (token/s)"].append("OOM")
            results["Peak Memory (MB)"].append("OOM")
            print("  CUDA out of memory error (OOM).")
        except Exception as e:
            results["Sequence Length"].append(seq_len)
            results["Model"].append("Caduceus")
            results["Throughput (token/s)"].append(f"Error: {type(e).__name__}")
            results["Peak Memory (MB)"].append(f"Error: {type(e).__name__}")
            print(f"  An unknown error occurred: {e}")


        # --- 配置 3: 全注意力模型 (Flash-Attention) ---
        print("Configuration 3: Flash-Attention...")
        try:
            # 创建一个包含所有层索引的列表，使模型完全由注意力构成
            all_attn_layers = list(range(N_LAYER))
            config_all_attn = CaduceusConfig(
                d_model=D_MODEL,
                n_modules=1,
                layers_per_module=N_LAYER,
                attn_layer_idx=all_attn_layers, # 所有层都是注意力层
                vocab_size=12,
                attn_cfg=attn_config
            )
            throughput, memory = benchmark_model(config_all_attn, seq_len, BATCH_SIZE, device)
            results["Sequence Length"].append(seq_len)
            results["Model"].append("Flash-Attention")
            results["Throughput (token/s)"].append(f"{throughput:.2f}")
            results["Peak Memory (MB)"].append(f"{memory:.2f}")
            print(f"  Done. Throughput: {throughput:.2f} token/s, Peak Memory: {memory:.2f} MB")
        except torch.cuda.OutOfMemoryError:
            results["Sequence Length"].append(seq_len)
            results["Model"].append("Flash-Attention")
            results["Throughput (token/s)"].append("OOM")
            results["Peak Memory (MB)"].append("OOM")
            print("  CUDA out of memory error (OOM).")
        except Exception as e:
            results["Sequence Length"].append(seq_len)
            results["Model"].append("Flash-Attention")
            results["Throughput (token/s)"].append(f"Error: {type(e).__name__}")
            results["Peak Memory (MB)"].append(f"Error: {type(e).__name__}")
            print(f"  An unknown error occurred: {e}")

    # --- 打印最终结果 ---
    print("\n\n" + "="*60)
    print("Benchmark Final Results")
    print("="*60)
    df = pd.DataFrame(results)
    
    # 为了更好的可读性，将数据透视
    pivot_df = df.pivot(index='Sequence Length', columns='Model')
    print(pivot_df)
    print("="*60)
    print("\nBenchmark finished.")


if __name__ == "__main__":
    main()