import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import numpy as np
import re

# ================================
# 实验配置
# ================================
# 1. 测试不同的序列长度
SEQ_LENS_TO_TEST = [1024, 8192, 16384]

# 2. 训练参数
MAX_STEPS = 5000
LEARNING_RATE = "1e-3"  # 为小型实验设置一个稳健的学习率
NUM_DEVICES = 2
ACCUMULATE_GRAD_BATCHES = 4
NUM_WORKERS = 8
VAL_CHECK_INTERVAL = MAX_STEPS // 5 * ACCUMULATE_GRAD_BATCHES  # 新增：验证频率

# 3. 固定的模型架构
D_MODEL = 256
D_INTERMEDIATE = 256
N_MODULES = 1
LAYERS_PER_MODULE = 3
CONV_LAYERS_PER_MODULE = 0  # 不使用卷积
# 注意力层在模块内的索引 (0-indexed)。第4层意味着索引为3。
# 由于模块内只有8层，所以索引范围是0-7。
# ATTN_LAYER_IN_MODULE = 2 
ATTN_LAYER_IDX = [0, 1, 2] # 动态生成索引列表

# 4. 注意力配置 (参考 run_pretrain_wisteria.sh)
ATTN_NUM_HEADS = 4
ATTN_HEAD_DIM = D_MODEL // ATTN_NUM_HEADS
ATTN_MLP_DIM = 256 # 通常是 d_model 的倍数

# 5. 傅里叶位置编码 (FoPE) 的核心配置
USE_FOURIER_POS_EMB = True # 这是一个模板值，会在循环中被覆盖
# FOURIER_MAX_SEQ_LEN = 8192 # 固定为预训练值
FOURIER_LEARNABLE = False    # 设置为可学习

# 6. 新增：更多可调整的 FoPE 参数 (参考 wisteria.yaml)
FOURIER_SEPARATE_BASIS = True
FOURIER_SEPARATE_HEAD = True
FOURIER_NORM = False
FOURIER_IGNORE_ZERO = True
FOURIER_INIT = "eye_xavier_norm"
FOURIER_INIT_NORM_GAIN = 0.3
# 使用 'null' 字符串来表示 yaml 中的 null 值，如果想指定维度，可设为如 32, 64 等
FOURIER_DIM = "null" 

# 7. 其他关键配置 (参考 run_pretrain_wisteria.sh)
RC_AUG = True
BIDIRECTIONAL = True
BIDIRECTIONAL_STRATEGY = "add"
BIDIRECTIONAL_WEIGHT_TIE = True

# 8. 输出目录
OUTPUT_DIR = "./fpe_vs_rope_comparison_results"

# 9. WandB 配置
WANDB_PROJECT = "fpe_vs_rope_comparison"

# ================================
# 辅助函数
# ================================
def run_experiment(seq_len, use_fpe):
    """启动单次训练实验"""
    # 动态计算批次大小
    global_batch_size = 1048576 // seq_len
    effective_batch_size = global_batch_size // NUM_DEVICES // ACCUMULATE_GRAD_BATCHES
    batch_size_eval = global_batch_size // NUM_DEVICES * 2  # 评估批次大小为训练批次的两倍

    tag = "FoPE" if use_fpe else "RoPE"
    exp_name = f"seq{seq_len}_{tag}_d{D_MODEL}_nModules{N_MODULES}_Layers{LAYERS_PER_MODULE}"
    run_dir = os.path.join(OUTPUT_DIR, exp_name)
    # 定义一个专门存放 stdout 日志的文件
    stdout_log_path = os.path.join(run_dir, "stdout.log")

    if os.path.exists(run_dir) and os.path.exists(stdout_log_path):
        print(f"目录和日志已存在，跳过实验: {run_dir}")
        return run_dir

    print("\n" + "="*80)
    print(f"运行实验: 序列长度={seq_len}, 使用位置编码={tag}")
    print(f"  - 全局批次大小: {global_batch_size}")
    print(f"  - 单设备有效批次大小: {effective_batch_size}")
    print("="*80)

    # Hydra的配置覆盖列表
    overrides = [
        f"experiment=hg38/hg38",
        # 数据集配置
        f"dataset.max_length={seq_len}",
        f"dataset.batch_size={effective_batch_size}",
        f"dataset.batch_size_eval={batch_size_eval}",  # 评估批次大小为训练批次的两倍
        f"dataset.mlm=true",
        f"dataset.mlm_probability=0.15",
        f"dataset.rc_aug={str(RC_AUG).lower()}",
        # 数据加载器配置
        f"loader.num_workers={NUM_WORKERS}",
        # 模型配置
        f"model=wisteria",
        f"model.config.d_model={D_MODEL}",
        f"model.config.d_intermediate={D_INTERMEDIATE}",
        f"model.config.n_modules={N_MODULES}",
        f"model.config.layers_per_module={LAYERS_PER_MODULE}",
        f"model.config.conv_layers_per_module={CONV_LAYERS_PER_MODULE}",
        # f"model.config.attn_layer_in_module={ATTN_LAYER_IN_MODULE}",
        f"model.config.attn_layer_idx={ATTN_LAYER_IDX}",
        f"model.config.bidirectional={str(BIDIRECTIONAL).lower()}",
        f"model.config.bidirectional_strategy={BIDIRECTIONAL_STRATEGY}",
        f"model.config.bidirectional_weight_tie={str(BIDIRECTIONAL_WEIGHT_TIE).lower()}",
        # 注意力头配置
        f"model.config.attn_cfg.num_heads={ATTN_NUM_HEADS}",
        f"model.config.attn_cfg.head_dim={ATTN_HEAD_DIM}",
        f"model.config.attn_cfg.mlp_dim={ATTN_MLP_DIM}",
    ]
    
    # 根据 use_fpe 动态添加位置编码配置
    overrides.append(f"model.config.use_fourier_pos_emb={str(use_fpe).lower()}")

    if use_fpe:
        # 如果使用 FoPE，则添加所有相关的详细参数
        overrides.extend([
            f"model.config.fourier_max_seq_len={seq_len}",
            f"model.config.fourier_learnable={str(FOURIER_LEARNABLE).lower()}",
            f"model.config.fourier_separate_basis={str(FOURIER_SEPARATE_BASIS).lower()}",
            f"model.config.fourier_separate_head={str(FOURIER_SEPARATE_HEAD).lower()}",
            f"model.config.fourier_norm={str(FOURIER_NORM).lower()}",
            f"model.config.fourier_ignore_zero={str(FOURIER_IGNORE_ZERO).lower()}",
            f"model.config.fourier_init={FOURIER_INIT}",
            f"model.config.fourier_init_norm_gain={FOURIER_INIT_NORM_GAIN}",
            f"model.config.fourier_dim={FOURIER_DIM}",
        ])
        # 禁用 RoPE
        overrides.append(f"model.config.attn_cfg.rotary_emb_dim=0")
    else:
        # 如果使用 RoPE，则设置其维度
        overrides.append(f"model.config.attn_cfg.rotary_emb_dim={ATTN_HEAD_DIM}")

    # 添加剩余的配置
    overrides.extend([
        # 优化器配置
        f"optimizer.lr={LEARNING_RATE}",
        # 训练器配置
        f"trainer.devices={NUM_DEVICES}",
        f"trainer.max_steps={MAX_STEPS}",
        f"trainer.accumulate_grad_batches={ACCUMULATE_GRAD_BATCHES}",
        "callbacks.model_checkpoint_every_n_steps.every_n_train_steps=500",
        f"+trainer.val_check_interval={VAL_CHECK_INTERVAL}",
        f"train.global_batch_size={global_batch_size}",
        # 路径和日志配置
        f"hydra.run.dir={run_dir}",
        # WandB 配置
        f"wandb.project={WANDB_PROJECT}",
        f"wandb.name={exp_name}",
        f"wandb.group=seq_len_{seq_len}",
    ])

    command = ["python", "-m", "train"] + overrides
    env = os.environ.copy()
    env["MKL_THREADING_LAYER"] = "GNU"
    env["HYDRA_FULL_ERROR"] = "1"

    try:
        # 确保目录存在
        os.makedirs(run_dir, exist_ok=True)
        
        # --- 修改开始：使用 Popen 实时显示和捕获输出 ---
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 将错误输出合并到标准输出
            text=True,
            encoding='utf-8',
            env=env
        )

        # 实时读取和打印输出，同时保存到列表中
        stdout_lines = []
        for line in process.stdout:
            print(line, end='')  # 实时打印到终端
            stdout_lines.append(line) # 保存以供后续写入文件

        # 等待进程结束
        process.wait()

        # 检查进程是否成功完成
        if process.returncode != 0:
            # 如果失败，手动抛出异常，行为与 check=True 类似
            raise subprocess.CalledProcessError(
                returncode=process.returncode,
                cmd=command,
                output=''.join(stdout_lines)
            )
        
        print(f"实验 {exp_name} 成功完成。")
        # 将捕获的 stdout 保存到文件
        with open(stdout_log_path, "w", encoding='utf-8') as f:
            f.write(''.join(stdout_lines))
        # --- 修改结束 ---

    except subprocess.CalledProcessError as e:
        print(f"运行实验 {exp_name} 时出错。返回码: {e.returncode}")
        print("\n--- 完整输出日志 ---")
        # e.output 现在包含了完整的日志，因为我们已经捕获了它
        print(e.output)
        print("--------------------")
        return None
        
    return run_dir

def plot_results(results):
    """从 stdout.log 文件解析损失并绘制对比图表"""
    num_seq_lens = len(SEQ_LENS_TO_TEST)
    fig, axes = plt.subplots(1, num_seq_lens, figsize=(6 * num_seq_lens, 5), sharey=True)
    if num_seq_lens == 1:
        axes = [axes]

    # 正则表达式用于从 TQDM 进度条中提取 step 和 loss
    # 匹配 'loss=X.XXX' 和 'step=Y'
    log_pattern = re.compile(r"loss=([0-9.]+).*?step=(\d+)")

    for i, seq_len in enumerate(SEQ_LENS_TO_TEST):
        ax = axes[i]
        for use_fpe in [True, False]:
            tag = "FoPE" if use_fpe else "RoPE"
            exp_name = f"seq{seq_len}_{tag}_d{D_MODEL}"
            
            if exp_name not in results or results[exp_name] is None:
                continue

            # 从新的 stdout.log 文件读取
            log_file = os.path.join(results[exp_name], "stdout.log")
            
            if not os.path.exists(log_file):
                print(f"警告: 未找到日志文件: {log_file}")
                continue

            steps, losses = [], []
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        # TQDM 在同一行更新，所以我们只关心包含 'loss=' 和 'step=' 的行
                        if 'loss=' in line and 'step=' in line:
                            match = log_pattern.search(line)
                            if match:
                                # 注意：分组顺序与新的正则表达式匹配
                                loss_val = float(match.group(1))
                                step_val = int(match.group(2))
                                steps.append(step_val)
                                losses.append(loss_val)
            except Exception as e:
                print(f"读取或解析日志文件 {log_file} 时出错: {e}")
                continue

            if not steps:
                print(f"警告: 未能在 {log_file} 中找到任何损失数据。")
                continue

            # 创建 DataFrame 并进行平滑处理
            df = pd.DataFrame({"step": steps, "train/loss": losses}).drop_duplicates(subset='step').set_index('step').sort_index()
            smoothed_loss = df['train/loss'].rolling(window=50, min_periods=1).mean()

            ax.plot(smoothed_loss.index, smoothed_loss.values, label=tag)

        ax.set_title(f"Sequence Length: {seq_len}")
        ax.set_xlabel("Training Steps")
        if i == 0:
            ax.set_ylabel("Training Loss (Smoothed)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle("FoPE vs. RoPE Training Loss Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plot_path = os.path.join(OUTPUT_DIR, "comparison_chart.png")
    plt.savefig(plot_path)
    print(f"\n对比图表已保存至: {plot_path}")
    plt.show()

# ================================
# 主执行流程
# ================================
def main():
    """主执行函数"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    experiment_results = {}
    for seq_len in SEQ_LENS_TO_TEST:
        # 运行 RoPE (use_fpe=False)
        exp_name_rope = f"seq{seq_len}_RoPE_d{D_MODEL}"
        run_dir_rope = run_experiment(seq_len, use_fpe=False)
        experiment_results[exp_name_rope] = run_dir_rope

        # 运行 FoPE (use_fpe=True)
        exp_name_fpe = f"seq{seq_len}_FoPE_d{D_MODEL}"
        run_dir_fpe = run_experiment(seq_len, use_fpe=True)
        experiment_results[exp_name_fpe] = run_dir_fpe

    print("\n所有实验已完成。正在生成图表...")
    plot_results(experiment_results)

if __name__ == "__main__":
    main()