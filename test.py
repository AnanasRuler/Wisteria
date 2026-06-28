import torch
from transformers import AutoConfig, AutoModelForMaskedLM, AutoModel
from hydra.utils import instantiate
from omegaconf import OmegaConf
from wisteria.modeling_wisteria import WisteriaForMaskedLM
from wisteria.configuration_wisteria import WisteriaConfig
from wisteria.tokenization_wisteria import WisteriaTokenizer
import wisteria
import json

# 修改为 GatedMambav1_final 模型的路径
model_path = "/data2/DNAINSP/outputs/MHA_mamba_1_3_MSC-0-nomlp-1024_d_model-256_n_layer-4_lr-8e-3_rcps-false"

# 加载配置文件
config = OmegaConf.load(f"{model_path}/config.json")
model_config = OmegaConf.load(f"{model_path}/model_config.json")

# 从 config.json 中获取 dataset.max_length
max_length = config.dataset.max_length
print(f"从配置文件中读取到的最大长度: {max_length}")

# 处理模型配置
hf_config = OmegaConf.to_container(model_config["config"], resolve=True)
hf_config.pop("_target_", None)
hf_config = WisteriaConfig(**hf_config)

# 保存为 HF 格式，使用新的模型名称
output_path = "./hf_model/MHA_mamba_1_3_MSC-0-nomlp"
hf_config.save_pretrained(output_path) 

# 使用从配置文件中读取的最大长度创建 tokenizer
tokenizer = WisteriaTokenizer(max_length)
tokenizer.save_pretrained(output_path)

print(f"配置类型: {type(hf_config)}")
print(f"Tokenizer 最大长度设置为: {max_length}")
model = WisteriaForMaskedLM(hf_config)

# 加载 PL 的 .ckpt 文件
ckpt = torch.load(f"{model_path}/checkpoints/last.ckpt")
pl_state_dict = ckpt["state_dict"]

# 去除 Lightning 自动添加的前缀（如 "model."）
hf_state_dict = {k.replace("model.", ""): v for k, v in pl_state_dict.items()}

# 加载权重
model.load_state_dict(hf_state_dict, strict=False)

# 保存完整模型
model.save_pretrained(output_path)

print(f"模型已成功转换并保存到: {output_path}")

# 验证转换结果
model1 = AutoModel.from_pretrained(output_path, trust_remote_code=True)
print(f"模型验证成功，类型: {type(model1)}")