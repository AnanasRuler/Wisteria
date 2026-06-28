"""Hugging Face config, model, and tokenizer for Wisteria.

"""
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer, AutoModel
from .configuration_wisteria import WisteriaConfig
from .modeling_wisteria import Wisteria, WisteriaForMaskedLM, WisteriaForSequenceClassification
from .tokenization_wisteria import WisteriaTokenizer


# 注册配置和模型
AutoConfig.register("wisteria", WisteriaConfig)
AutoModel.register(WisteriaConfig, WisteriaForMaskedLM)

# 注册 Tokenizer（如果是自定义的）
AutoTokenizer.register(WisteriaConfig, WisteriaTokenizer)