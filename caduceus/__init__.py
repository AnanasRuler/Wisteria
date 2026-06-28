"""Hugging Face config, model, and tokenizer for Caduceus.

"""
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer, AutoModel
from .configuration_caduceus import CaduceusConfig
from .modeling_caduceus import Caduceus, CaduceusForMaskedLM, CaduceusForSequenceClassification
from .tokenization_caduceus import CaduceusTokenizer


# 注册配置和模型
AutoConfig.register("caduceus", CaduceusConfig)
AutoModel.register(CaduceusConfig, CaduceusForMaskedLM)

# 注册 Tokenizer（如果是自定义的）
AutoTokenizer.register(CaduceusConfig, CaduceusTokenizer)