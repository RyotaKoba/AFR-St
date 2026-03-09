# custom_checkpoint.py
import torch
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model
from transformers import AutoTokenizer, AutoConfig

@register_model("custom_checkpoint")
class CheckpointCausalLM(HFLM):
    def __init__(self, checkpoint=None, pretrained=None, **kwargs):
        if checkpoint is not None:
            print(f"Loading pruned model from checkpoint: {checkpoint}")
            
            # まず通常のHFLMとして初期化（ダミーモデルで）
            # pretrainedが必要なので、config_pretrainedを使う
            super().__init__(
                pretrained=pretrained,
                backend="causal",
                **kwargs
            )
            
            # checkpointをロード
            pruned_dict = torch.load(checkpoint, map_location='cuda', weights_only=False)
            
            # モデルだけ差し替え
            self._model = pruned_dict.get('model').half().eval()
            
            # Tokenizerも差し替え（もしcheckpointに含まれていれば）
            if 'tokenizer' in pruned_dict:
                self.tokenizer = pruned_dict['tokenizer']
            
            # デバイスに移動
            if not kwargs.get('device_map'):
                self._model = self._model.to(self._device)
            
            print(f"Successfully loaded pruned model from {checkpoint}")
            
        else:
            # 通常のHFLM初期化
            super().__init__(pretrained=pretrained, **kwargs)
    
    @property
    def model(self):
        """モデルへのアクセスを提供"""
        return self._model if hasattr(self, '_model') else super().model