"""OpenVLA-7B policy wrapper. Requires the 'vla' optional dependencies."""
from __future__ import annotations

import numpy as np
from PIL import Image


class OpenVLAPolicy:
    MODEL_ID = "openvla/openvla-7b"

    def __init__(
        self,
        unnorm_key: str = "bridge_orig",
        device: str = "cuda:0",
        model_id: str | None = None,
    ) -> None:
        # Defer heavy imports so the module is importable without torch/transformers
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.unnorm_key = unnorm_key
        self.device = device
        mid = model_id or self.MODEL_ID

        self.processor = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            mid,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()
        self._torch = torch

    def predict(self, image: Image.Image, instruction: str) -> np.ndarray:
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = self.processor(prompt, image).to(self.device, dtype=self._torch.bfloat16)
        with self._torch.no_grad():
            action = self.model.predict_action(
                **inputs, unnorm_key=self.unnorm_key, do_sample=False
            )
        return np.array(action, dtype=np.float32)
