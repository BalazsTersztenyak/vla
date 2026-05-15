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

        print(f"Loading OpenVLA processor from {mid} …")
        self.processor = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
        print(f"Loading OpenVLA model from {mid} (this may take a minute) …")
        self.model = AutoModelForVision2Seq.from_pretrained(
            mid,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()

        print("Warming up JIT / CUDA kernels (first call compiles TorchScript) …")
        _dummy = torch.zeros(1, 3, 224, 224, dtype=torch.bfloat16, device=device)
        _dummy_inputs = self.processor(
            "In: What action should the robot take to warm up?\nOut:",
            Image.fromarray((_dummy[0].permute(1, 2, 0).cpu().float().numpy() * 255).astype("uint8")),
        ).to(device, dtype=torch.bfloat16)
        with torch.no_grad():
            self.model.predict_action(**_dummy_inputs, unnorm_key=self.unnorm_key, do_sample=False)
        print("OpenVLA model ready.")
        self._torch = torch

    def predict(self, image: Image.Image, instruction: str) -> np.ndarray:
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = self.processor(prompt, image).to(self.device, dtype=self._torch.bfloat16)
        with self._torch.no_grad():
            action = self.model.predict_action(
                **inputs, unnorm_key=self.unnorm_key, do_sample=False
            )
        return np.array(action, dtype=np.float32)
