"""Convert official DINOv2 .pth → HuggingFace format for local loading."""
import torch
import json
from pathlib import Path

PTH = Path("/root/code/vepfs/HUG-for-Recon-Gen/dinov2/dinov2_vitb14_reg4_pretrain.pth")
OUT = Path("/root/code/vepfs/HUG-for-Recon-Gen/dinov2/hf_model")
OUT.mkdir(exist_ok=True)

# ---- config.json ----
config = {
    "architectures": ["Dinov2Model"],
    "model_type": "dinov2",
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "image_size": 224,
    "patch_size": 14,
    "num_channels": 3,
    "qkv_bias": True,
    "layer_norm_eps": 1e-6,
    "num_register_tokens": 4,
    "torch_dtype": "float32",
    "transformers_version": "4.40.0",
}
with open(OUT / "config.json", "w") as f:
    json.dump(config, f, indent=2)

# ---- weight conversion ----
pth = torch.load(PTH, map_location="cpu", weights_only=True)
hf_state = {}

for key, val in pth.items():
    if key == "cls_token":
        hf_state["embeddings.cls_token"] = val
    elif key == "register_tokens":
        hf_state["embeddings.register_tokens"] = val
    elif key == "pos_embed":
        hf_state["embeddings.position_embeddings"] = val
    elif key == "mask_token":
        hf_state["embeddings.mask_token"] = val
    elif key.startswith("patch_embed.proj"):
        hf_state[key.replace("patch_embed.proj", "embeddings.patch_embeddings.projection")] = val
    elif key.startswith("norm."):
        hf_state[key.replace("norm.", "layernorm.")] = val
    elif key.startswith("blocks."):
        parts = key.split(".", 2)
        idx = parts[1]
        rest = parts[2]
        pfx = f"encoder.layer.{idx}."
        if rest.startswith("norm1."):
            hf_state[pfx + "norm1." + rest[6:]] = val
        elif rest.startswith("norm2."):
            hf_state[pfx + "norm2." + rest[6:]] = val
        elif rest == "ls1.gamma":
            hf_state[pfx + "layer_scale1.lambda1"] = val
        elif rest == "ls2.gamma":
            hf_state[pfx + "layer_scale2.lambda1"] = val
        elif rest.startswith("attn.qkv."):
            param = rest[9:]
            q, k, v = val.chunk(3, dim=0)
            hf_state[pfx + f"attention.attention.query.{param}"] = q
            hf_state[pfx + f"attention.attention.key.{param}"] = k
            hf_state[pfx + f"attention.attention.value.{param}"] = v
        elif rest.startswith("attn.proj."):
            hf_state[pfx + "attention.output.dense." + rest[10:]] = val
        elif rest.startswith("mlp.fc1."):
            hf_state[pfx + "mlp.fc1." + rest[8:]] = val
        elif rest.startswith("mlp.fc2."):
            hf_state[pfx + "mlp.fc2." + rest[8:]] = val

torch.save(hf_state, OUT / "pytorch_model.bin")
print(f"Done → {OUT}/pytorch_model.bin  ({len(hf_state)} keys)")
