---
license: apache-2.0
tags:
- vision
- image-text
- contrastive-learning
- zero-shot
- feature-extraction
- arxiv:2604.12012
library_name: transformers
pipeline_tag: zero-shot-image-classification
---

# TIPSv2 — L/14

TIPSv2 (Text-Image Pre-training with Spatial awareness) is a family of contrastive vision-language models that produce spatially rich image features aligned with text embeddings. This is the Large variant with 303M vision params and 184M text params. Try the code snippets below or check out the [GitHub repo](https://github.com/google-deepmind/tips) for more use cases and visualizations, including zero-shot segmentation.

| Variant | Vision params | Text params | Embed dim | DPT Heads |
|---------|--------------|-------------|-----------|-----------|
| [B/14](https://huggingface.co/google/tipsv2-b14) | 86M | 110M | 768 | [B/14-dpt](https://huggingface.co/google/tipsv2-b14-dpt) |
| [L/14](https://huggingface.co/google/tipsv2-l14) | 303M | 184M | 1024 | [L/14-dpt](https://huggingface.co/google/tipsv2-l14-dpt) |
| [SO400m/14](https://huggingface.co/google/tipsv2-so400m14) | 412M | 448M | 1152 | [SO400m/14-dpt](https://huggingface.co/google/tipsv2-so400m14-dpt) |
| [g/14](https://huggingface.co/google/tipsv2-g14) | 1.1B | 389M | 1536 | [g/14-dpt](https://huggingface.co/google/tipsv2-g14-dpt) |

## Usage

```bash
pip install transformers torch torchvision sentencepiece scikit-learn
```

### Load the model

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("google/tipsv2-l14", trust_remote_code=True)
model.eval()
```

### Encode images

Images should be tensors in `[0, 1]` range (just `ToTensor()`, no ImageNet normalization).

```python
from torchvision import transforms
from PIL import Image
import requests

transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])

url = "https://huggingface.co/spaces/google/TIPSv2/resolve/main/examples/zeroseg/pascal_context_00049_image.png"
image = Image.open(requests.get(url, stream=True).raw)
pixel_values = transform(image).unsqueeze(0)
out = model.encode_image(pixel_values)

print(out.cls_token.shape)     # (1, 1, 1024) — global image embedding
print(out.patch_tokens.shape)  # (1, 1024, 1024) — per-patch spatial features
```

### Encode text

```python
text_emb = model.encode_text(["a photo of a bus", "a photo of a dog"])
print(text_emb.shape)  # (2, 1024) — one embedding per query
```

### Zero-shot classification

```python
import torch.nn.functional as F

classes = ["bus", "car", "dog", "cat"]
cls = F.normalize(out.cls_token[:, 0, :], dim=-1)
text_emb = F.normalize(model.encode_text(classes), dim=-1)
similarity = cls @ text_emb.T
print(classes[similarity.argmax()])  # bus — predicted class
```

### Visualize spatial features

```python
import numpy as np
from sklearn.decomposition import PCA

spatial = out.patch_tokens.reshape(1, 32, 32, 1024)
feat = spatial[0].detach().cpu().numpy().reshape(-1, 1024)
rgb = PCA(n_components=3, whiten=True).fit_transform(feat).reshape(32, 32, 3)
rgb = 1 / (1 + np.exp(-2.0 * rgb))  # sigmoid for [0, 1] range with good contrast
print(rgb.shape)  # (32, 32, 3) — PCA of patch features as RGB
```

### GPU inference

```python
model = model.cuda()
out = model.encode_image(pixel_values.cuda())
text_emb = model.encode_text(["a city"])
```

## Model details

- **Architecture**: ViT vision encoder (24 layers) + Transformer text encoder (12 layers)
- **Image preprocessing**: resize to any resolution, convert to `[0, 1]` (no ImageNet normalization)
- **Text preprocessing**: SentencePiece tokenizer, lowercased, max 64 tokens
- **Patch size**: 14x14 pixels

## License

Apache 2.0

## Citation

```bibtex
@inproceedings{cao2026tipsv2,
  title     = {{TIPSv2: Advancing Vision-Language Pretraining with Enhanced Patch-Text Alignment}},
  author    = {Cao, Bingyi and Chen, Koert and Maninis, Kevis-Kokitsi and Chen, Kaifeng and Karpur, Arjun and Xia, Ye and Dua, Sahil and Dabral, Tanmaya and Han, Guangxing and Han, Bohyung and Ainslie, Joshua and Bewley, Alex and Jacob, Mithun and Wagner, Rene and Ramos, Washington and Choromanski, Krzysztof and Seyedhosseini, Mojtaba and Zhou, Howard and Araujo, Andre},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2604.12012}
}
```
