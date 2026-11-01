---
tags:
- siglip
- siglip2
- vision
library_name: open_clip
pipeline_tag: zero-shot-image-classification
license: apache-2.0
datasets:
- webli
---
# Model card for ViT-B-32-SigLIP2-256

## Model Details
A SigLIP 2 Vision-Lanuage model trained on WebLI. 

This model has been converted for use in OpenCLIP from the original JAX checkpoints in [Big Vision](https://github.com/google-research/big_vision).

## Model Details
- **Model Type:** Contrastive Image-Text, Zero-Shot Image Classification.
- **Original:** https://github.com/google-research/big_vision
- **Dataset:** WebLI
- **Papers:**
  - SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features: https://arxiv.org/abs/2502.14786
  - Sigmoid loss for language image pre-training: https://arxiv.org/abs/2303.15343

## Model Usage

```python
import torch
import torch.nn.functional as F
from urllib.request import urlopen
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch >= 2.31.0, timm >= 1.0.15

model, preprocess = create_model_from_pretrained('hf-hub:timm/ViT-B-32-SigLIP2-256')
tokenizer = get_tokenizer('hf-hub:timm/ViT-B-32-SigLIP2-256')

image = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))
image = preprocess(image).unsqueeze(0)

labels_list = ["a dog", "a cat", "a donut", "a beignet"]
text = tokenizer(labels_list, context_length=model.context_length)

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image, normalize=True)
    text_features = model.encode_text(text, normalize=True)
    text_probs = torch.sigmoid(image_features @ text_features.T * model.logit_scale.exp() + model.logit_bias)

zipped_list = list(zip(labels_list, [100 * round(p.item(), 3) for p in text_probs[0]]))
print("Label probabilities: ", zipped_list)
```

## Citation
```bibtex
@article{tschannen2025siglip,
  title={SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features},
  author={Tschannen, Michael and Gritsenko, Alexey and Wang, Xiao and Naeem, Muhammad Ferjad and Alabdulmohsin, Ibrahim and Parthasarathy, Nikhil and Evans, Talfan and Beyer, Lucas and Xia, Ye and Mustafa, Basil and H'enaff, Olivier and Harmsen, Jeremiah and Steiner, Andreas and Zhai, Xiaohua},
  year={2025},
  journal={arXiv preprint arXiv:2502.14786}
}        
```
```bibtex
@article{zhai2023sigmoid,
  title={Sigmoid loss for language image pre-training},
  author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
  journal={arXiv preprint arXiv:2303.15343},
  year={2023}
}
```
```bibtex
@misc{big_vision,
  author = {Beyer, Lucas and Zhai, Xiaohua and Kolesnikov, Alexander},
  title = {Big Vision},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/google-research/big_vision}}
}
```