---
pipeline_tag: zero-shot-image-classification
language: pol
license: apache-2.0
tags:
  - trimmed
library_name: sentence-transformers
base_model: google/siglip2-base-patch16-512
base_model_relation: quantized
datasets:
  - lbourdois/fineweb-2-trimming
---

# siglip2-base-patch16-512-pol-32768
This model is a **45.62%** smaller version of [google/siglip2-base-patch16-512](https://huggingface.co/google/siglip2-base-patch16-512) optimized for **Polish** language via vocabulary size reduction using the [trimming](https://huggingface.co/blog/lbourdois/introduction-to-trimming) method.  
This trimmed model should perform similarly to the original model with only 32,768 tokens and a much smaller memory footprint. However, it may not perform well for other languages as tokens not commonly used in the selected languages were removed from the vocabulary.

## Model Statistics
| Metric | Original | Trimmed | Reduction |
|--------|----------|---------|-----------|
| **Vocabulary size** | 256,000 tokens | 32,768 tokens | **87.20%** |
| **Model size** | 375,823,874 params | 204,381,698  params | **45.62%** |

![image](https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Trimming/siglip2-base-patch16-512-32768.png)

## Mining Dataset Statistics
- **Number of texts used for mining**: 200,000 texts  
- **Dataset**: [lbourdois/fineweb-2-trimming](https://huggingface.co/datasets/lbourdois/fineweb-2-trimming)

## Usage

#### Transformers (zero-shot image classification)
```python
from transformers import pipeline

# load pipeline
image_classifier = pipeline(model="alphaedge-ai/siglip2-base-patch16-512-pol-32768", task="zero-shot-image-classification")

# load image and candidate labels
image = "http://images.cocodataset.org/val2017/000000039769.jpg"
candidate_labels = ["Potential label 1 in Polish", "Potential label 2 in Polish", "Potential label 3 in Polish", "Potential label 4 in Polish"]

# run inference
outputs = image_classifier(image, candidate_labels)
print(outputs)
```

#### Sentence-transformers (texts-images similarity)
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("alphaedge-ai/siglip2-base-patch16-512-pol-32768")

images = [
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
    "https://huggingface.co/datasets/huggingface/cats-image/resolve/main/cats_image.jpeg"
]
texts = ["Text 1 in Polish", "Text 2 in Polish", "Text 3 in Polish", "Text 4 in Polish"]

image_embeddings = model.encode(images)
text_embeddings = model.encode(texts)
print(image_embeddings.shape, text_embeddings.shape)

similarities = model.similarity(image_embeddings, text_embeddings)
print(similarities)
```

## Citations

#### SigLIP 2
```
@misc{tschannen2025siglip2multilingualvisionlanguage,
      title={SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features}, 
      author={Michael Tschannen and Alexey Gritsenko and Xiao Wang and Muhammad Ferjad Naeem and Ibrahim Alabdulmohsin and Nikhil Parthasarathy and Talfan Evans and Lucas Beyer and Ye Xia and Basil Mustafa and Olivier Hénaff and Jeremiah Harmsen and Andreas Steiner and Xiaohua Zhai},
      year={2025},
      eprint={2502.14786},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.14786}, 
}
```

#### Trimming blog post
```
@misc{hf_blogpost_trimming,
      title={Introduction to Trimming}, 
      author={Loïck BOURDOIS and Tom AARSEN and Bram VANROY and Christopher AKIKI and Woojun JUNG and Manuel ROMERO and Prithiv SAKTHI},
      year={2026},
      url={https://huggingface.co/blog/lbourdois/introduction-to-trimming}, 
}
```