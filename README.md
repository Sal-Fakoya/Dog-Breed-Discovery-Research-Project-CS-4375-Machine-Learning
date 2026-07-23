# Coarse and Fine Grained Dog Breed Discovery

Unsupervised breed level clustering using frozen pretrained CNN and Vision Transformer feature extractors. No breed labels are used during clustering, only for evaluation afterward.

**Authors:** Salamot Fakoya, Isabella Ran — University of Texas at Dallas, CS 4375 Machine Learning

**Live walkthrough:** https://sal-fakoya.github.io/Dog-Breed-Discovery-Research-Project-CS-4375-Machine-Learning/

---

## What this project does

Pretrained image models (ResNet-18, VGG-16, ViT-B/16, ViT-L/16) are used purely as frozen feature extractors, none are fine tuned on dog breeds. Each image is converted into a feature vector, features are standardized and reduced with PCA, and K-Means clusters the results. Breed labels are only used afterward, to check whether the clusters the algorithm found on its own actually correspond to real breeds.

The question being tested: do general purpose pretrained representations already encode enough breed relevant structure to recover fine grained categories with zero supervision, and does that structure hold up under background removal and poor lighting?

## Key results

| Experiment | Headline finding |
|---|---|
| CIFAR-10 (32×32, dropped) | Both models fail. Best silhouette 0.014 (ResNet-18), 0.054 (VGG-16). No breed sub-labels exist, so ARI/NMI can't even be computed. |
| Baseline clustering | ResNet-18: ARI 0.581 (Imagewoof), 0.530 (Oxford Dogs, 90.2% purity). VGG-16 is close behind despite an 8× larger feature space, neither model consistently wins. |
| SAM background removal | Baseline beats both black-fill and white-fill variants in all 18 dataset/model comparisons. Segmentation hurts, it doesn't help. |
| Lighting degradation | Dog clustering ARI drops 52–85% in low light; cats drop only 23–69%. Dog breeds lean more on coat color, which fades first. |
| Architecture comparison | ViT-B/16 reaches ARI 0.796 on Oxford Dogs, a 50% improvement over ResNet-18 and 62% over VGG-16. Self attention outperforms global average pooling by a wide margin. |

Full methodology, all four experiments, and the complete discussion of limitations are in the [live presentation](https://sal-fakoya.github.io/Dog-Breed-Discovery-Research-Project-CS-4375-Machine-Learning/) and the project report.

## Datasets

| Dataset | Images | Breeds | Resolution | Role |
|---|---|---|---|---|
| CIFAR-10 Dogs | 6,000 | 1 (no sub-labels) | 32×32 | Sanity check, abandoned after resolution failure |
| Imagewoof | 9,025 | 10 | 224×224 | Primary baseline benchmark |
| Oxford Dogs | 4,978 | 25 | Variable (Flickr) | Hardest benchmark, used for every robustness experiment |
| Oxford Cats | 2,371 | 12 | Variable (Flickr) | Tests whether findings generalize beyond dogs |

## Pipeline

```
raw image → frozen pretrained feature extractor → StandardScaler → PCA → K-Means → evaluate against ground truth
```

- **K-Means:** k-means++ initialization, 10 restarts, 300 max iterations, K selected per dataset by maximizing silhouette score on a training subsample
- **Evaluation metrics:** Adjusted Rand Index (primary), Normalized Mutual Information, cluster purity, silhouette score (used only for model selection, not as a quality metric on its own)

| Model | Feature dimension | Parameters |
|---|---|---|
| ResNet-18 | 512 | 11.7M |
| VGG-16 | 4,096 | 134M |
| ViT-B/16 | 768 | 86M |
| ViT-L/16 | 1,024 | 307M |

## Repository contents

| File | Description |
|---|---|
| `CS_4375_ResNet_Cifar.ipynb` | ResNet-18 baseline on CIFAR-10 dogs (failure case) |
| `CS_4375_ResNet-breed-pipeline.ipynb` | ResNet-18 pipeline across Imagewoof, Oxford Dogs, Oxford Cats, including SAM and lighting experiments |
| `vgg16_cifar_model.ipynb` | VGG-16 baseline on CIFAR-10 dogs |
| `vgg16_breed_pipeline_.ipynb` | VGG-16 pipeline across the three breed datasets |
| `Midterm_report_CS4375.pdf` | Midterm project report |
| `4375_project-slides-updated.pptx` | Presentation slides |
| `image_classifier.html` | Interactive results walkthrough (the live link above) |
| `img/` | Figures referenced by `image_classifier.html` |

## Limitations

- K-Means assumes roughly spherical, evenly sized clusters, an assumption visibly violated by the oversized clusters observed in several experiments
- PCA used a fixed component count rather than a variance threshold; VGG-16 retained only 55.13% of variance at 100 components
- No built-in robustness to non-standard lighting was added to the pipeline itself, the lighting experiment only measures the failure
- All models were used frozen, with no task specific fine tuning

See the full discussion section in the live walkthrough for details and proposed follow-up work (HDBSCAN, UMAP, brightness augmentation, and partial fine tuning).

## Acknowledgment

Claude was used to accelerate implementation and debugging across the experiment pipeline. All methodology decisions, dataset selection, experiment design, and interpretation of results were made by the authors.
