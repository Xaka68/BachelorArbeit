# Active Deep Learning from Data Lakes

**B.Sc. thesis in Informatics, Technical University of Munich (submitted 1 Sep 2025)**
Author: Xavier Igor Kamsu Nzodoum · Supervisor: Herbert Woisetschläger · Advisor: Prof. Dr. Viktor Leis

📄 Full thesis: [`thesis/`](thesis/)

## Summary

Labeling data is the main cost of training deep learning models. **Deep active learning (DAL)** cuts that cost by letting the model choose which unlabeled samples to annotate next. Many strategies exist, but they are usually evaluated on a single domain and only by overall accuracy.

This thesis runs a **cross-domain benchmark of 11 DAL strategies on 3 image and 2 text datasets** under one unified protocol. It measures accuracy, per-class accuracy, selection time and training time. It also introduces a stricter label-efficiency criterion: the minimum number of labels needed to reach 95 % of fully supervised accuracy **overall and for every single class**.

## Key findings

- **Active learning always beats random sampling, but no strategy wins everywhere.** The best choice depends on dataset properties: class separability, redundancy, semantic variation and label noise.
- **BALD is the most label-efficient on structured images.** On EuroSAT it reaches the 95 % target with **44 % of the labels, versus 62.5 % for random sampling**.
- **Representativeness matters for heterogeneous text.** On Multi-Domain Sentiment, FASS reaches the target with **33 % of the labels, versus 52 % for random sampling**, while BALD needs 71 %.
- **Overall accuracy can hide weak classes.** On EuroSAT, BADGE reaches 0.91 overall accuracy early but needs over 12,000 labels before every class meets its target.
- **GLISTER depends on stable gradients.** It has the best EuroSAT accuracy (0.92 vs 0.83 for random after 12 rounds) but falls behind on the noisy FER2013 dataset.
- **Adversarial strategies are too slow in practice.** DeepFool's selection time on EuroSAT is about **100× that of Margin sampling** (61,207 s vs 575 s).

## Strategies compared

| Family | Strategies |
|---|---|
| Uncertainty-based | Entropy, Margin, Least Confidence, BALD |
| Diversity-based | CoreSet |
| Hybrid | BADGE |
| Representativeness-based | FASS |
| Optimization-based | GLISTER |
| Adversarial | DeepFool, BIM |
| Baseline | Random sampling |

## Experimental setup

| | Image tasks | Text tasks |
|---|---|---|
| Datasets | EuroSAT (27,000 satellite images, 10 classes), Intel Scenes (6 classes), FER2013 (35,887 faces, 7 emotions) | AG News (4 topics), Multi-Domain Sentiment v2.0 (books, binary) |
| Model | ResNet18 | ModernBERT-base (max. 256 tokens) |
| Optimizer | SGD, lr 0.01, batch size 4 | SGD, lr 2e-5, batch size 4 |

- **Protocol:** 500 randomly drawn seed labels (no forced class balance), then rounds of query → label → retrain from scratch, with adaptive early stopping. 12 rounds for image datasets, 8 for text.
- **Metrics:** overall accuracy, per-class accuracy, selection time, training time, and the minimum labels n* to reach 95 % of fully supervised accuracy overall and per class.
- **Framework:** built on [DISTIL](https://github.com/decile-team/distil), into which the five datasets were integrated. Implemented in PyTorch and run on an NVIDIA RTX 8000 GPU.

## Label efficiency at a glance

Labels needed to reach 95 % of fully supervised accuracy, overall and for every class (as % of the training set):

| Dataset | Random | Best strategy |
|---|---|---|
| EuroSAT | 62.5 % | BALD: 44.0 % |
| Multi-Domain Sentiment | 52.2 % | FASS: 33.2 % |
| FER2013 | not reached | Entropy: 47.0 % (only strategy to reach it) |
| Intel Scenes | not reached | BALD, Least Conf., BADGE, BIM: 89.1 % |

AG News was excluded from this analysis because of its runtime.
