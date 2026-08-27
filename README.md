# EFE-VLA: Epistemic Uncertainty-Aware Vision-Language-Action Models

An uncertainty-aware VLA fine-tuning method that learns **robot actions and input-dependent uncertainty** together. A heteroscedastic prediction head estimates when action predictions are less reliable, particularly in difficult or visually ambiguous manipulation states.

## Overview

```text
Visual + Language + Robot State
              ↓
          VLA Model
          ↙       ↘
      Action    Uncertainty
```

## Key Features

- Uncertainty-aware VLA fine-tuning
- Input-dependent action uncertainty
- Evaluation on **BridgeData V2**
- Analysis of uncertainty under **occluded vs. open** observations
- Designed for detecting difficult and potentially unreliable states

## Installation

```bash
git clone https://github.com/dronry/EFE-VLA-Epistemic-Uncertainty-Aware-Vision-Language-Action-Models.git
cd EFE-VLA-Epistemic-Uncertainty-Aware-Vision-Language-Action-Models
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

For evaluation:

```bash
python Evaluation.py
```

## Research Goal

> **Can a VLA model learn to recognize when its own action predictions are unreliable?**

