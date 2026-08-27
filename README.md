# EFE-VLA-Epistemic-Uncertainty-Aware-Vision-Language-Action-Models
built an uncertainty-aware VLA fine-tuning method that learns input-dependent uncertainty alongside robot actions, using a heteroscedastic prediction head to identify difficult or visually ambiguous manipulation states and quantify when the policy's action predictions are less reliable

Visual + Language + Robot State
              ↓
          VLA Model
          ↙       ↘
      Action    Uncertainty

The goal is to identify difficult, ambiguous or potentially unreliable manipulation states, particularly under visual occlusion and distribution shift.

# Key Features
> Uncertainty-aware VLA fine-tuning
> Input-dependent action uncertainty
> Evaluation on BridgeData V2
> Analysis of uncertainty under occluded vs. open observations

# Installation
git clone https://github.com/dronry/EFE-VLA-Epistemic-Uncertainty-Aware-Vision-Language-Action-Models.git
cd EFE-VLA-Epistemic-Uncertainty-Aware-Vision-Language-Action-Models
pip install -r requirements.txt

Run
python main.py

For evaluation:

python Evaluation.py

## Research Goal

Can a VLA model learn to recognize when its own action predictions are unreliable
