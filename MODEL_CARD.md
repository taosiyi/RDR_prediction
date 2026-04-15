---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

# Model Card for {{ model_id | default("Model ID", true) }}

<!-- Provide a quick summary of what the model is/does. -->

BatteryMFT is an end-to-end multimodal deep learning framework that jointly performs battery dynamic reconstruction, state estimation, and remaining driving range (RDR) prediction.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

BatteryMFT is an end-to-end battery dynamic multimodal model that is sensitive to internal electrochemical states. The model is capable of accurately reconstructing historical signals, monitoring the battery’s current state, and enabling accurate and practical RDR prediction under dynamic discharging conditions during real-world driving. 

- **Developed by:** Siyi Tao
- **Model type:** Deep Learning
- **Language(s) (NLP):** N/A
- **License:** MIT

### Model Sources

<!-- Provide the basic links for the model. -->

- **Repository:** https://github.com/taosiyi/RDR_prediction.git
- **Paper:** Remaining Driving Range Prediction for Electric Vehicles with Multimodal Model and Transfer Learning

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

This model is intended for:
- Historical signals reconstruction for battery systems (e.g. Voltage reconstruction)
- Current state monitoring for battery systems (e.g. SOC estimation)
- Battery-related dynamic tasks (e.g. RDR prediction)

### Downstream Use

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

The model can be fine-tuned or adapted to:
- Different EV platforms
- Different battery chemistries (e.g., LFP, NMC)
- Fleet-level energy management systems

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

- Real-time safety-critical decision-making without further validation
- Deployment without domain adaptation on unseen vehicle types
- Use outside battery/electric vehicle systems

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

The model is trained on real-world EV fleet data collected from multiple vehicle platforms. Performance may degrade under:
- Unseen vehicle types or battery chemistries
- Strong distribution shifts in driving behavior
- Extreme environmental conditions not present in training data

The model is not guaranteed to generalize without fine-tuning to new domains.

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users should:
- Validate performance before deployment in new environments
- Apply transfer learning for domain adaptation
- Ensure training data coverage matches target scenarios

## How to Get Started with the Model
To reproduce the main results, run: 
```bash
rdr_prediction.ipynb
```

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

Raw BMS signals are first cleaned by removing abnormal values and ensuring temporal consistency. The data are then segmented into continuous driving/charging/parking sessions. Input data are organized into synchronized multimodal sequences using sliding-window segmentation. 
Training data is in `data/train`.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

- **Training regime:** fp32 (with optional mixed precision fp16 acceleration during training)

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

Evaluation is conducted on held-out real-world EV fleet data, including unseen vehicles from all three platforms. The test set covers heterogeneous driving conditions, SOC ranges, and battery aging states.
Test data is in `data/test`.

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

Model performance is evaluated across:
- Vehicle platform differences with battery chemistry varies (NMC vs LFP)
- Driving behavior (low/medium/high speed regimes)

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

### Results

| Car | Volt RMSE (V) | Volt MAE (V) | Volt MAPE (%) | SOC RMSE (%) | SOC MAE (%) | SOC MAPE (%) | DR5 RMSE (km) | DR5 MAE (km) | DR5 MAPE (%) |
|-----|------------------|---------|----------|---------------|---------|----------|---------------|-----------|----------|
| Test set | 0.964 | 0.764 | 0.189 | 0.946 | 0.660 | 1.206 | 2.484 | 1.814 | 14.064 |
| P208 | 0.932 | 0.749 | 0.185 | 0.932 | 0.652 | 1.197 | 2.601 | 1.910 | 13.673 |
| P216 | 1.192 | 0.921 | 0.229 | 1.328 | 0.958 | 1.719 | 2.842 | 2.165 | 20.258 |
| P218 | 0.871 | 0.695 | 0.172 | 0.692 | 0.514 | 0.939 | 1.562 | 1.178 | 8.969 |
| P404 | 0.941 | 0.755 | 0.187 | 0.905 | 0.635 | 1.171 | 2.701 | 1.998 | 15.089 |

#### Summary

These results highlight the effectiveness of integrating large-scale multimodal learning with transfer learning to capture latent electrochemical and operational characteristics, enabling accurate and generalizable safety- and health-related predictions under real-world conditions.

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** NVIDIA GeForce RTX 3060 Ti GPU
- **Hours used:** ~ 6 hours
- **Cloud Provider:** No
- **Compute Region:** No
- **Carbon Emitted:** 0.65 kg CO2eq

## Model Card Contact

taosiyi406ttt@gmail.com
