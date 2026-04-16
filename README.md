# Remaining Driving Range Prediction for Electric Vehicles with Multimodal Model and Transfer Learning
This repository provides the implementation for the paper:
**"Remaining Driving Range Prediction for Electric Vehicles with Multimodal Model and Transfer Learning"**

## 📂 Data
The `data/` folder contains a **sample dataset** to ensure that reviewers and users can successfully run the code.
The **full dataset will be made publicly available upon publication** of the paper at: 
https://doi.org/10.5281/zenodo.19588333
The `model/` folder contains **models trained on the full dataset**, which can be used to directly evaluate the results.

## ⚙️ Environment Setup
Install the required dependencies using:
```bash
pip install -r requirements.txt
```

## 🚀 Usage
To reproduce the main results, run:
```bash
rdr_prediction.ipynb
```
Make sure that `utils.py` is located in the **same directory** as `rdr_prediction.ipynb`.
This notebook includes:
* End-to-end model training
* Transfer learning experiments
* Model evaluation on the test set

## 📌 Notes
* The provided sample data is only for demonstration and reproducibility.
* For full-scale experiments, please use the complete dataset (to be released).
* See `MODEL_CARD.md` for model details.
