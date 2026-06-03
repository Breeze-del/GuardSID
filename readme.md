
# Code Release

This repository contains the official implementation of the paper *“GuardSID: Exposure-Guarded Semantic ID for Generative Next POI Recommendation.”*

The following components are already available:

---

### **PGRQ Module**

The core implementation of the Prefix-Guarded Residual Quantization (PGRQ) is provided.

* using `code/train_pgrq.py` to train a custom codebook.
* After training, use `codebook.py` to generate the mapping from discrete token IDs to semantic IDs.

---

### **Sample Dataset**

A sample dataset based on the **NYC** check-in data is included for demonstration and evaluation.

We directly adopt the preprocessed dataset from **LLM4POI**:  
https://github.com/neolifer/LLM4POI/tree/main

To reproduce our data pipeline, please follow the steps below:

1. **Download the dataset**  
   Download the dataset from the LLM4POI repository.

2. **Generate POI metadata**  
   Run the following notebook to obtain `poi_info.csv`: `V1/datasets/dataprocess.ipynb`

3. **Generate Semantic IDs (SIDs)**  
Execute the codebook quantization module to obtain SID representations for POIs.

4. **Prepare LLM training data**  
Run the following notebook to construct the final dataset for LLM fine-tuning: `V1/datasets/llm_dataprocess.ipynb`

---

### **Model Fine-tuning**

Model fine-tuning is conducted using the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) framework.
You can reproduce the fine-tuning and evaluation workflows using our processed datasets, following the official instructions provided in the LLaMA-Factory repository.
---
