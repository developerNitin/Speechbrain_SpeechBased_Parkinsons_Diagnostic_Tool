# Parkinson's Disease Detection: Audio Model Analysis

This repository presents a comparative analysis of various deep learning architectures—ranging from traditional X-vectors to state-of-the-art Transformers—applied to the detection of Parkinson's Disease through speech analysis.

## 📊 Summary of Test Accuracies

| Model | Data Scale | Training Type | Test Accuracy | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **ECAPA-TDNN (Transfer)** | 50% | Fine-tuned (Frozen) | **100.0%** | **Best overall result** |
| **X-vector** | 100% | From Scratch | 99.1% | Best non-transfer result |
| **ECAPA-TDNN** | 50% | From Scratch | 98.8% | Best 'from scratch' on 50% data |
| **HuBERT** | 50% | Fine-tuned | 98.2% | Good result, minor overfitting |
| **X-vector** | 50% | From Scratch | 97.6% | Performance drop vs 100% data |
| **Wav2Vec2** | 50% | Fine-tuned | 95.8% | Significant overfitting signs |
| **X-vector (Augmented)** | 50% | From Scratch | 50.6% | **Augmentation strategy failed** |

---

## 🔍 Key Findings & Narrative Analysis

### 🏆 1. The Power of Transfer Learning
The clear winner was **Transfer Learning using the pre-trained ECAPA-TDNN model**. By freezing the backbone and fine-tuning only the final classifier on just 50% of the data, we achieved a perfect **100.0% test accuracy**. This demonstrates that leveraging strong, pre-existing acoustic features is more effective than training from scratch, even with less data.

### 🏗️ 2. Architecture Comparison (Scratch Training)
When models were trained without pre-existing weights on the 50% dataset:
* **ECAPA-TDNN** (98.8%) significantly outperformed the **X-vector** configuration (97.6%).
* This suggests the ECAPA architecture is more robust and efficient at extracting diagnostic biomarkers for Parkinson's given limited data.

### 🧪 3. Fine-tuning Large Scale Models
Fine-tuning **HuBERT** and **Wav2Vec2** (Large pre-trained speech models) yielded mixed results:
* **HuBERT** showed good generalization with 98.2% accuracy.
* **Wav2Vec2** performed the worst among successful models (95.8%) due to stronger overfitting.
* **Takeaway:** Large models are powerful but highly susceptible to overfitting on smaller medical datasets; meticulous fine-tuning is crucial.

### 📉 4. Impact of Data Size and Augmentation
* **Data Scale:** Training the X-vector model on 100% of the data (99.1%) was superior to 50% data (97.6%), confirming that data volume remains a critical factor for scratch-trained models.
* **Augmentation Failure:** The augmentation strategy (noise + speed perturbation) on the X-vector model caused performance to plummet to **50.6%** (near random). This highlights that poorly tuned augmentation can destroy the subtle acoustic features necessary for medical classification.

---

## 💡 Conclusions
1. **Transfer Learning is King:** Pre-trained ECAPA-TDNN models are highly recommended for this task, even when data is reduced.
2. **Architecture Matters:** ECAPA-TDNN shows a clear advantage over X-vectors when training from scratch on limited samples.
3. **Handle with Care:** Large models (HuBERT/Wav2Vec2) and data augmentation strategies require careful tuning to avoid overfitting or signal degradation in the context of Parkinson's detection.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Signal Processing:** SpeechBrain / Torchaudio
* **Models:** ECAPA-TDNN, X-vector, HuBERT, Wav2Vec2
* **Environment:** Jupyter Notebook (`.ipynb`)
