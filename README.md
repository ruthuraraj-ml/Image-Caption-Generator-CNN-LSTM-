# 🖼️ Image Caption Generator (CNN–LSTM) — Generative AI Project

This project implements a **Generative AI–based Image Caption Generator** using a classic **CNN–LSTM encoder–decoder architecture**.  
The model extracts high-level image features using **InceptionV3** and generates natural-language captions using an **LSTM decoder**, demonstrating cross-modal generative capability.

The project aligns with foundational multimodal AI systems such as Show-and-Tell, BLIP, and Vision–Language Transformers.

---

## 📌 Project Objective
The objective is to automatically generate human-like text descriptions for images by learning the relationship between:

- **Visual domain** → image features  
- **Language domain** → textual captions  

This is achieved using a CNN encoder and an LSTM decoder trained on a dataset of image–caption pairs.

---

## 📁 Dataset Details

- **Dataset:** Open Images Captions (Micro) — Hugging Face  
- **Samples:** ~4,900 images  
- **Annotations:** One caption per image  
- **Fields:**  
  - `image` — RGB input  
  - `text` — human-written caption  

### Why this dataset?
- Lightweight and diverse  
- Ideal for training image captioning models on Colab  
- Contains real-world scenes (people, objects, landscapes, indoor/outdoor)

---

## 🔍 Data Preprocessing

### **Text Preprocessing**
- Lowercasing  
- Removing punctuation and digits  
- Tokenization (top 5,000 words)  
- Sequence padding (max length = 40)  
- Start `<start>` and end `<end>` markers added  

### **Image Preprocessing**
- Resize to (299 × 299)  
- Pass through **InceptionV3** (pretrained on ImageNet)  
- Extract **2048-dimensional** feature embeddings  

These steps ensure aligned visual and textual representations.

---

## 🧱 Model Architecture — Encoder–Decoder (CNN–LSTM)

### **Encoder — InceptionV3**
- Pretrained CNN  
- Extracts semantic image features  
- Output: 2048-dimensional vector  

### **Decoder — LSTM Network**
- Embedding layer (256 units)  
- LSTM (256 units)  
- Dense Softmax output layer  
- Trained to predict the **next word** in sequence  

### Fusion
Image embedding + caption embedding → concatenated → fed into LSTM.

---

## ⚙️ Training Configuration

| Parameter | Value |
|----------|--------|
| Loss | Categorical Cross-Entropy |
| Optimizer | Adam (LR = 0.001 → 0.0001 during fine-tuning) |
| Batch Size | 32 |
| Epochs | 45 |
| Vocabulary Size | 5,000 |
| Embedding Dim | 256 |
| LSTM Units | 256 |
| Feature Vector Size | 2048 |
| Regularization | Dropout (0.5) |

### Training Behavior
- Loss reduced from **4.8 → 2.24**  
- No overfitting (dropout + moderate architecture size)  
- Smooth convergence  

---

## ✨ Decoding Strategy (Caption Generation)

### Used:
- **Greedy decoding (baseline)**
- **Top-K sampling (k = 3–5)**
- **Temperature scaling (τ = 0.6)**
- **Early stopping at `<end>`**

### Result:
- Much more fluent, natural, non-repetitive captions  
- Reduced loops (“end end end …”)  

---

## 📊 Evaluation Results

Evaluation performed using BLEU scores (standard in machine translation & captioning):

| Metric | Score |
|--------|--------|
| **BLEU-1** | **0.294** |
| **BLEU-2** | **0.176** |
| Training Loss | **2.24** |

### Interpretation
- BLEU-1 ≈ 0.29 → Good word-level match  
- BLEU-2 ≈ 0.18 → Moderate phrase-level match  
- Strong alignment between image features and textual semantics  
- Good baseline performance for a CNN–LSTM model trained on a small dataset

---

## 🖼️ Qualitative Results
Generated captions show:

- Grammatically fluent sentences  
- Correct recognition of **people**, **objects**, **scenes**, **actions**  
- Occasional mistakes due to limited dataset  
- Reduced repetition due to improved decoding  

---

## 🧠 Strengths

- Strong cross-modal alignment (image → language)  
- Fluent, human-like sentence generation  
- Good performance on small dataset  
- Efficient: trains in ~2.5 hours on Google Colab (T4 GPU)  
- Forms a foundation for modern multimodal GenAI systems  

---

## ⚠️ Limitations

- Some semantic drift (“man holding camera” when none exists)  
- Limited vocabulary (5k words)  
- Small dataset restricts generalization  
- Lacks attention mechanism → struggles with fine-grained details  

---

## 🚀 Future Improvements

- Add **Bahdanau or Luong Attention**  
- Replace LSTM with a **Transformer decoder (GPT-2, T5)**  
- Train on MS-COCO or Flickr30k for higher BLEU scores  
- Add Grad-CAM visualization for explainability  
- Enable multilingual caption generation  

---

## 📦 Summary Table

| Category | Result |
|---------|--------|
| Architecture | CNN–LSTM (Encoder–Decoder) |
| Encoder | InceptionV3 (ImageNet) |
| Decoder | LSTM (256 units) |
| Dataset | 4,900 images |
| Final Loss | 2.24 |
| BLEU-1 | 0.294 |
| BLEU-2 | 0.176 |
| Decoding | Top-K + Temperature |
| Strength | Fluent captions, cross-modal generation |
| Limitation | Dataset too small |
| Application | Assistive tech, VQA, indexing, GenAI research |

---

## 🧱 Tech Stack

- Python  
- TensorFlow / Keras  
- NumPy, Pandas  
- Matplotlib  
- Hugging Face Datasets  
- NLTK  
- Google Colab (GPU)  
