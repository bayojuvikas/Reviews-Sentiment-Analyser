
# 🧠 Sentiment Analysis on Skincare Product Reviews using DistilBERT 💬🧴

![Model](https://img.shields.io/badge/Model-DistilBERT-blue) ![Made with PyTorch](https://img.shields.io/badge/Made%20with-PyTorch-red) ![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

A powerful **NLP project** that leverages 🤖 **DistilBERT** and 🧪 structured metadata to classify customer sentiments from real-world skincare product reviews. This solution is tailored for **e-commerce platforms**, enabling smarter review analytics and better product understanding.

---

## 🚀 Features

✅ Fine-tuned `DistilBERT` on review texts  
✅ Included metadata like `Verified Buyer`, `Review Location`, `Upvotes/Downvotes`, etc.  
✅ Achieved **~80% accuracy** on real-world test data  
✅ Exported predictions to CSV (`output.csv`)  
✅ Clean, modular pipeline with PyTorch + Hugging Face

---

## 📂 Dataset

🔹 **Train File** (`train.csv`) – Labeled customer reviews with text and metadata  
🔹 **Test File** (`test.csv`) – Unlabeled data for prediction  
🔹 **Output File** (`output.csv`) – Predictions with `ID` and `sentiment`

Each entry includes:

```text
ID | Review_Text | Verified_Buyer | Review_Date | Review_Location | Review_Upvotes | Review_Downvotes | Product | Brand | Review_Title | Sentiment
````

---

## 🛠️ Tech Stack

| Tool / Framework              | Role                          |
| ----------------------------- | ----------------------------- |
| 🧠 Transformers (HuggingFace) | Pre-trained DistilBERT model  |
| 🔥 PyTorch + Dataloaders      | Training pipeline             |
| 📊 Pandas & NumPy             | Data preprocessing & analysis |
| 🧪 Sklearn                    | Evaluation & Metrics          |

---

## 📈 Results

| Metric    | Value                |
| --------- | -------------------- |
| Accuracy  | 🟢 80%               |
| Loss      | 🔵 0.058             |
| Inference | ✅ 100% test coverage |

---

## 📁 Project Structure

```bash
.
├── train.csv
├── test.csv
├── output.csv
├── main.py
├── README.md
```

---

## 🔮 How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train the model:
3. Run predictions:

```bash
python main.py
```

4. Check results in:

```bash
output.csv
```

---

## 📌 Sample Prediction Output

```csv
ID,sentiment
3000,Negative
3001,Negative
3002,Positive
...
```

---

## 📄 License

MIT License © 2025 Vikas Bayoju

---

## ⭐️ Show Your Support

If you like this project, give it a ⭐ and feel free to fork 🍴 and contribute!

Let me know if you'd like to attach sample visuals like charts or model architecture diagrams — I can create those too.
