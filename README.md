# 🎓 EduPredict AI — Student Performance Predictor

> Deep Learning–powered student risk classification app built with Streamlit

---

## 📁 Project Structure

```
streamlit_app/
├── app.py                        # Main Streamlit UI application
├── train_model.py                # Script to train & save the model
├── STUDENT_PERFORMANCE_EXPANDED.csv   # Expanded dataset (2,395 records)
├── requirements.txt              # Python dependencies
├── .streamlit/
│   └── config.toml               # Theme & server configuration
└── model/                        # Auto-created after training
    ├── student_ann_model.pkl     # Trained MLPClassifier
    └── scaler.pkl                # Fitted StandardScaler
```

---

## 🚀 Quick Start (Local)

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Place the dataset
Make sure `STUDENT_PERFORMANCE_EXPANDED.csv` is in the same folder as `train_model.py`.

### Step 3 — Train the model
```bash
python train_model.py
```
This trains the ANN for **100 full epochs** and saves:
- `model/student_ann_model.pkl`
- `model/scaler.pkl`

Expected output: **~91–93% accuracy**

### Step 4 — Launch the app
```bash
streamlit run app.py
```
Open your browser at **http://localhost:8501**

---

## ☁️ Deploy to Streamlit Cloud (Free)

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "EduPredict AI - initial commit"
git remote add origin https://github.com/YOUR_USERNAME/edupredict-ai.git
git push -u origin main
```

### Step 2 — Upload pre-trained model files
Since training happens locally, commit the `model/` folder to GitHub after training:
```bash
git add model/
git commit -m "Add trained model files"
git push
```

### Step 3 — Deploy on Streamlit Cloud
1. Go to **https://share.streamlit.io**
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository, branch (`main`), and main file (`app.py`)
5. Click **"Deploy!"**

Your app will be live at:
`https://YOUR_USERNAME-edupredict-ai-app-XXXXX.streamlit.app`

---

## 🎯 Risk Classes

| Class | Grade Range | Meaning |
|-------|-------------|---------|
| 🔴 At Risk | G3 < 10 | Needs immediate intervention |
| 🟡 Average | 10 ≤ G3 < 14 | Room for improvement |
| 🟢 High Performer | G3 ≥ 14 | Performing excellently |

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| Architecture | MLP: 256 → 128 → 64 → 32 → 3 |
| Activation | ReLU (hidden), Softmax (output) |
| Optimizer | Adam (lr=0.001) |
| Training epochs | 100 (no early stopping) |
| Test accuracy | ~91.9% |
| Dataset | 2,395 students, 32 features |

---

## 👩‍💻 Author

**Margaret Mukima Murungaru**
Admission No: CT204/109398/22
Supervisor: Mr. Kibaara
