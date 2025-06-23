# 🌌 Galactic Vision

**Galactic Vision** is an AI-powered constellation detection system that uses machine learning and computer vision to recognize star constellations from night sky images. Designed to make stargazing more accessible, this project combines deep learning with a beautiful frontend and real-time prediction capabilities.

---

## 🚀 Features

- ✨ **Real-time Constellation Detection**
- 🧠 **Trained on Annotated Celestial Datasets**
- 📱 **Interactive Web Interface (Frontend with 3D sky view)**
- 🌐 **API-Driven Flask Backend**
- 🎯 **High Accuracy on Real-world Sky Images**

---

## 🧠 Model Architecture

- **Backbone:** DenseNet 
- **Frameworks:** TensorFlow 
- **Training Dataset:** Real-world annotated constellation images
- **Techniques:** Image preprocessing, data augmentation, fine-tuning

---

## 🕸️ Tech Stack

| Component   | Tech Used                                  |
|------------|---------------------------------------------|
| Frontend   | HTML,CSS,Javascript, Three.js (3D Sky View) |
| Backend    | Flask (Python)                              |
| Model      | TensorFlow                                  |
| Hosting    | Hugging face (backend) + Vercel (frontend)  |
| Data       | Custom-labeled constellation dataset        |

---

## 🛠️ Getting Started

### 🔧 Prerequisites

- Python 3.9+
- Node.js & npm
- pip, virtualenv (optional)

### 🐍 Backend Setup

```bash
cd backend
pip install -r requirements.txt
python app.py
```
### Frontend Setup
 ```bash
cd frontend
npm install
npm run dev
```

### Model Training (Optional)
If you'd like to train your own model:
```
cd training
python train.py --config configs/vit.yaml
```

Deployment
Frontend deployed via Vercel

Backend deployed via Railway

Configure CORS and update API endpoints accordingly

Contributing
Contributions are welcome! Open an issue or submit a PR.
