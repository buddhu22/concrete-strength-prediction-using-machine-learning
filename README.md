# 🏗️ Concrete Compressive Strength Prediction (AI Estimator)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?logo=pandas)
![License](https://img.shields.io/badge/License-MIT-green)

A premium, production-grade Machine Learning Dashboard built with **Streamlit**. This application accurately predicts the compressive strength of high-performance concrete using advanced Machine Learning algorithms, effectively bypassing the traditional, time-consuming 28-day physical destruction tests.

---

## 🌟 Key Features

- **🧠 Advanced AI Prediction**: Uses highly accurate **Random Forest Regressor** and **Linear Regression** models to estimate concrete strength instantly based on 8 key parameters.
- **📱 Premium SaaS Dashboard**: Modern, responsive, and beautiful UI/UX utilizing custom CSS, glassmorphism, and interactive layouts.
- **📈 Data Visualization**: Interactive **Plotly** charts for feature importance, correlation heatmaps, and dataset distribution.
- **📄 Export Reports**: Generate and download professional prediction reports directly in **CSV** or **PDF (Text)** formats.
- **🎓 Educational Focus**: Built-in "How It Works" and "About Project" sections making it perfect for academic presentations, civil engineering students, and researchers.

---

## 🛠️ Technology Stack

* **Frontend/UI**: Streamlit, Custom HTML/CSS
* **Backend/Data Science**: Python 3, Pandas, NumPy
* **Machine Learning**: Scikit-Learn
* **Data Visualization**: Plotly, Matplotlib, Seaborn

---

## 🚀 Installation & Setup

Follow these steps to run the AI Estimator on your local machine:

**1. Clone the repository**
```bash
git clone https://github.com/buddhu22/concrete-strength-prediction-using-machine-learning.git
cd concrete-strength-prediction-using-machine-learning
```

**2. Create a Virtual Environment (Optional but Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Train the ML Models**
*(If the models aren't present in the `models/` directory, you must run this script first)*
```bash
python src/train_model.py
```

**5. Launch the Streamlit App**
```bash
streamlit run app.py
```
The app will automatically open in your default browser at `http://localhost:8501/`.

---

## 📂 Project Structure

```text
├── .streamlit/             # Custom Streamlit theme configuration
├── assets/                 # Images and UI assets
├── data/                   # Raw and processed datasets (Concrete_Data.xls)
├── models/                 # Saved ML models (Random Forest, Linear Regression, Scaler)
├── notebooks/              # Jupyter notebooks for EDA and testing
├── src/                    # Source code (preprocessing, training, evaluation scripts)
├── app.py                  # Main Streamlit Dashboard application
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 🎯 How It Works

1. **Input Variables**: The user inputs the quantities of Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate, Fine Aggregate, and Curing Age (in days).
2. **Preprocessing**: The inputs are normalized using a trained `StandardScaler`.
3. **AI Inference**: The pre-trained Random Forest model analyzes the chemical/physical ratios.
4. **Output**: The system displays the predicted Compressive Strength in **Megapascals (MPa)** along with a confidence score and structural engineering recommendations.

---

## 👨‍💻 Developer

**Designed and Developed by [Abhay Mishra](https://github.com/buddhu22)**  
*Built for Academic and Professional Presentations.*

If you found this project helpful, please give it a ⭐ on GitHub!
