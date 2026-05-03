# 🧠 Analizador de Sentimientos con Machine Learning

Este mini proyecto implementa un sistema de análisis de reacciones utilizando técnicas de procesamiento de lenguaje natural (NLP) y modelos de Machine Learning.

## 🚀 Características

- Preprocesamiento de texto (limpieza, emoticonos, negaciones)
- Extracción de características con TF-IDF
- Entrenamiento de múltiples modelos:
  - SVM
  - Random Forest
- Evaluación con métricas:
  - Accuracy
  - Precision
  - Recall
  - F1-score


## ⚙️ Tecnologías utilizadas

- Python
- scikit-learn
- NLTK
- Gensim
- NumPy
- Matplotlib

## ▶️ Ejecución

```bash
pip install -r requirements.txt
python src/main.py

## 📂 Estructura del proyecto

analizador-reacciones/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── modeling.py
│   ├── evaluation.py
│   └── main.py
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
└── requirements.txt
