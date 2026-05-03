from preprocessing import TextPreprocessor, EmoticonHandler, NegationExpander
from feature_extraction import FeatureExtractor
from modeling import ModelTrainer
from evaluation import ModelEvaluator

# 🔹 Datos 
textos = [
    "Me encanta este producto 😊",
    "No me gusta nada 😡",
    "Está bien, pero podría mejorar",
    "Excelente servicio 👍",
    "Muy malo 👎",
    "Horrible experiencia 😡",
    "Muy recomendado 😊",
    "No lo volvería a comprar 👎",
    "Excelente calidad 😊",
    "Pésimo servicio 😡"
]

labels = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]


# 🔹 Preprocesamiento
pre = TextPreprocessor()
emo = EmoticonHandler()
neg = NegationExpander()

textos_procesados = []

for t in textos:
    t = emo.procesar_emoticonos(t)
    t = pre.limpiar_texto(t)
    t = neg.expandir_negaciones(t)
    textos_procesados.append(t)

# 🔹 Features
fe = FeatureExtractor()
X = fe.extraer_tfidf(textos_procesados)

print("Textos:", len(textos_procesados))
print("Labels:", len(labels))

# 🔹 División de datos
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    labels,
    test_size=0.3,
    random_state=42,
    stratify=labels   # 🔥 ESTA LÍNEA ES LA CLAVE
)

# 🔹 Modelo
trainer = ModelTrainer()
modelos = trainer.entrenar_modelos(X_train, y_train)

# 🔹 Evaluación
evaluator = ModelEvaluator()

for nombre, modelo in modelos.items():
    resultado = evaluator.evaluar_modelo(modelo, X_test, y_test)
    print(f"\nModelo: {nombre}")
    print("Accuracy:", resultado["accuracy"])
    print("Precision:", resultado["precision"])
    print("Recall:", resultado["recall"])
    print("F1:", resultado["f1"])