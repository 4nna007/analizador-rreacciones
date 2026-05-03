from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ModelEvaluator:
    def evaluar_modelo(self, modelo, X, y):
        y_pred = modelo.predict(X)

        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0)
        }


class CrossValidator:
    def validar(self, modelo, X, y):
        print("Validación cruzada no implementada aún")