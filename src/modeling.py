from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


class ModelTrainer:
    def entrenar_modelos(self, X, y):
        modelos = {
            "RandomForest": RandomForestClassifier(),
            "SVM": SVC()
        }

        for nombre, modelo in modelos.items():
            modelo.fit(X, y)

        return modelos