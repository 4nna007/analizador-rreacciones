from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureExtractor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def extraer_tfidf(self, textos):
        return self.vectorizer.fit_transform(textos)


class VectorizerComparison:
    def comparar(self):
        print("Comparación de vectorizadores no implementada aún")


class FeatureSelector:
    def seleccionar(self):
        print("Selección de características no implementada aún")