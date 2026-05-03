import re

class TextPreprocessor:
    def limpiar_texto(self, texto):
        texto = texto.lower()
        texto = re.sub(r'[^\w\s]', '', texto)
        return texto


class EmoticonHandler:
    def procesar_emoticonos(self, texto):
        texto = texto.replace("😊", " positivo ")
        texto = texto.replace("👍", " positivo ")
        texto = texto.replace("😡", " negativo ")
        texto = texto.replace("👎", " negativo ")
        return texto


class NegationExpander:
    def expandir_negaciones(self, texto):
        texto = texto.replace("no", "no_")
        return texto