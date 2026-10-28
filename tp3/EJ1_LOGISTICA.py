import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

def entrenar_regresion_logistica(X_train, y_train):
    """
    Propósito: Entrenar el modelo de regresión logística binaria.
    Tipo: LogisticRegression
    """
    modelo = LogisticRegression()
    modelo.fit(X_train, y_train)
    return modelo

def evaluar_clasificador(y_real, y_pred):
    """
    Propósito: Calcular la matriz de confusión, el accuracy y el F1-score del modelo.
    Tipo: tuple
    """
    matriz = confusion_matrix(y_real, y_pred)
    acc = accuracy_score(y_real, y_pred)
    f1 = f1_score(y_real, y_pred, average='weighted')
    return matriz, acc, f1

def predecir_estudiante(modelo, estudiante, variables_predictoras):
    """
    Propósito: Hace una predicción de la condición del estudiante pasado como parámetro.
    Tipo: tuple (clase predicha, probabilidades)
    """
    # Corregido: pasamos los datos como una lista de listas [[...]] para que tenga la forma (1, n) correcta
    X_nuevo = pd.DataFrame([estudiante], columns=variables_predictoras)
    
    # Predecir la clase (0 o 1) y obtener sus probabilidades
    prediccion = modelo.predict(X_nuevo)[0]
    probabilidad = modelo.predict_proba(X_nuevo)[0]
    
    return prediccion, probabilidad

# Cargar los datos
datos = pd.read_csv("Estudiantes.csv")
datos_originales = datos.copy()

# Division del conjunto (80% Entrenamiento / 20% Prueba)
n_entrenamiento = int(len(datos) * 0.80)

entrenamiento = datos.iloc[:n_entrenamiento].copy()
testeo = datos.iloc[n_entrenamiento:].copy()

print("\n")
print("Regresion Lineal Multiple \n")
print(f"Total de ejemplos: {len(datos)}")
print(f"Entrenamiento: {len(entrenamiento)} | Prueba: {len(testeo)}\n")

# Variables predictoras y de respuesta
variables_predictoras = ['Horas de estudio por semana', 'Porcentaje de asistencia', 'Calificaciones previas']
variable_respuesta = 'Condición'

# Preparar conjuntos de entrenamiento y prueba
X_train = entrenamiento[variables_predictoras]
y_train_log = entrenamiento[variable_respuesta]

X_test = testeo[variables_predictoras]
y_test_log = testeo[variable_respuesta]

# Entrenar modelo
modelo_logistico = entrenar_regresion_logistica(X_train, y_train_log)

# Extraer intercepto y coeficientes del modelo logístico binario
intercepto = modelo_logistico.intercept_[0]
coeficientes = modelo_logistico.coef_[0]

print("\n")
print(f"Ecuación del Logit (z) = {intercepto:.4f}", end="")
for i, var in enumerate(variables_predictoras):
    print(f" + ({coeficientes[i]:.4f} * {var})", end="")
print("\n")

# Realizar predicciones y calcular métricas en el conjunto de prueba
y_pred_log = modelo_logistico.predict(X_test)
matriz_conf, accuracy, f1 = evaluar_clasificador(y_test_log, y_pred_log)
tn, fp, fn, tp = matriz_conf.ravel()

print("Matriz de Confusión (Test):")
print(f"Verdaderos Positivos: {tp}")
print(f"Falsos Positivos: {fp}")
print(f"Verdaderos Negativos: {tn}")
print(f"Falsos Negativos: {fn}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}\n")

# Predicción de estudiante ejemplo
estudiante_ejemplo = [25, 58, 68]  # [Horas de estudio, Porcentaje de asistencia, Calificaciones previas]
condicion_predicha, probabilidades = predecir_estudiante(modelo_logistico, estudiante_ejemplo, variables_predictoras)

print("Predicción de estudiante:")
for var, valor in zip(variables_predictoras, estudiante_ejemplo):
    print(f"- {var}: {valor}")
print("\n")
print(f"Condición Predicha: {'Aprobado (1)' if condicion_predicha == 1 else 'Reprobado (0)'}")
print(f"Probabilidades -> Reprobado: {probabilidades[0]:.4f} | Aprobado: {probabilidades[1]:.4f}")
print("\n")