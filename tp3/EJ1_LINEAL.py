import numpy as np
import pandas as pd

def entrenar_regresion_lineal(X_train, y_train):
    """
    Proposito: Calcular los coeficientes de la RLM mediante la ecuación normal.
    Tipo: np.ndarray
    """

    # Agregar una columna de 1s para el termino indepentendiente β0.
    X_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]

    # Aplicar la fórmula de la ecuación normal: beta = (X^T * X)^(-1) * X^T * y
    beta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
    return beta # Coeficientes 

def predecir(X, beta):
    """
    Proposito: Calcular las predicciones a partir de los coeficientes.
    Tipo: np.ndarray
    """
    # Agregar un 1 a la matriz de datos para calcular la predicción.
    X_b = np.c_[np.ones((X.shape[0], 1)), X]

    # 
    return X_b.dot(beta)

def predecir_estudiante(estudiante, beta):
    """
    Propósito: Recibe los valores de las variables predictoras de un estudiante,
    les agrega el intercepto y calcula la calificación predicha.
    Tipo: Float
    """
    # Convertimos los datos a un arreglo de numpy asegurando que tenga la forma correcta (1, n)
    X_nuevo = np.array([estudiante])
    
    calificacion_predicha = beta[0] + beta[1]*estudiante[0] + beta[2]*estudiante[1] + beta[3]*estudiante[2]
    
    return float(calificacion_predicha)

def calcular_r2(y_real, y_pred):
    """
    Proposito: Calcular el coeficiente de determinación R^2 para calcular el ajuste del modelo.
    Tipo: Float
    """
    # Suma de cuadrados de los errores.
    sse = np.sum((y_real - y_pred) ** 2)
    
    # Suma de cuadrados totales.
    sst = np.sum((y_real - np.mean(y_real)) ** 2)

    # Coeficiente de determinación R^2
    r2 = 1 - (sse / sst)
    return float(r2)

# Cargar los datos
print("\n")
print("Regresion Lineal Multiple \n")
datos = pd.read_csv("Estudiantes.csv")
datos_originales = datos.copy()

# 1. DIVISIÓN DEL CONJUNTO (80% Entrenamiento / 20% Prueba)
n_entrenamiento = int(len(datos) * 0.80)

entrenamiento = datos.iloc[:n_entrenamiento].copy()
testeo = datos.iloc[n_entrenamiento:].copy()

print(f"Total de ejemplos: {len(datos)}")
print(f"Entrenamiento: {len(entrenamiento)} | Prueba: {len(testeo)}\n")

# Variables
variables_predictoras = ['Horas de estudio por semana', 'Calificaciones previas', 'Porcentaje de asistencia']

# x1: horas de estudio,
# x2: calificaciónes previa,
# x3: porcentaje de asistencia

variable_respuesta = 'Calificación'

# y = calificación
# y = β0 + β1x1 + β2x2 + β3x3 + ε.
# β = (β0, β1, β2, β3)

# Matrices de entrenamiento y prueba
X_train = entrenamiento[variables_predictoras].values
y_train = entrenamiento[variable_respuesta].values

X_test = testeo[variables_predictoras].values
y_test = testeo[variable_respuesta].values

# 2. ECUACIÓN DEL MODELO Y COEFICIENTE R^2
beta = entrenar_regresion_lineal(X_train, y_train)

intercepto = beta[0]

coeficientes = beta[1:]

print("Coeficientes:\n")
print(f"β0 = {intercepto:.4f}")
for i, var in enumerate(variables_predictoras):
    print(f"β{i+1} = {coeficientes[i]:.4f}")
print("\n")


print(f"y^ = {intercepto:.4f}", end="")
for i, var in enumerate(variables_predictoras):
    print(f" + {coeficientes[i]:.4f}x{i+1} ", end="")
print("\n")

print(f"{variable_respuesta} = {intercepto:.4f}", end="")
for i, var in enumerate(variables_predictoras):
    print(f" + ({coeficientes[i]:.4f} * {var})", end="")
print("\n")

# Realizar predicciones y calcular R^2 en el conjunto de prueba
y_pred_test = predecir(X_test, beta)
r2_score = calcular_r2(y_test, y_pred_test)

print(f"Coeficiente de determinación R^2 (en conjunto de prueba): {r2_score:.4f}")
print("\n")

nuevo_estudiante = [25, 58, 68]

calificacion_estimada = predecir_estudiante(nuevo_estudiante, beta)

print("Prediccion de estudiante:")
for var, valor in zip(variables_predictoras, nuevo_estudiante):
    print(f"- {var}: {valor}")
print("\n")

print(f"{variable_respuesta} = {intercepto:.4f}", end="")
for i, var in enumerate(variables_predictoras):
    print(f" + ({coeficientes[i]:.4f} * {nuevo_estudiante[i]})", end="")
print(f" = {calificacion_estimada:.4f} puntos")
print(f"Condición: {'Aprobado' if calificacion_estimada >= 60 else 'Reprobado'}")
print("\n")

