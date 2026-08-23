import pandas as pd

# X: Personas de entre 40 y 45 años
# c: Persona de entre 10 y 45 años a la que se le OTORGA el préstamo
# h(x) = OTORGADO -> h([h_sexo, h_educacion, h_estado, h_prestamos])

# Cargar datos del csv
data = pd.read_csv('Préstamo.csv')
data.columns = data.columns.str.strip()

# Filtrar para personas de 50 años y seleccionar características
data_filtrada = data[(data['Edad'] >= 40) & (data['Edad'] <= 45)].copy()

atributos = ['Sexo', 'Mayor nivel educativo', 'Estado de vivienda', 'Préstamos previos impagos']
atributo_concepto = 'Estado'

PORCENTAJE_ENTRENAMIENTO = 0.80

# Partimos el set de datos en 80/20
index_corte = int(len(data_filtrada) * PORCENTAJE_ENTRENAMIENTO)

# Mezclar y extraer el 80% de los datos para entrenamiento
set_entrenamiento = data_filtrada.sample(frac=PORCENTAJE_ENTRENAMIENTO, random_state=42)

# random_state -> Seed para que sea reproducible, si no se pone, cada vez que se corre el programa va a dar un resultado distinto

# El set  de prueba es el resto del dataset 
set_prueba = data_filtrada.drop(set_entrenamiento.index)

print(f"Ejemplos totales (50 años): {len(data_filtrada)}")
print(f"Ejemplos de entrenamiento: {len(set_entrenamiento)}")
print(f"Ejemplos de prueba: {len(set_prueba)}")

# FIND-S
def find_s(X_ejemplos, y_conceptos, concepto_positivo):
    # Arranca todo en vacio
    hipotesis = [None] * len(X_ejemplos.columns) #Vacio en todas las caracteristicas
    
    # Convertimos los datos a listas para poder iterar sobre ellos
    lista_ejemplos = X_ejemplos.values.tolist()
    lista_conceptos = y_conceptos.tolist()
    
    for i in range(len(lista_ejemplos)): #Iteramos sobre los ejemplos para ir definiendo la hipotesis
        # Solo usamos los positivos (OTORGADO en este caso)
        if lista_conceptos[i] == concepto_positivo:
            #Tratamos con el ejemplo seleccionado
            x_ejemplo = lista_ejemplos[i]
            for j in range(len(hipotesis)):
                if hipotesis[j] is None: #Si esta vacio y cumplio, nos quedamos con el atributo del ejemplo
                    hipotesis[j] = x_ejemplo[j]
                elif hipotesis[j] != x_ejemplo[j]: #Si cumplio y no es el atibuto del ejemplo, podemos generalizar
                    hipotesis[j] = "?" 
                    
    return hipotesis

#Set de entrenamiento separado en atributos y objetivo
X_entrenamiento_caracteristicas = set_entrenamiento[atributos]
X_entrenamiento_objetivo = set_entrenamiento[atributo_concepto]

h_final = find_s(X_entrenamiento_caracteristicas, X_entrenamiento_objetivo, concepto_positivo="OTORGADO")

print("Hipótesis más específica obtenida:")
for campo, valor in zip(atributos, h_final):
    print(f"- {campo}: {valor}")
print()

# Predicción en el conjunto de prueba
def predecir_find_s(hipotesis, X_prueba, campo_positivo, campo_negativo):

    predicciones = []

    for ejemplo in X_prueba:
        match = True
        for j in range(len(hipotesis)):
            # Si la caracterisitca comparada no es general ni coincide con el ejemplo, no califica
            if hipotesis[j] != '?' and hipotesis[j] != ejemplo[j]:
                match = False
                break

        #Vamos agregando las predicciones a la lista
        if match:
            predicciones.append(campo_positivo)
        else:
            predicciones.append(campo_negativo)
            
    return predicciones

# Separar los atributos y predecir
X_prueba_caracteristicas = set_prueba[atributos].values.tolist()
X_prueba_objetivo = set_prueba[atributo_concepto].tolist()
cantidad_ejemplos_prueba = len(set_prueba)

lista_predicciones = predecir_find_s(h_final, X_prueba_caracteristicas, campo_positivo="OTORGADO", campo_negativo="RECHAZADO")


# Calcular aciertos y accuracy
bien = 0
for i in range(cantidad_ejemplos_prueba):
    if lista_predicciones[i] == X_prueba_objetivo[i]:
        bien += 1

accuracy = bien / cantidad_ejemplos_prueba

print(f"Aciertos: {bien} / {cantidad_ejemplos_prueba}")
print(f"Accuracy en Test: {accuracy * 100:.2f}%")

# (2)
# Matriz de confusión
valores_predichos = lista_predicciones
valores_reales = set_prueba[atributo_concepto].tolist()

def matriz_confusion(valores_reales, valores_predichos):
    # Inicializamos la matriz de confusión
    matriz = {
        "TP": 0,  # Verdaderos positivos
        "TN": 0,  # Verdaderos negativos
        "FP": 0,  # Falsos positivos
        "FN": 0,   # Falsos negativos
        "TOTAL": len(valores_reales)  # Total de ejemplos
    }

    for real, predicho in zip(valores_reales, valores_predichos):
        if real == "OTORGADO" and predicho == "OTORGADO":
            matriz["TP"] += 1
        elif real == "RECHAZADO" and predicho == "RECHAZADO":
            matriz["TN"] += 1
        elif real == "RECHAZADO" and predicho == "OTORGADO":
            matriz["FP"] += 1
        elif real == "OTORGADO" and predicho == "RECHAZADO":
            matriz["FN"] += 1

    return matriz

matriz = matriz_confusion(valores_reales, valores_predichos)

print("\nMatriz de Confusión:")
print(f"Verdaderos Positivos (TP): {matriz['TP']}")
print(f"Verdaderos Negativos (TN): {matriz['TN']}")
print(f"Falsos Positivos (FP): {matriz['FP']}")
print(f"Falsos Negativos (FN): {matriz['FN']}")

# accuracy
# proporcion de instancias que han sido correctamente clasificadas
accuracy = (matriz['TP'] + matriz['TN']) / matriz['TOTAL']
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# recall
# probabilidad de que el clasificador detecte un caso positivo cuando en verdad lo es
recall = matriz['TP'] / (matriz['TP'] + matriz['FN']) 
print(f"Recall: {recall * 100:.2f}%")

# especificidad
# probabilidad de que el clasificador detecte un caso negativo cuando en verdad lo es
especificidad = matriz['TN'] / (matriz['TN'] + matriz['FP'])
print(f"Especificidad: {especificidad * 100:.2f}%")

# precisión
# probabilidad de que el clasificador detecte correctamente un caso positivo
precision = matriz['TP'] / (matriz['TP'] + matriz['FP'])
print(f"Precisión: {precision * 100:.2f}%")

# F1-score
# combinación de precisión y recall para devolver una medida de calidad mas general del modelo
f1_score = 2 * (precision * recall) / (precision + recall)
print(f"F1-score: {f1_score * 100:.2f}%")

# Tasa de verdaderos positivos (TPR)
tpr = matriz['TP'] / (matriz['TP'] + matriz['FN'])
print(f"Tasa de Verdaderos Positivos (TPR): {tpr * 100:.2f}%")

# Tasa de falsos positivos (FPR)
fpr = matriz['FP'] / (matriz['FP'] + matriz['TN'])
print(f"Tasa de Falsos Positivos (FPR): {fpr * 100:.2f}%")  

# Espacio ROC
print(f"Espacio ROC: ({tpr}, {fpr})%")  

print(f"Espacio ROC: ({tpr}, {fpr})%")  

