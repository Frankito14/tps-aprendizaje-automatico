import pandas as pd

# X: Personas de 50 años
# c: Persona de 50 años a la que se le OTORGA el préstamo
# h(x) = OTORGADO -> h([h_sexo, h_educacion, h_estado, h_prestamos])

# Cargar datos del csv
data = pd.read_csv('Préstamo.csv')
data.columns = data.columns.str.strip()

# Filtrar para personas de 50 años y seleccionar características
data_filtrada = data[data['Edad'] == 50].copy()

atributos = ['Sexo', 'Mayor nivel educativo', 'Estado de vivienda', 'Préstamos previos impagos']
atributo_concepto = 'Estado'

# Partimos el set de datos en 75/25
index_corte = int(len(data_filtrada) * 0.75)

set_entrenamiento = data_filtrada.iloc[:index_corte]
set_prueba = data_filtrada.iloc[index_corte:]

print(f"Ejemplos totales (50 años): {len(data_filtrada)}")
print(f"Ejemplos de entrenamiento: {len(set_entrenamiento)}")
print(f"Ejemplos de prueba: {len(set_prueba)}")

# FIND-S
def find_s(X_ejemplos, y_conceptos, concepto_positivo):
    """
    X_ejemplos: Set de ejemplos
    y_conceptos: Set de conceptos (objetivo)
    concepto_positivo: Concepto positivo a buscar (OTORGADO)
    """
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
    """
    hipotesis: Hipótesis obtenida del algoritmo FIND-S
    X_prueba: Set de prueba (características)
    campo_positivo: Valor a predecir si coincide con la hipótesis
    campo_negativo: Valor a predecir si no coincide con la hipótesis
    """

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
