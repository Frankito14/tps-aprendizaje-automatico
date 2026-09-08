import pandas as pd
import matplotlib.pyplot as plt

# X: Personas de entre 40 y 45 años
# c: Persona de entre 40 y 45 años a la que se le OTORGA el préstamo
# h(x) = OTORGADO -> h([h_sexo, h_educacion, h_estado, h_prestamos])

# Cargar datos del csv
data = pd.read_csv('Préstamo.csv')
data.columns = data.columns.str.strip()

# Filtrar para personas de entre 40 y 45 años
data_filtrada = data[(data['Edad'] >= 40) & (data['Edad'] <= 45)].copy()

atributos = ['Sexo', 'Mayor nivel educativo', 'Estado de vivienda', 'Préstamos previos impagos']
atributo_concepto = 'Estado'

PORCENTAJE_ENTRENAMIENTO = 0.80

# Mezclar y extraer el 80% de los datos para entrenamiento de forma aleatoria
set_entrenamiento = data_filtrada.sample(frac=PORCENTAJE_ENTRENAMIENTO, random_state=42)

# El set de prueba es el resto del dataset (20%)
set_prueba = data_filtrada.drop(set_entrenamiento.index)

print(f"Ejemplos totales (40-45 años): {len(data_filtrada)}")
print(f"Ejemplos de entrenamiento (80%): {len(set_entrenamiento)}")
print(f"Ejemplos de prueba (20%): {len(set_prueba)}")


# ENTRENAMIENTO MANUAL DE NAIVE BAYES

# ESPACIO MUESTRAL = [RECHAZADO, OTORGADO]
# P(R) = RECHAZADO / TOTAL
# P(O) = OTORGADO / TOTAL
# R ∩ O = VACIO -> MUTUAMENTE EXCLUYENTES

print("ENTRENAMIETO DEL MODELO")
print("\nDatos usados para entrenamiento de Naive Bayes:") 
print(f"Total de ejemplos: N = {len(set_entrenamiento)}")
print(f"Cantidad de atributos: d = {len(atributos)}")
print(f"Cantidad de clases: K =  2 (OTORGADO, RECHAZADO)")

# Valores unicos que puede obtener cada atributo (Mj)
atributos_valores_unicos = {}
for col in atributos:
    atributos_valores_unicos[col] = list(data_filtrada[col].unique())

print("\nCantidad de posibles valores de los atributos:")
for index, (col, vals) in enumerate(atributos_valores_unicos.items()):
    print(f"M{index} = {len(vals)} ({col}) -> {vals}")

clases = ['OTORGADO', 'RECHAZADO']
total_ejemplos_entrenamiento = len(set_entrenamiento)

# Total de ejemplos para cada clase
total_otorgados = set_entrenamiento[atributo_concepto].value_counts().to_dict().get('OTORGADO', 0)
total_rechazados = set_entrenamiento[atributo_concepto].value_counts().to_dict().get('RECHAZADO', 0)

# Probabilidades a priori de cada clase
def priori(cantidad, total):
    return (cantidad) / (total)

def priori_suavizado(cantidad, total, total_clases, l=1):
    numerador = cantidad + l
    denominador = total + (l * total_clases)
    return numerador / denominador

prioris = {}
prioris["OTORGADO"] = priori_suavizado(total_otorgados, total_ejemplos_entrenamiento, len(clases))
prioris["RECHAZADO"] = priori_suavizado(total_rechazados, total_ejemplos_entrenamiento, len(clases))

print("\nProbabilidades a prori:")
print(f"P(OTROGADO) = {total_otorgados} / {total_ejemplos_entrenamiento}  = {prioris['OTORGADO']:.2f}")
print(f"P(RECHAZADO) = {total_rechazados} / {total_ejemplos_entrenamiento} = {prioris['RECHAZADO']:.2f}")

# Verosimilitudes con lapace

# Separar los ejemplos segun cada clase
#ejemplos_otorgados = set_entrenamiento[set_entrenamiento[atributo_concepto] == 'OTORGADO']
#ejemplos_rechazados = set_entrenamiento[set_entrenamiento[atributo_concepto] == 'RECHAZADO']

def verosimilitud_suavizada(df, clase, columna, valor, Mj, l=1):
    #df: dataframe (ejemplos)
    #clase: la clase a buscar 
    #columna: columna a evaluar (campo del atributo)
    #valor: valor que buscamos en el df : string 
    #Mj: cantidad de valores únicos : number
    #l: suavizado (1)
  
    # (Xj = xjm ^ Y = Ck) + l
    casos_favorables = ((df[columna] == valor) & (df[atributo_concepto] == clase)).sum()
    numerador = casos_favorables + l
    
    # (Y = Ck) + l
    total_clase = (df[atributo_concepto] == clase).sum()
    denominador = total_clase + (l * Mj)
    
    return numerador / denominador

verosimilitudes = {
    "OTORGADO": {},
    "RECHAZADO": {}
}

for clase in clases:
    for col in atributos:
        verosimilitudes[clase][col] = {}
        cantidad_valores_unicos = len(atributos_valores_unicos[col])
        for valor in atributos_valores_unicos[col]:
            prob= verosimilitud_suavizada(
                df=set_entrenamiento, 
                clase=clase, 
                columna=col, 
                valor=valor, 
                Mj=cantidad_valores_unicos, 
                l=1
            )
            verosimilitudes[clase][col][valor] = prob

# Mostrar comparacion de verosimilitudes
def mostrar_verosimilitudes():
    print("\nVerosimilitudes calculadas con suavizado de Laplace:")
    for clase, atributos_dict in verosimilitudes.items():
        print(f"\n[ CLASE: {clase} ]")
        for col, valores_dict in atributos_dict.items():
            print(f"  Atributo: '{col}'")
            for val, prob in valores_dict.items():
                # Formateamos con la notación formal P(Atributo = Valor | Clase)
                print(f"    P({col} = '{val}' | {clase}) = {prob:.4f}")


# Predecir UN ejemplo con Naive Bayes
def predecir_ejemplo_bayes(ejemplo, prioris, verosimilitudes, atributos):
    """
    ejemplo: ejemplo de prueba (lista de valores de atributos)
    prioris: Diccionario con las probabilidades a priori de cada clase
    verosimilitudes: El diccionario de verosimilitudes_modelo que ya calculamos
    atributos: Lista con los nombres de las columnas de atributos
    """
    #scores de las clases
    printear = False
    score_otorgado = prioris['OTORGADO']
    score_rechazado = prioris['RECHAZADO']
    
    # Multiplicamos las verosimilitudes de cada atributo para cada clase
    for i in range(len(atributos)):
        columna = atributos[i]
        valor = ejemplo[i]
        
        # P(Atributo = Valor | OTORGADO)
        prob_cond_otorgado = verosimilitudes['OTORGADO'][columna][valor]
        score_otorgado *= prob_cond_otorgado
        
        # P(Atributo = Valor | RECHAZADO)
        prob_cond_rechazado = verosimilitudes['RECHAZADO'][columna][valor]
        score_rechazado *= prob_cond_rechazado
        
    # argmax para determinar la clase con mayor score
    if score_otorgado >= score_rechazado: # hollaaa
        prediccion = 'OTORGADO'
        if printear:
            print(f"\nPredicción para el ejemplo {ejemplo}:")
            print(f"Score(OTORGADO) = {score_otorgado:.6f}")
            print(f"Score(RECHAZADO) = {score_rechazado:.6f}")
    else:
        prediccion = 'RECHAZADO'
        if printear:
                print(f"\nPredicción para el ejemplo {ejemplo}:")
                print(f"Score(OTORGADO) = {score_otorgado:.6f}")
                print(f"Score(RECHAZADO) = {score_rechazado:.6f}")
        
    # Normalización bayesiana para obtener la probabilidad exacta de 'OTORGADO'
    # P(OTORGADO | x) = Score(OTORGADO) / (Score(OTORGADO) + Score(RECHAZADO))
    prob_otorgado = score_otorgado / (score_otorgado + score_rechazado)
    
    return prediccion, prob_otorgado

mostrar_verosimilitudes()

lista_predicciones = []
probabilidades_test = []

# Hacemos las predicciones para cada ejemplo del set de prueba
for x in set_prueba[atributos].values.tolist():
    prediccion, probabilidad_posicion = predecir_ejemplo_bayes(
        ejemplo=x,
        prioris=prioris,
        verosimilitudes=verosimilitudes, 
        atributos=atributos
    )
    
    lista_predicciones.append(prediccion)
    probabilidades_test.append(probabilidad_posicion)

# Contamos las preddicciones correctas
bien = 0
valores_reales = set_prueba[atributo_concepto].tolist()

for i in range(len(set_prueba)):
    if lista_predicciones[i] == valores_reales[i]:
        bien += 1

accuracy = bien / len(set_prueba)

print("TESTEO DEL MODELO")

print("\nMetricas de Naive Bayes en el set de prueba:")
print(f"Aciertos: {bien} / {len(set_prueba)}")
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

#Curva ROC
P = matriz['TP'] + matriz['FN']  
N = matriz['TN'] + matriz['FP']  
puntos_fpr = [0.0]  # Arrancamos en el punto (0,0) (Umbral u > 1.0) 
puntos_tpr = [0.0]
umbrales_unicos = sorted(list(set(probabilidades_test)), reverse=True)

# Iteramos por cada umbral para calcular su matriz de confusión y sus tasas [1, 3-6]
for u in umbrales_unicos:
    tp = 0
    fp = 0
    
    # Clasificamos cada ejemplo de prueba según el umbral actual "u"
    for i in range(len(set_prueba)):
        prob = probabilidades_test[i]
        real = valores_reales[i]
        
        # Si la probabilidad es mayor o igual al umbral, se predice Positivo ('OTORGADO')
        if prob >= u:
            if real == "OTORGADO":
                tp += 1      # Verdadero Positivo
            else:
                fp += 1      # Falso Positivo
                
    # Calculamos las tasas teóricas correspondientes para este umbral [1, 7]
    tpr = tp / P if P > 0 else 0.0  # Tasa Verdaderos Positivos (Recall)
    fpr = fp / N if N > 0 else 0.0  # Tasa Falsos Positivos (1 - Especificidad)
    
    puntos_fpr.append(fpr)
    puntos_tpr.append(tpr)

# Forzamos el último punto en (1,1) (Umbral u = 0.0) [6]
if puntos_fpr[-1] != 1.0 or puntos_tpr[-1] != 1.0:
    puntos_fpr.append(1.0)
    puntos_tpr.append(1.0)



# Calcular AUC 
auc = 0.0
for i in range(1, len(puntos_fpr)):
    # Ancho del trapecio en el eje X (FPR)
    base = puntos_fpr[i] - puntos_fpr[i-1]
    # Altura promedio en el eje Y (TPR)
    altura_promedio = (puntos_tpr[i] + puntos_tpr[i-1]) / 2.0
    auc += base * altura_promedio


# Graficar la curva ROC y el AUC
plt.figure(figsize=(7, 6))

# Dibujamos la diagonal de adivinación aleatoria (FPR = TPR, AUC = 0.50) [8, 9]
plt.plot([10], color='red', linestyle='--', label='Adivinación al Azar (AUC = 0.50)')

# Dibujamos los puntos y la curva ROC de nuestro modelo Naive Bayes [6]
plt.plot(puntos_fpr, puntos_tpr, color='blue', marker='o', linewidth=2, label=f'Naive Bayes (AUC = {auc:.4f})')

# Rellenamos el área bajo la curva con un color suave para ilustrar el AUC
plt.fill_between(puntos_fpr, puntos_tpr, color='blue', alpha=0.1)

# Configuramos títulos y etiquetas alineadas a la materia [7]
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR) / (1 - Especificidad)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR) / Recall')
plt.title('Curva ROC - Naive Bayes (Conjunto de Prueba 40-45 años)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower right')

# Guardamos el archivo de la imagen y la mostramos
plt.savefig('curva_roc_test.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n==============================================")
print("EVALUACIÓN DEL ESPACIO ROC")
print("==============================================")
print(f"Puntos FPR evaluados: {[round(f, 4) for f in puntos_fpr]}")
print(f"Puntos TPR evaluados: {[round(t, 4) for t in puntos_tpr]}")
print(f"Área Bajo la Curva ROC (AUC): {auc:.4f}")
print("==============================================")
