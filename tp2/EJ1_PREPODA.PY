import pandas as pd
import numpy as np
import math

# FUNCIONES AUXILIARES

# PROB. PRIORI
def calcular_prioris(df, columna_objetivo):
    """
    p_k = N_k(D) / N(D)
    Calcula las proporciones/probabilidades a priori para cada clase en el nodo o conjunto actual
    """
    total = len(df)
    if total == 0:
        return {}
    conteos = df[columna_objetivo].value_counts()
    prioris = {clase: conteo / total for clase, conteo in conteos.items()}
    return prioris

def entropia_shannon(df, columna_objetivo):
    """
    H(D) = - sum(p_k * log2(p_k))
    Calcula la Entropía de Shannon
    Si un p_k es 0, se considera nulo (lim x->0 x*ln(x) = 0)
    """
    prioris = calcular_prioris(df, columna_objetivo)
    h = 0.0
    for p in prioris.values():
        if p > 0:
            h -= p * math.log2(p)
    return h


def entropia_condicional(df, atributo, columna_objetivo):
    """
    H(D|A) = sum( (N(D_a) / N(D)) * H(D_a) )
    Calcula la entropía ponderada de los nodos hijos H(D|A):
    """
    total_ejemplos = len(df)
    if total_ejemplos == 0:
        return 0.0

    h_cond = 0.0
    valores_atributo = df[atributo].unique()

    for val in valores_atributo:
        sub_df = df[df[atributo] == val]
        peso = len(sub_df) / total_ejemplos
        h_cond += peso * entropia_shannon(sub_df, columna_objetivo)

    return h_cond


def ganancia_informacion(df, atributo, columna_objetivo):
    """
    IG(A) = H(D) - H(D|A)
    Calcula la Ganancia de Información (Information Gain) IG(A)
    """
    h_padre = entropia_shannon(df, columna_objetivo)
    h_hijos = entropia_condicional(df, atributo, columna_objetivo)
    return h_padre - h_hijos


def obtener_clase_mayoritaria(df, columna_objetivo):
    """
    Retorna la clase mayoritaria
    """
    return df[columna_objetivo].mode().iloc[0]


# ALGORITMO ID3

def construir_arbol_id3(df, atributos, columna_objetivo,
                         profundidad_max=None, division_minima=2, ganancia_minima=0.0,
                         profundidad_actual=0):
    """
    Genera un árbol de decisión ID3 con soporte para poda.
    """
    clases_unicas = df[columna_objetivo].unique()

    if len(clases_unicas) == 1:
        return clases_unicas[0]

    if len(atributos) == 0:
        return obtener_clase_mayoritaria(df, columna_objetivo)

    if profundidad_max is not None and profundidad_actual >= profundidad_max:
        return obtener_clase_mayoritaria(df, columna_objetivo)

    if len(df) < division_minima:
        return obtener_clase_mayoritaria(df, columna_objetivo)

    ganancias = {at: ganancia_informacion(df, at, columna_objetivo) for at in atributos}
    mejor_atributo = max(ganancias, key=ganancias.get)

    if ganancias[mejor_atributo] <= ganancia_minima:
        return obtener_clase_mayoritaria(df, columna_objetivo)

    # Crear arbol
    arbol = {mejor_atributo: {}}
    atributos_restantes = [at for at in atributos if at != mejor_atributo]

    # Particionar según los valores posibles del mejor atributo
    for valor in df[mejor_atributo].unique():
        sub_df = df[df[mejor_atributo] == valor]
        if len(sub_df) == 0:
            arbol[mejor_atributo][valor] = obtener_clase_mayoritaria(df, columna_objetivo)
        else:
            arbol[mejor_atributo][valor] = construir_arbol_id3(
                sub_df, atributos_restantes, columna_objetivo,
                profundidad_max=profundidad_max,
                division_minima=division_minima,
                ganancia_minima=ganancia_minima,
                profundidad_actual=profundidad_actual + 1,
            )

    return arbol


def predecir_ejemplo(arbol, ejemplo, clase_mayoritaria):
    """
    Clasifica un ejemplo individual recorriendo las ramas del árbol ID3.
    """
    if not isinstance(arbol, dict):
        return arbol  # Hoja

    atributo = list(arbol.keys())[0]
    valor_ejemplo = ejemplo.get(atributo)

    if valor_ejemplo in arbol[atributo]:
        return predecir_ejemplo(arbol[atributo][valor_ejemplo], ejemplo, clase_mayoritaria)
    else:
        # Si aparece un valor no visto en el entrenamiento, se usa voto por mayoría
        return clase_mayoritaria


def predecir_dataset(arbol, df_prueba, clase_mayoritaria):
    """
    Genera la lista de predicciones para todo el conjunto de prueba.
    Retorna una lista con las predicciones.
    """
    predicciones = []

    for _, fila in df_prueba.iterrows():
        pred = predecir_ejemplo(arbol, fila.to_dict(), clase_mayoritaria)
        predicciones.append(pred)

    return predicciones


# VISUALIZACIÓN DEL ÁRBOL

def imprimir_arbol(arbol, prefijo="", es_ultimo=True, es_raiz=True):
    """
    Imprime el árbol en consola
    """
    conector = "" if es_raiz else ("└─ " if es_ultimo else "├─ ")

    if not isinstance(arbol, dict):
        # Nodo hoja: se imprime la clase final
        print(f"{prefijo}{conector}-> {arbol}")
        return

    # Nodo interno: un único atributo como clave
    atributo = list(arbol.keys())[0]
    ramas = arbol[atributo]

    if es_raiz:
        print(f"[{atributo}]")
        nuevo_prefijo = ""
    else:
        print(f"{prefijo}{conector}[{atributo}]")
        nuevo_prefijo = prefijo + ("    " if es_ultimo else "│   ")

    valores = list(ramas.items())
    for i, (valor, subarbol) in enumerate(valores):
        ultimo = (i == len(valores) - 1)
        rama_conector = "└─ " if ultimo else "├─ "
        print(f"{nuevo_prefijo}{rama_conector}{valor}")
        sub_prefijo = nuevo_prefijo + ("    " if ultimo else "│   ")
        imprimir_arbol(subarbol, prefijo=sub_prefijo, es_ultimo=True, es_raiz=False)



# METRICAS DE EVALUACIÓN

def calcular_matriz_confusion(y_reales, y_predichos, pos_label='OTORGADO', neg_label='RECHAZADO'):
    """
    Construye la Matriz de Confusión manual (TP, TN, FP, FN).
    Retorna un diccionario
    """
    tp, tn, fp, fn = 0, 0, 0, 0
    for real, pred in zip(y_reales, y_predichos):
        if real == pos_label and pred == pos_label:
            tp += 1
        elif real == neg_label and pred == neg_label:
            tn += 1
        elif real == neg_label and pred == pos_label:
            fp += 1
        elif real == pos_label and pred == neg_label:
            fn += 1

    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn, "TOTAL": len(y_reales)}


def calcular_accuracy(matriz):
    """ Exactitud predictiva: (TP + TN) / Total """
    total = matriz["TOTAL"]
    return (matriz["TP"] + matriz["TN"]) / total if total > 0 else 0.0


def calcular_precision(matriz):
    """ Precisión: TP / (TP + FP) """
    den = matriz["TP"] + matriz["FP"]
    return matriz["TP"] / den if den > 0 else 0.0


def calcular_recall(matriz):
    """ Sensibilidad / Exhaustividad / Recall: TP / (TP + FN) """
    den = matriz["TP"] + matriz["FN"]
    return matriz["TP"] / den if den > 0 else 0.0


def calcular_f1_score(matriz):
    """ F1-Score: 2 * (Precision * Recall) / (Precision + Recall) """
    prec = calcular_precision(matriz)
    rec = calcular_recall(matriz)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0


# EJECUCIÓN PRINCIPAL

if __name__ == "__main__":
    # Cargar datos del CSV
    data = pd.read_csv('Préstamo.csv')
    data.columns = data.columns.str.strip()

    # Filtrar personas de entre 40 y 45 años
    data_filtrada = data[(data['Edad'] >= 40) & (data['Edad'] <= 45)].copy()

    # Atributos categóricos y variable objetivo
    atributos = ['Sexo', 'Mayor nivel educativo', 'Estado de vivienda', 'Préstamos previos impagos']
    atributo_concepto = 'Estado'

    data_clean = data_filtrada[atributos + [atributo_concepto]].dropna()

    PORCENTAJE_ENTRENAMIENTO = 0.80
    set_entrenamiento = data_clean.sample(frac=PORCENTAJE_ENTRENAMIENTO, random_state=42)
    set_prueba = data_clean.drop(set_entrenamiento.index)

    print("ALGORIMO ID3 (ARBOLES DE DECISIÓN)")
    print(f"Ejemplos totales (40-45 años): {len(data_clean)}")
    print(f"Set de Entrenamiento (80%): {len(set_entrenamiento)}")
    print(f"Set de Prueba (20%): {len(set_prueba)}")

    # Hiperparámetros de PREPODA
    PROFUNDIDAD_MAX = 2
    DIVISION_MINIMA = 20
    GANANCIA_MINIMA = 0.01

    # Entrenar el modelo
    arbol_id3 = construir_arbol_id3(
        set_entrenamiento, atributos, atributo_concepto,
        profundidad_max=PROFUNDIDAD_MAX,
        division_minima=DIVISION_MINIMA,
        ganancia_minima=GANANCIA_MINIMA,
    )

    # Clase mayoritaria del SET DE ENTRENAMIENTO (para valores no vistos)
    clase_mayoritaria_train = obtener_clase_mayoritaria(set_entrenamiento, atributo_concepto)

    # Clasificacion del set de pruebas
    y_reales = set_prueba[atributo_concepto].tolist()
    y_predichos = predecir_dataset(arbol_id3, set_prueba, clase_mayoritaria_train)

    print("\nEstructura del Árbol de Decisión ID3 Generado:")
    imprimir_arbol(arbol_id3)

    # Metricas de evaluacion
    matriz = calcular_matriz_confusion(y_reales, y_predichos, pos_label='OTORGADO', neg_label='RECHAZADO')
    
    print("\n")
    print("MATRIZ DE CONFUSIÓN (Conjunto de prueba) \n")
    print(f"Verdaderos Positivos (TP): {matriz['TP']}")
    print(f"Verdaderos Negativos (TN): {matriz['TN']}")
    print(f"Falsos Positivos (FP):     {matriz['FP']}")
    print(f"Falsos Negativos (FN):     {matriz['FN']}")

    acc = calcular_accuracy(matriz)
    f1 = calcular_f1_score(matriz)
    recall = calcular_recall(matriz)
    precision = calcular_precision(matriz)
    print("\n")
    print("MÉTRICAS DE EVALUACIÓN (Conjunto de prueba) \n")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"F1-Score: {f1 * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")