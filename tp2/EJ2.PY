import pandas as pd
import numpy as np
import math
import random

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
    Devuelve la clase mayoritaria como valor escalar (no como Series),
    resolviendo empates tomando la primera en orden alfabético/orden de aparición.
    """
    return df[columna_objetivo].mode().iloc[0]


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
    """
    predicciones = []

    for _, fila in df_prueba.iterrows():
        pred = predecir_ejemplo(arbol, fila.to_dict(), clase_mayoritaria)
        predicciones.append(pred)

    return predicciones


def generar_muestra_bootstrap(df, random_state=None):
   
    return df.sample(n=len(df), replace=True, random_state=random_state)


def muestrear_atributos_aleatorios(atributos_disponibles, num_atributos_muestreados):

    cantidad_a_tomar = min(num_atributos_muestreados, len(atributos_disponibles))

    return random.sample(atributos_disponibles, cantidad_a_tomar)


def construir_arbol_id3_aleatorio(df, atributos, columna_objetivo,
                                   num_atributos_muestreados, tamano_nodo=1,
                                   profundidad_actual=0):
    """
    Construir un arbol id3 sin poda.
    """

    #Clases unicas
    clases_unicas = df[columna_objetivo].unique()

    # Criterios de corte:
    # Solo queda una clase en el nodo
    if len(clases_unicas) == 1:
        return clases_unicas[0]

    # No quedan atributos para seguir 
    if len(atributos) == 0:
        return obtener_clase_mayoritaria(df, columna_objetivo)

    # Los ejemplos restantes son menores al tamaño elegido
    if len(df) < tamano_nodo:
        return obtener_clase_mayoritaria(df, columna_objetivo)

    # Subconjunto random de atributos candidatos para este split
    atributos_candidatos = muestrear_atributos_aleatorios(atributos, num_atributos_muestreados)

    ganancias = {at: ganancia_informacion(df, at, columna_objetivo) for at in atributos_candidatos}
    mejor_atributo = max(ganancias, key=ganancias.get)

    # Si ni el mejor atributo del subconjunto aporta ganancia, no tiene sentido dividir
    if ganancias[mejor_atributo] <= 0:
        return obtener_clase_mayoritaria(df, columna_objetivo)

    arbol = {mejor_atributo: {}}
    atributos_restantes = [at for at in atributos if at != mejor_atributo]

    for valor in df[mejor_atributo].unique():
        sub_df = df[df[mejor_atributo] == valor]
        if len(sub_df) == 0:
            arbol[mejor_atributo][valor] = obtener_clase_mayoritaria(df, columna_objetivo)
        else:
            arbol[mejor_atributo][valor] = construir_arbol_id3_aleatorio(
                sub_df, atributos_restantes, columna_objetivo,
                num_atributos_muestreados=num_atributos_muestreados,
                tamano_nodo=tamano_nodo,
                profundidad_actual=profundidad_actual + 1,
            )

    return arbol


def entrenar_bosque_aleatorio(df, atributos, columna_objetivo,
                               numero_arboles=100, num_atributos_muestreados=None,
                               tamano_nodo=1, semilla=42):
    """
    Entrena un Random Forest y retorna un bosque aleatorio
    - numero_arboles: cantidad de árboles del bosque.
    - num_atributos_muestreados: cantidad de atributos a sortear. Si es None, se usa la raiz.
    - tamano_nodo: tamaño mínimo de nodo para seguir dividiendo
    - semilla: seed para que siempre quede igual y hacer pruebas
    """
    random.seed(semilla)

    if num_atributos_muestreados is None:
        num_atributos_muestreados = max(1, round(math.sqrt(len(atributos))))

    bosque = []
    for i in range(numero_arboles):
        # BAGGING: una muestra bootstrap distinta para cada árbol
        muestra_bootstrap = generar_muestra_bootstrap(df, random_state=semilla + i)
        # ENTRENAMIENTO: cada arbol se entrena con su propia muestra
        arbol = construir_arbol_id3_aleatorio(
            muestra_bootstrap, atributos, columna_objetivo,
            num_atributos_muestreados=num_atributos_muestreados,
            tamano_nodo=tamano_nodo,
        )
        bosque.append(arbol)

    return bosque


def votar_ejemplo_bosque(bosque, ejemplo, clase_mayoritaria):
    """
    Realizar la votacion del bosque ne base a un ejemplo.

    Retorna (clase_ganadora, votos), donde votos es un
    diccionario {clase: cantidad_de_arboles_que_votaron_por_esa_clase}.
    """
    votos = {}
    for arbol in bosque:
        prediccion = predecir_ejemplo(arbol, ejemplo, clase_mayoritaria)
        votos[prediccion] = votos.get(prediccion, 0) + 1

    clase_ganadora = max(votos, key=votos.get)
    return clase_ganadora, votos


def predecir_dataset_bosque(bosque, df_prueba, clase_mayoritaria):
    """
    Genera las predicciones del Random Forest para todo el set de prueba.

    Retorna (predicciones, votos_por_ejemplo):
    - predicciones: lista con la clase ganadora para cada ejemplo.
    - votos_por_ejemplo: lista de diccionarios {clase: cantidad_de_votos},
    """
    predicciones = []
    votos_por_ejemplo = []

    for _, fila in df_prueba.iterrows():
        ejemplo = fila.to_dict()
        clase_ganadora, votos = votar_ejemplo_bosque(bosque, ejemplo, clase_mayoritaria)
        predicciones.append(clase_ganadora)
        votos_por_ejemplo.append(votos)

    return predicciones, votos_por_ejemplo


# IMPRIMIR RESULTADO DEL BOSQUE
def imprimir_votacion_ejemplo(bosque, ejemplo, clase_mayoritaria, indice=None,
                               maximo_arboles_a_listar=20):
    """
    Mostrar en consola el resultado de la votacion arbol x arbol
    """
    votos = {}
    votos_individuales = []
    for arbol in bosque:
        prediccion = predecir_ejemplo(arbol, ejemplo, clase_mayoritaria)
        votos[prediccion] = votos.get(prediccion, 0) + 1
        votos_individuales.append(prediccion)

    titulo = f"Votación del bosque para el ejemplo #{indice}" if indice is not None else "Votación del bosque"
    print(f"\n{titulo}")
    print("-" * len(titulo))

    for i, prediccion in enumerate(votos_individuales[:maximo_arboles_a_listar], start=1):
        print(f"Árbol {i:>3}: {prediccion}")
    if len(votos_individuales) > maximo_arboles_a_listar:
        restantes = len(votos_individuales) - maximo_arboles_a_listar
        print(f"({restantes} árboles más, ver resumen abajo)")

    total = sum(votos.values())
    clase_ganadora = max(votos, key=votos.get)
    print("\n  Resumen de la votación:")
    for clase, cantidad in sorted(votos.items(), key=lambda item: -item[1]):
        print(f"    {clase}: {cantidad}/{total} votos ({100 * cantidad / total:.1f}%)")
    print(f"Predicción final del bosque: {clase_ganadora}")

    return clase_ganadora, votos


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
 
    print("=" * 60)
    print("ALGORIMO RANDOM FOREST")
    print("=" * 60)
    print(f"Ejemplos totales (40-45 años): {len(data_clean)}")
    print(f"Set de Entrenamiento (80%): {len(set_entrenamiento)}")
    print(f"Set de Prueba (20%): {len(set_prueba)}")
 
    # Sin poda configurada
    PROFUNDIDAD_MAX = None     
    DIVISION_MINIMA = 2         
    GANANCIA_MINIMA = 0.0       
 
    # RANDOM FOREST
  
    # Hiperparámetros del Random Forest
    NUMERO_ARBOLES = 10            # cantidad de árboles del bosque
    NUM_ATRIBUTOS_MUESTREADOS = None  # None -> se usa round(sqrt(cant. atributos)) = 2
    TAMANO_NODO = 1                 # tamaño mínimo de nodo (árboles sin podar)
 
    print("\n" + "=" * 60)
    print("RANDOM FOREST ")
    print("=" * 60)
    print(f"Número de árboles: {NUMERO_ARBOLES}")
    mtry_usado = NUM_ATRIBUTOS_MUESTREADOS or max(1, round(math.sqrt(len(atributos))))
    print(f"Atributos muestreados por división (mtry): {mtry_usado} de {len(atributos)} totales")
    print(f"Tamaño mínimo de nodo: {TAMANO_NODO}")
 
    bosque_id3 = entrenar_bosque_aleatorio(
        set_entrenamiento, atributos, atributo_concepto,
        numero_arboles=NUMERO_ARBOLES,
        num_atributos_muestreados=NUM_ATRIBUTOS_MUESTREADOS,
        tamano_nodo=TAMANO_NODO,
        semilla=42,
    )

    # Clase mayoritaria del SET DE ENTRENAMIENTO (para valores no vistos)
    clase_mayoritaria_train = obtener_clase_mayoritaria(set_entrenamiento, atributo_concepto)

    # Clasificacion del set de pruebas
    y_reales = set_prueba[atributo_concepto].tolist()
    
    y_predichos_bosque, votos_por_ejemplo = predecir_dataset_bosque(
        bosque_id3, set_prueba, clase_mayoritaria_train
    )
 
    matriz_bosque = calcular_matriz_confusion(
        y_reales, y_predichos_bosque, pos_label='OTORGADO', neg_label='RECHAZADO'
    )

    print("\n" + "=" * 60)
    print("MATRIZ DE CONFUSIÓN (Set de prueba, Random Forest)")
    print("=" * 60)
    print(f"Verdaderos Positivos (TP): {matriz_bosque['TP']}")
    print(f"Verdaderos Negativos (TN): {matriz_bosque['TN']}")
    print(f"Falsos Positivos (FP):     {matriz_bosque['FP']}")
    print(f"Falsos Negativos (FN):     {matriz_bosque['FN']}")
 
    acc_bosque = calcular_accuracy(matriz_bosque)
    prec_bosque = calcular_precision(matriz_bosque)
    rec_bosque = calcular_recall(matriz_bosque)
    f1_bosque = calcular_f1_score(matriz_bosque)
 
    print(f"\nAccuracy: {acc_bosque * 100:.2f}%")
    print(f"F1-Score: {f1_bosque * 100:.2f}%")
    print(f"Recall: {rec_bosque * 100:.2f}%")
    print(f"Precision: {prec_bosque * 100:.2f}%")
 
    # Votación detallada del bosque para el primer ejemplo del set de prueba
    primer_ejemplo = set_prueba.iloc[0].to_dict()
    imprimir_votacion_ejemplo(bosque_id3, primer_ejemplo, clase_mayoritaria_train, indice=0)
