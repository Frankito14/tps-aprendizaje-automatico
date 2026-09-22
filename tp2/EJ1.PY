import pandas as pd
import numpy as np
import math
# Grafico del arbol de decision
import itertools
import matplotlib.pyplot as plt

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


# ALGORITMO ID3

def construir_arbol_id3(df, atributos, columna_objetivo):
    """
    Genera un arbol de desicion usando el algortimo ID3
    - Si todos los ejemplos son de la misma clase -> Nodo hoja (Puro)
    - Si no quedan atributos -> Nodo hoja con voto por mayoría
    - Elegir atributo con MAYOR ganancia de información
    - Particionar recursivamente para cada valor del atributo
    """
    clases_unicas = df[columna_objetivo].unique()

    # Caso FIN 1: Nodo puro (todos los ejemplos pertenecen a la misma clase)
    # FIX: se devolvía el array completo (clases_unicas) en vez del valor escalar.
    if len(clases_unicas) == 1:
        return clases_unicas[0]

    # Caso FIN 2: No quedan atributos para evaluar (Voto por mayoria)
    # FIX: .mode() devuelve una Series; se necesita el valor escalar.
    if len(atributos) == 0:
        return obtener_clase_mayoritaria(df, columna_objetivo)

    # Paso 2 y 3 ID3: Calcular IG para cada atributo y elegir el máximo
    ganancias = {at: ganancia_informacion(df, at, columna_objetivo) for at in atributos}
    mejor_atributo = max(ganancias, key=ganancias.get)

    # Si la ganancia de información es 0, asignamos la clase mayoritaria
    if ganancias[mejor_atributo] <= 0:
        return obtener_clase_mayoritaria(df, columna_objetivo)

    # Estructuramos el árbol como un diccionario anidado
    arbol = {mejor_atributo: {}}
    atributos_restantes = [at for at in atributos if at != mejor_atributo]

    # Paso 4 ID3: Particionar según los valores posibles del mejor atributo
    for valor in df[mejor_atributo].unique():
        sub_df = df[df[mejor_atributo] == valor]
        if len(sub_df) == 0:
            arbol[mejor_atributo][valor] = obtener_clase_mayoritaria(df, columna_objetivo)
        else:
            arbol[mejor_atributo][valor] = construir_arbol_id3(
                sub_df, atributos_restantes, columna_objetivo
            )

    return arbol


def predecir_ejemplo(arbol, ejemplo, clase_mayoritaria):
    """
    Clasifica un ejemplo individual recorriendo las ramas del árbol ID3.
    """
    if not isinstance(arbol, dict):
        return arbol  # Es un nodo hoja (etiqueta final)

    # Extraemos el nombre del atributo (clave única del diccionario en este nivel)
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

    FIX: la clase mayoritaria usada como respaldo (para valores de atributo
    no vistos durante el entrenamiento) ahora se recibe como parámetro,
    calculada a partir del SET DE ENTRENAMIENTO. Antes se calculaba con
    las etiquetas reales del set de prueba, lo cual es una fuga de datos
    (no se puede usar la variable objetivo del set de prueba para predecir).
    """
    predicciones = []

    for _, fila in df_prueba.iterrows():
        pred = predecir_ejemplo(arbol, fila.to_dict(), clase_mayoritaria)
        predicciones.append(pred)

    return predicciones

# VISUALIZACIÓN DEL ÁRBOL
 
def imprimir_arbol(arbol, prefijo="", es_ultimo=True, es_raiz=True):
    """
    Imprime el árbol de decisión en consola con formato tipo "árbol de carpetas"
    (usando ├─, └─ y │), fácil de leer en la terminal.
 
    - Los nodos internos muestran el atributo evaluado, entre corchetes: [Atributo]
    - Las ramas muestran el valor que toma ese atributo: ── Valor
    - Las hojas muestran la clase predicha con una flecha: -> Clase
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

def _construir_layout_arbol(arbol, contador, nodos, edges, profundidad=0, x_siguiente=0.0):
    """
    Función auxiliar recursiva que recorre el árbol y le asigna a cada nodo
    una posición (x, y) para poder graficarlo: las hojas se numeran de
    izquierda a derecha y cada nodo interno se ubica en el centro (promedio)
    de sus hijos.
    Devuelve (id_del_nodo, siguiente_x_libre).
    """
    node_id = next(contador)
 
    if not isinstance(arbol, dict):
        # Nodo hoja
        nodos[node_id] = {"label": str(arbol), "x": x_siguiente, "y": -profundidad, "hoja": True}
        return node_id, x_siguiente + 1.0
 
    atributo = list(arbol.keys())[0]
    ramas = arbol[atributo]
 
    hijos_ids = []
    xs_hijos = []
    for valor, subarbol in ramas.items():
        child_id, x_siguiente = _construir_layout_arbol(
            subarbol, contador, nodos, edges, profundidad + 1, x_siguiente
        )
        hijos_ids.append(child_id)
        xs_hijos.append(nodos[child_id]["x"])
        edges.append((node_id, child_id, str(valor)))
 
    x_propio = sum(xs_hijos) / len(xs_hijos)
    nodos[node_id] = {"label": atributo, "x": x_propio, "y": -profundidad, "hoja": False}
    return node_id, x_siguiente
 
 
def graficar_arbol(arbol, ruta_salida="arbol_id3.png", colores_clase=None, titulo="Árbol de Decisión ID3"):
    """
    Genera una imagen (PNG) del árbol de decisión usando matplotlib.
    - Los nodos internos (atributos) se dibujan como cajas celestes.
    - Las hojas (clases predichas) se dibujan como cajas coloreadas según su clase.
    - Las aristas se etiquetan con el valor del atributo que representan.
 
    colores_clase: diccionario opcional {clase: color}, por ejemplo
                   {'OTORGADO': '#8FD694', 'RECHAZADO': '#F29494'}
    """
    if colores_clase is None:
        colores_clase = {}
    paleta_por_defecto = itertools.cycle(
        ["#8FD694", "#F29494", "#A9C7F2", "#F2E08F", "#D0A9F2", "#F2B48F"]
    )
 
    nodos = {}
    edges = []
    contador = itertools.count()
    _construir_layout_arbol(arbol, contador, nodos, edges)
 
    fig_ancho = max(6, 1.4 * (max(n["x"] for n in nodos.values()) + 1))
    fig_alto = max(4, 1.6 * (max(-n["y"] for n in nodos.values()) + 1))
    fig, ax = plt.subplots(figsize=(fig_ancho, fig_alto))
 
    # Dibujar aristas primero (para que queden detrás de las cajas)
    for padre_id, hijo_id, etiqueta in edges:
        p = nodos[padre_id]
        h = nodos[hijo_id]
        ax.plot([p["x"], h["x"]], [p["y"], h["y"]], color="gray", zorder=1, linewidth=1.2)
        x_medio = (p["x"] + h["x"]) / 2
        y_medio = (p["y"] + h["y"]) / 2
        ax.text(
            x_medio, y_medio, etiqueta, fontsize=8, ha="center", va="center",
            color="black", backgroundcolor="white",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none")
        )
 
    # Dibujar nodos
    for n in nodos.values():
        if n["hoja"]:
            clase = n["label"]
            if clase not in colores_clase:
                colores_clase[clase] = next(paleta_por_defecto)
            color = colores_clase[clase]
            estilo_caja = "round,pad=0.4"
        else:
            color = "#BFD7EA"
            estilo_caja = "round,pad=0.4"
 
        ax.text(
            n["x"], n["y"], n["label"], fontsize=9, ha="center", va="center",
            fontweight="bold" if not n["hoja"] else "normal",
            bbox=dict(boxstyle=estilo_caja, fc=color, ec="black", linewidth=1),
            zorder=2,
        )
 
    ax.set_title(titulo, fontsize=13, fontweight="bold")
    ax.axis("off")
    ax.set_xlim(-1, max(n["x"] for n in nodos.values()) + 1)
    ax.set_ylim(min(n["y"] for n in nodos.values()) - 1, 1)
 
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nGráfico del árbol guardado en: {ruta_salida}")
 

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

    # Entrenar el modelo
    arbol_id3 = construir_arbol_id3(set_entrenamiento, atributos, atributo_concepto)

    print("\nEstructura del Árbol de Decisión ID3 Generado:")
    imprimir_arbol(arbol_id3)
 
    # Gráfico del árbol (se guarda como imagen PNG)
    # graficar_arbol(arbol_id3,ruta_salida="arbol_id3.png",colores_clase={"OTORGADO": "#8FD694", "RECHAZADO": "#F29494"},)

    # Clase mayoritaria del SET DE ENTRENAMIENTO (para valores no vistos)
    clase_mayoritaria_train = obtener_clase_mayoritaria(set_entrenamiento, atributo_concepto)

    # Clasificacion del set de pruebas
    y_reales = set_prueba[atributo_concepto].tolist()
    y_predichos = predecir_dataset(arbol_id3, set_prueba, clase_mayoritaria_train)

    # Metricas de evaluacion
    matriz = calcular_matriz_confusion(y_reales, y_predichos, pos_label='OTORGADO', neg_label='RECHAZADO')

    print("\n")
    print("MATRIZ DE CONFUSIÓN (SET DE PRUEBA) \n")
    print(f"Verdaderos Positivos (TP): {matriz['TP']}")
    print(f"Verdaderos Negativos (TN): {matriz['TN']}")
    print(f"Falsos Positivos (FP):     {matriz['FP']}")
    print(f"Falsos Negativos (FN):     {matriz['FN']}")

    acc = calcular_accuracy(matriz)
    f1 = calcular_f1_score(matriz)

    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"F1-Score: {f1 * 100:.2f}%")