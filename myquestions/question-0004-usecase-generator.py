import pandas as pd
import numpy as np
import random
from sklearn.feature_selection import VarianceThreshold

def generar_caso_de_uso_eliminar_baja_varianza():
    """
    Genera un caso de uso aleatorio (input y output esperado)
    para la función eliminar_baja_varianza.
    """

    # 1. Configuración aleatoria del número de filas
    n_rows = random.randint(6, 12)

    # 2. Generar columnas con distinta varianza
    col_1 = np.random.normal(0, 1.0, size=n_rows)
    col_2 = np.random.normal(5, 0.1, size=n_rows)    # baja varianza
    col_3 = np.random.normal(10, 2.0, size=n_rows)
    col_4 = np.random.normal(-3, 0.05, size=n_rows)  # muy baja varianza
    col_cat = np.random.choice(["A", "B", "C"], size=n_rows)

    df = pd.DataFrame({
        "feature_1": col_1,
        "feature_2": col_2,
        "feature_3": col_3,
        "feature_4": col_4,
        "categoria": col_cat
    })

    umbral = random.choice([0.01, 0.05, 0.1, 0.2])

    # ---------------------------------------------------------
    # 3. Construir el objeto INPUT
    # ---------------------------------------------------------
    input_data = {
        "df": df.copy(),
        "umbral": umbral
    }

    # ---------------------------------------------------------
    # 4. Calcular el OUTPUT esperado (Ground Truth)
    #    Aquí replicamos la lógica de la función a resolver
    # ---------------------------------------------------------
    df_numerico = df.select_dtypes(include=[np.number]).copy()

    selector = VarianceThreshold(threshold=umbral)
    X_sel = selector.fit_transform(df_numerico)

    columnas_seleccionadas = df_numerico.columns[selector.get_support()]
    df_resultado = pd.DataFrame(
        X_sel,
        columns=columnas_seleccionadas,
        index=df.index
    )

    output_data = df_resultado

    return input_data, output_data


# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_eliminar_baja_varianza()

    print("=== INPUT (Diccionario) ===")
    print(f"Umbral de varianza: {entrada['umbral']}")
    print("DataFrame original:")
    print(entrada["df"].head())

    print("\n=== OUTPUT ESPERADO ===")
    print("DataFrame con columnas seleccionadas:")
    print(salida_esperada.head())