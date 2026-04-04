import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler

def generar_caso_de_uso_limpiar_duplicados_y_estandarizar():
    """
    Genera un caso de uso aleatorio (input y output esperado)
    para la función limpiar_duplicados_y_estandarizar.
    """

    # 1. Configuración aleatoria del número de filas
    n_rows = random.randint(6, 12)

    # 2. Generar datos aleatorios
    df = pd.DataFrame({
        "edad": np.random.randint(18, 60, size=n_rows),
        "ingresos": np.random.uniform(1000, 5000, size=n_rows),
        "gasto_mensual": np.random.uniform(300, 2500, size=n_rows),
        "ciudad": np.random.choice(["Bogota", "Medellin", "Cali"], size=n_rows)
    })

    # Agregar algunas filas duplicadas
    n_dups = random.randint(1, 3)
    filas_duplicadas = df.sample(
        n=n_dups,
        replace=True,
        random_state=random.randint(0, 100)
    )

    df = pd.concat([df, filas_duplicadas], ignore_index=True)

    # Mezclar filas para que los duplicados no queden necesariamente al final
    df = df.sample(frac=1, random_state=random.randint(0, 100)).reset_index(drop=True)

    # ---------------------------------------------------------
    # 3. Construir el objeto INPUT
    # ---------------------------------------------------------
    input_data = {
        "df": df.copy()
    }

    # ---------------------------------------------------------
    # 4. Calcular el OUTPUT esperado (Ground Truth)
    #    Aquí replicamos la lógica de la función a resolver
    # ---------------------------------------------------------
    df_limpio = df.drop_duplicates().reset_index(drop=True)
    df_numerico = df_limpio.select_dtypes(include=[np.number]).copy()

    scaler = StandardScaler()
    df_escalado = pd.DataFrame(
        scaler.fit_transform(df_numerico),
        columns=df_numerico.columns
    ).reset_index(drop=True)

    output_data = df_escalado

    return input_data, output_data


# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_limpiar_duplicados_y_estandarizar()

    print("=== INPUT (Diccionario) ===")
    print("DataFrame original:")
    print(entrada["df"].head())

    print("\n=== OUTPUT ESPERADO ===")
    print("DataFrame limpio y estandarizado:")
    print(salida_esperada.head())