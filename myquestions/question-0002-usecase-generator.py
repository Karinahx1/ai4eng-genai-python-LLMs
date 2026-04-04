import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LinearRegression

def generar_caso_de_uso_resumir_ventas_y_tendencia():
    """
    Genera un caso de uso aleatorio (input y output esperado)
    para la función resumir_ventas_y_tendencia.
    """

    # 1. Configuración aleatoria del número de días
    n_dias = random.randint(60, 120)
    fechas = pd.date_range("2024-01-01", periods=n_dias, freq="D")

    # 2. Generar ventas aleatorias con cierta tendencia
    base = random.randint(50, 120)
    tendencia = random.uniform(-1.5, 2.0)

    ventas = []
    for i in range(n_dias):
        valor = base + tendencia * i + np.random.normal(0, 5)
        ventas.append(max(1, float(valor)))

    df = pd.DataFrame({
        "fecha": fechas,
        "ventas": ventas
    })

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
    temp = df.copy()
    temp["fecha"] = pd.to_datetime(temp["fecha"])
    temp["mes"] = temp["fecha"].dt.to_period("M").astype(str)

    resumen = (
        temp.groupby("mes", as_index=False)["ventas"]
        .sum()
        .reset_index(drop=True)
    )

    X = np.arange(len(resumen)).reshape(-1, 1)
    y = resumen["ventas"].to_numpy()

    model = LinearRegression()
    model.fit(X, y)

    pendiente = float(model.coef_[0])

    output_data = (resumen, pendiente)

    return input_data, output_data


# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_resumir_ventas_y_tendencia()

    print("=== INPUT (Diccionario) ===")
    print("DataFrame (primeras 5 filas):")
    print(entrada["df"].head())

    print("\n=== OUTPUT ESPERADO ===")
    resumen, pendiente = salida_esperada
    print("Resumen mensual:")
    print(resumen)
    print(f"\nPendiente de la regresión: {pendiente}")