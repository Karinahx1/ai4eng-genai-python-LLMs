import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def generar_caso_de_uso_ajustar_regresion_polinomica():
    """
    Genera un caso de uso aleatorio (input y output esperado)
    para la función ajustar_regresion_polinomica.
    """

    # 1. Configuración aleatoria del tamaño del problema
    n_rows = random.randint(6, 12)

    # 2. Generar datos aleatorios
    x1 = np.random.uniform(-3, 3, size=n_rows)
    x2 = np.random.uniform(-2, 2, size=n_rows)

    # Creamos una relación no lineal simple para la variable objetivo
    y = 2 * (x1 ** 2) + 3 * x2 + np.random.normal(0, 0.5, size=n_rows)

    df = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "target": y
    })

    grado = random.choice([2, 3])

    # ---------------------------------------------------------
    # 3. Construir el objeto INPUT
    # ---------------------------------------------------------
    input_data = {
        "df": df.copy(),
        "target_col": "target",
        "grado": grado
    }

    # ---------------------------------------------------------
    # 4. Calcular el OUTPUT esperado (Ground Truth)
    #    Aquí replicamos la lógica de la función a resolver
    # ---------------------------------------------------------
    X = df.drop(columns=["target"])
    y_expected = df["target"].to_numpy()

    poly = PolynomialFeatures(degree=grado, include_bias=False)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y_expected)

    predicciones = model.predict(X_poly)
    coeficientes = model.coef_

    output_data = (predicciones, coeficientes)

    return input_data, output_data


# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_ajustar_regresion_polinomica()

    print("=== INPUT (Diccionario) ===")
    print(f"Target Column: {entrada['target_col']}")
    print(f"Grado polinómico: {entrada['grado']}")
    print("DataFrame (primeras 5 filas):")
    print(entrada["df"].head())

    print("\n=== OUTPUT ESPERADO ===")
    predicciones, coeficientes = salida_esperada
    print(f"Shape de predicciones: {predicciones.shape}")
    print(f"Shape de coeficientes: {coeficientes.shape}")
    print("Ejemplo de primeras predicciones:", predicciones[:3])