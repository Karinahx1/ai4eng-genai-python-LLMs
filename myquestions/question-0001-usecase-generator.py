import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def generar_caso_de_uso_ajustar_regresion_polinomica():
    """
    Genera un caso de uso aleatorio para la función
    ajustar_regresion_polinomica(df, target_col, grado).
    Devuelve:
        input_data: dict con los argumentos de entrada
        output_data: tupla(predicciones, coeficientes)
    """

    n_rows = random.randint(6, 12)

    x1 = np.random.uniform(-3, 3, size=n_rows)
    x2 = np.random.uniform(-2, 2, size=n_rows)

    # Relación no lineal
    y = 2 * (x1 ** 2) + 3 * x2 + np.random.normal(0, 0.5, size=n_rows)

    df = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "target": y
    })

    grado = random.choice([2, 3])

    input_data = {
        "df": df.copy(),
        "target_col": "target",
        "grado": grado
    }

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