from sklearn.linear_model import LogisticRegression

def entrenar_y_predecir_clasificador(X, y, X_nuevo):

    modelo = LogisticRegression(max_iter=1000)

    modelo.fit(X, y)

    return modelo.predict(X_nuevo)