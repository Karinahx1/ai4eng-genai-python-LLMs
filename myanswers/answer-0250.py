import numpy as np
from sklearn.ensemble import RandomForestClassifier

def seleccionar_y_predecir(X, y, k_top):
    modelo_1 = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo_1.fit(X, y)
    
    importancias = modelo_1.feature_importances_
    indices_top = np.argsort(importancias)[-k_top:]
    
    X_filtrado = X[:, indices_top]
    
    modelo_2 = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo_2.fit(X_filtrado, y)
    
    return modelo_2.predict(X_filtrado)