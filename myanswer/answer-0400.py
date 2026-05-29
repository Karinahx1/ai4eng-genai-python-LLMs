import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.neural_network import MLPClassifier

def predecir_epoca_dinosaurio(df_train, df_test, target_col):
    
    X_train = df_train.drop(columns=[target_col]).values
    y_train = df_train[target_col].values
    
    X_test = df_test.drop(columns=[target_col]).values
    
    qt = QuantileTransformer(
        n_quantiles=10,
        random_state=42
    )
    
    X_train_t = qt.fit_transform(X_train)
    X_test_t = qt.transform(X_test)
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(10,),
        max_iter=200,
        random_state=42
    )
    
    mlp.fit(X_train_t, y_train)
    
    probs = mlp.predict_proba(X_test_t)
    
    return probs