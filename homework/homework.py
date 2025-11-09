# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#



import os
import pandas as pd
import gzip
import pickle
import json

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Rutas 

dir_input_test = "files/input/test_data.csv.zip"
dir_input_train = "files/input/train_data.csv.zip"
dir_models = "files/models/model.pkl.gz"
dir_metrics = "files/output/metrics.json"

# Paso 1: Cargar y limpiar datos

def load_data():
    df_train = pd.read_csv(dir_input_train, compression='zip')
    df_test = pd.read_csv(dir_input_test, compression='zip')
    return df_train, df_test

def clean_data(df):
    df = df.rename(columns={"default payment next month": "default"})
    df = df.drop(columns=["ID"])

    df = df[df['EDUCATION'] != 0]
    df = df[df['MARRIAGE'] != 0]
    
    df = df.dropna()

    df['EDUCATION'] = df['EDUCATION'].apply(lambda x: 4 if x > 4 else x)
    
    return df

df_train, df_test = load_data()
df_train = clean_data(df_train)
df_test = clean_data(df_test)

print("Paso 1: Datos cargados y limpiados")
print(f"{df_train.head()}\n\n{df_test.head()}")


# Paso 2: Separar en x_train, y_train, x_test, y_test

def split_data(df_train, df_test):
    x_train = df_train.drop(columns=["default"])
    y_train = df_train["default"]
    x_test = df_test.drop(columns=["default"])
    y_test = df_test["default"]
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = split_data(df_train, df_test)
print("Paso 2: Datos separados")


# Paso 3: Crear pipeline de modelo

categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE',
                    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
numeric_cols = [c for c in x_train.columns if c not in categorical_cols]
preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numeric_cols),],
    remainder='drop')

def create_pipeline():
    pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('rf', RandomForestClassifier(random_state=42))
    ])
    return pipeline

pipeline = create_pipeline()

print("Paso 3: Pipeline creado")


# Paso 4: Optimizar hiperparámetros

def optimize_hyperparameters(pipeline, x_train, y_train):
    param_grid = {
        'rf__n_estimators': [400, 600], 
        'rf__max_depth': [20, 30],      
        'rf__min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10, 
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(x_train, y_train)
    print("Mejores parámetros:", grid_search.best_params_)
    print("Mejor balanced_accuracy:", grid_search.best_score_)
    return grid_search

best_pipeline = optimize_hyperparameters(pipeline, x_train, y_train)
print("Paso 4: Hiperparámetros optimizados")


# Paso 5: Guardar modelo

def save_model(model, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with gzip.open(filepath, 'wb') as f:
        pickle.dump(model, f)

save_model(best_pipeline,  dir_models)
print(f"Paso 5: Modelo guardado en {dir_models}")


# Paso 6: Calcular métricas y guardar en metrics.json

def calculate_metrics(model, x, y, dataset_type):
    y_pred = model.predict(x)
    precision = precision_score(y, y_pred, zero_division=0) 
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    balanced_acc = balanced_accuracy_score(y, y_pred)

    return {
        'type': 'metrics', 
        'dataset': dataset_type,
        'precision': round(precision, 4),
        'balanced_accuracy': round(balanced_acc, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4)
    }

metrics = []
metrics.append(calculate_metrics(best_pipeline, x_train, y_train, 'train'))
metrics.append(calculate_metrics(best_pipeline, x_test, y_test, 'test'))

os.makedirs(os.path.dirname(dir_metrics), exist_ok=True)
with open(dir_metrics, 'w') as f:
    for metric in metrics:
        f.write(json.dumps(metric) + '\n')

print(f"Paso 6: Métricas guardadas en {dir_metrics}")


# Paso 7: Matrices de confusión

def calculate_confusion_matrix(model, x, y, dataset_type):
    y_pred = model.predict(x)
    cm = confusion_matrix(y, y_pred)

    return {
        'type': 'cm_matrix',
        'dataset': dataset_type,
        'true_0': {
            'predicted_0': int(cm[0, 0]),
            'predicted_1': int(cm[0, 1])
        },
        'true_1': {
            'predicted_0': int(cm[1, 0]),
            'predicted_1': int(cm[1, 1])
        }
    }

cm_train = calculate_confusion_matrix(best_pipeline, x_train, y_train, 'train')
cm_test = calculate_confusion_matrix(best_pipeline, x_test, y_test, 'test')

with open(dir_metrics, 'a') as f:
    f.write(json.dumps(cm_train) + '\n')
    f.write(json.dumps(cm_test) + '\n')

print(f"Paso 7: Matrices de confusión guardadas en {dir_metrics}")
print("Proceso completado.")