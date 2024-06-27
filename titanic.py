import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar los datos
data = []
with open('train.csv') as f:
    reader = csv.reader(f)
    next(reader)  # Saltar la cabecera
    for row in reader:
        data.append({
            "Pclass": float(row[2]),
            "Sex": 1.0 if row[4] == "male" else 0.0,
            "Age": float(row[5]) if row[5] else None,
            "SibSp": float(row[6]),
            "Parch": float(row[7]),
            "Fare": float(row[9]) if row[9] else None,
            "Embarked": row[11],
            "Survived": int(row[1])
        })

# Convertir datos a DataFrame para facilitar el manejo
df = pd.DataFrame(data)

# Manejo de datos faltantes
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Codificación de variables categóricas
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Seleccionar características y etiquetas
X = df.drop('Survived', axis=1).values
y = df['Survived'].values

# Escalado de características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Separar los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convertir a numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Crear la red neuronal
model = tf.keras.models.Sequential()

# Añadir las capas ocultas y la capa de salida
model.add(tf.keras.layers.Dense(8, input_shape=(X_train.shape[1],), activation='relu'))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Compilar el modelo
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Entrenar el modelo
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.3)

# Evaluar el modelo
loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f'Model accuracy: {accuracy}')
