import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from pathlib import Path
import os

# Carregando o dataset
BASE_DIR = "D:/project_hub/Flower Recognization"
TRAIN_DIR = os.path.join(BASE_DIR, "processed", "train")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Carregando os dados
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    color_mode="rgb",
    seed=123
)

# Normalização dos dados
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

def compute_loss(model, x, y):
    # Calcula a função de perda (cross-entropy)
    predictions = model(x, training=False)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions)
    return tf.reduce_mean(loss)

def compute_gradients(model, x, y):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x, y)
    return tape.gradient(loss, model.trainable_variables)

def visualize_gradient_descent(model, x_batch, y_batch, lr, steps=10):
    """
    Visualiza o caminho do gradiente descendente para um único batch
    sem realmente atualizar os pesos do modelo
    """
    history = []
    current_weights = [w.numpy() for w in model.trainable_variables]
    
    for step in range(steps):
        # Calcula gradientes
        gradients = compute_gradients(model, x_batch, y_batch)
        
        # Calcula loss atual
        loss = compute_loss(model, x_batch, y_batch)
        history.append(loss.numpy())
        
        print(f"Passo {step + 1}, Loss: {loss:.4f}")
        
        # Simula o movimento do gradiente descendente
        for i, (weight, grad) in enumerate(zip(current_weights, gradients)):
            current_weights[i] = weight - lr * grad.numpy()
    
    return history

# Carregando o modelo existente
model_path = os.path.join(BASE_DIR, 'models', 'modelo_flores.keras')
try:
    model = tf.keras.models.load_model(model_path)
    print("Modelo carregado com sucesso!")
except:
    print(f"Tentando carregar modelo .h5...")
    model_path = os.path.join(BASE_DIR, 'models', 'modelo_flores.h5')
    try:
        model = tf.keras.models.load_model(model_path)
        print("Modelo .h5 carregado com sucesso!")
    except:
        print(f"Erro: Modelo não encontrado em {model_path}")
        print("Por favor, execute o notebook primeiro para treinar e salvar o modelo.")
        exit()

# Pegando um batch de dados para demonstração
for x_batch, y_batch in train_ds.take(1):
    break

# Visualizando o caminho do gradiente descendente
history = visualize_gradient_descent(model, x_batch, y_batch, lr=0.001, steps=10)

# Visualizando a convergência
plt.figure(figsize=(10, 6))
plt.plot(history, 'b-', label='Loss')
plt.grid(True)
plt.legend()
plt.title('Demonstração do Gradiente Descendente')
plt.xlabel('Iteração')
plt.ylabel('Loss')
plt.show()