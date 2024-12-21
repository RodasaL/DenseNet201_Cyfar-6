# -*- coding: utf-8 -*-
"""
Projeto Fase III - Aprendizagem por Transferência e Deployment (DenseNet201)
"""

import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score, roc_auc_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configurações do modelo e hiperparâmetros
NUM_CLASSES = 6
BATCH_SIZE = 32
EPOCHS = 25
DATA_DIR = r'D:\OneDriveIsec\OneDrive - ISEC\IC\files'
LEARNING_RATE = 1e-4
DENSE_UNITS = 256
DROPOUT_RATE = 0.5

# 1. Funções de pré-processamento e carregamento de dados
def load_cifar_batch(file):
    with open(file, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
        images = batch['data']
        labels = batch['labels']
        images = images.reshape(-1, 3, 32, 32)
        images = np.transpose(images, (0, 2, 3, 1))
        return images, labels

def load_cifar10(data_dir):
    x_train, y_train = [], []
    for i in range(1, 6):  # Incluindo os 5 batches de treino
        batch_file = os.path.join(data_dir, f'data_batch_{i}')
        images, labels = load_cifar_batch(batch_file)
        x_train.append(images)
        y_train += labels
    x_train = np.concatenate(x_train)
    y_train = np.array(y_train)
    test_file = os.path.join(data_dir, 'test_batch')
    x_test, y_test = load_cifar_batch(test_file)
    return x_train, y_train, x_test, y_test

def normalize_data(x):
    return x.astype('float32') / 255.0

def filter_classes(x, y, classes):
    x = np.array(x)
    y = np.array(y)
    mask = np.isin(y, classes)
    return x[mask], y[mask]

def resize_images(images, target_size=(128, 128)):  # Ajustado para 128x128 para DenseNet201
    return np.array([tf.image.resize(image, target_size).numpy() for image in images])

# 2. Modelo pré-treinado
def build_model(learning_rate=1e-4, dense_units=256, dropout_rate=0.5):
    base_model = tf.keras.applications.DenseNet201(
        weights='imagenet',
        include_top=False,  # Remove a cabeça densa para customização
        input_shape=(128, 128, 3)  # Ajustado para DenseNet201
    )

    for layer in base_model.layers[:-20]:  # Torna as últimas 20 camadas treináveis
        layer.trainable = False

    for layer in base_model.layers[-20:]:
        layer.trainable = True

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(dense_units, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(dropout_rate),
        layers.Dense(dense_units // 2, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(dropout_rate),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 3. Treinamento e avaliação
def train_model(model, x_train, y_train, x_val, y_val, epochs, batch_size):
    datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    train_data_generator = datagen.flow(x_train, y_train, batch_size=batch_size)
    return model.fit(train_data_generator, epochs=epochs, validation_data=(x_val, y_val), verbose=1)

def save_results_to_excel(history, y_true, y_pred, filename, class_labels):
    results = pd.DataFrame({
        "epoch": list(range(1, len(history.history['loss']) + 1)),
        "train_loss": history.history['loss'],
        "val_loss": history.history['val_loss'],
        "train_accuracy": history.history['accuracy'],
        "val_accuracy": history.history['val_accuracy'],
        "learning_rate": [LEARNING_RATE] * len(history.history['loss']),
        "dense_units": [DENSE_UNITS] * len(history.history['loss']),
        "dropout_rate": [DROPOUT_RATE] * len(history.history['loss'])
    })

    # Calcula métricas para o conjunto de teste
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    auc = roc_auc_score(pd.get_dummies(y_true), pd.get_dummies(y_pred), multi_class='ovr')
    accuracy = accuracy_score(y_true, y_pred)

    results_test = {
        "class_label": class_labels,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": [accuracy] * len(class_labels),
        "AUC": [auc] * len(class_labels),
        "learning_rate": [LEARNING_RATE] * len(class_labels),
        "dense_units": [DENSE_UNITS] * len(class_labels),
        "dropout_rate": [DROPOUT_RATE] * len(class_labels)
    }

    results_test_df = pd.DataFrame(results_test)

    with pd.ExcelWriter(filename) as writer:
        results.to_excel(writer, sheet_name="Train_Validation_Results", index=False)
        results_test_df.to_excel(writer, sheet_name="Test_Results", index=False)

def plot_confusion_matrix(y_true, y_pred, class_labels):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Matriz de Confusão')
    plt.show()

def plot_training_validation_performance(history):
    epochs = range(1, len(history.history['accuracy']) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history.history['accuracy'], label='Training Accuracy', color='blue')
    plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.plot(epochs, history.history['loss'], label='Training Loss', linestyle='--', color='blue')
    plt.plot(epochs, history.history['val_loss'], label='Validation Loss', linestyle='--', color='orange')

    plt.title('Training and Validation Performance')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

# Fluxo principal
model = build_model(learning_rate=LEARNING_RATE, dense_units=DENSE_UNITS, dropout_rate=DROPOUT_RATE)

x_train, y_train, x_test, y_test = load_cifar10(DATA_DIR)
classes = [2, 3, 4, 5, 6, 7]
x_train, y_train = filter_classes(x_train, y_train, classes)
x_test, y_test = filter_classes(x_test, y_test, classes)

class_mapping = {original: new for new, original in enumerate(classes)}
y_train = np.array([class_mapping[label] for label in y_train])
y_test = np.array([class_mapping[label] for label in y_test])
class_labels = ["passaro", "gato", "veado", "cao", "sapo", "cavalo"]

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
x_train = normalize_data(resize_images(x_train))
x_val = normalize_data(resize_images(x_val))
x_test = normalize_data(resize_images(x_test))

history = train_model(model, x_train, y_train, x_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE)

plot_training_validation_performance(history)
y_pred = np.argmax(model.predict(x_test), axis=1)
plot_confusion_matrix(y_test, y_pred, class_labels)

final_accuracy = accuracy_score(y_test, y_pred)
print(f"Final Test Accuracy: {final_accuracy:.2f}")

save_results_to_excel(history, y_test, y_pred, "results_densenet201.xlsx", class_labels)
model.save("animal_classifier_model_densenet201.h5")
