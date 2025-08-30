import tensorflow as tf
from tensorflow import keras
from keras import layers

# HybridModel.py: Parallel amplitude (MLP) and temporal (GRU) branches for EMG gesture classification
# Input 1: amplitude features (shape: [batch, n_features])
# Input 2: temporal features (shape: [batch, n_steps, n_temporal_features])

def build_hybrid_model(n_features=60, n_steps=4, n_temporal_features=60, n_classes=10):
    # Amplitude branch (MLP)
    amp_input = keras.Input(shape=(n_features,), name='amplitude_features')
    x1 = layers.Dense(32, activation='relu')(amp_input)
    x1 = layers.Dropout(0.3)(x1)
    #x1 = layers.Dense(16, activation='relu')(x1)
    amp_out = layers.Dense(16, activation='relu')(x1)

    # Temporal branch (GRU)
    temp_input = keras.Input(shape=(n_steps, n_temporal_features), name='temporal_features')
    x2 = layers.GRU(16, return_sequences=False)(temp_input)
    #x2 = layers.Dense(32, activation='relu')(x2)
    temp_out = layers.Dropout(0.3)(x2)

    # Merge branches
    merged = layers.Concatenate()([amp_out, temp_out])
    x = layers.Dense(16, activation='relu')(merged)
    output = layers.Dense(n_classes, activation='softmax')(x)

    model = keras.Model(inputs=[amp_input, temp_input], outputs=output)
    return model

# Example usage:
# model = build_hybrid_model()
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()
