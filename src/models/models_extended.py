#!/usr/bin/env python3
"""
Extended Deep Learning Models for Gait-based Deepfake Detection

Includes additional architectures:
- Bidirectional LSTM
- GRU (Gated Recurrent Unit)
- CNN-Transformer hybrid
- Attention-LSTM hybrid

Based on research: Hybrid CNN-LSTM and CNN-Transformer architectures
are state-of-the-art for gait recognition using pose keypoints.
"""

import os
import numpy as np

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


class GaitModelBuilder:
    """Builder class for all gait recognition model architectures"""
    
    def __init__(self, sequence_length=64, num_features=70, num_classes=None):
        """
        Initialize model builder.
        
        Args:
            sequence_length: Number of frames per sequence
            num_features: Number of features per frame (66 coords + 4 angles)
            num_classes: Number of output classes (None for binary)
        """
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.num_classes = num_classes if num_classes else 2
        
    def _compile_model(self, model):
        """Compile model with standard settings"""
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                     keras.metrics.Precision(name='precision'),
                     keras.metrics.Recall(name='recall')]
        )
        return model
    
    # ==================== BASIC MODELS ====================
    
    def build_lstm(self, units=[64, 32], dropout=0.3):
        """Basic LSTM model"""
        model = keras.Sequential([
            layers.Input(shape=(self.sequence_length, self.num_features)),
            layers.LSTM(units[0], return_sequences=True, dropout=dropout),
            layers.BatchNormalization(),
            layers.LSTM(units[1], dropout=dropout),
            layers.BatchNormalization(),
            layers.Dense(32, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(self.num_classes, activation='softmax')
        ], name='LSTM')
        return self._compile_model(model)
    
    def build_bilstm(self, units=[64, 32], dropout=0.3):
        """Bidirectional LSTM - captures forward and backward temporal context"""
        model = keras.Sequential([
            layers.Input(shape=(self.sequence_length, self.num_features)),
            layers.Bidirectional(layers.LSTM(units[0], return_sequences=True, dropout=dropout)),
            layers.BatchNormalization(),
            layers.Bidirectional(layers.LSTM(units[1], dropout=dropout)),
            layers.BatchNormalization(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(32, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(self.num_classes, activation='softmax')
        ], name='BiLSTM')
        return self._compile_model(model)
    
    def build_gru(self, units=[64, 32], dropout=0.3):
        """GRU model - faster training with similar performance to LSTM"""
        model = keras.Sequential([
            layers.Input(shape=(self.sequence_length, self.num_features)),
            layers.GRU(units[0], return_sequences=True, dropout=dropout),
            layers.BatchNormalization(),
            layers.GRU(units[1], dropout=dropout),
            layers.BatchNormalization(),
            layers.Dense(32, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(self.num_classes, activation='softmax')
        ], name='GRU')
        return self._compile_model(model)
    
    def build_cnn(self, filters=[32, 64, 128], kernel_size=3, dropout=0.3):
        """1D CNN for spatial pattern extraction"""
        model = keras.Sequential([
            layers.Input(shape=(self.sequence_length, self.num_features)),
            # Conv block 1
            layers.Conv1D(filters[0], kernel_size, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(dropout),
            # Conv block 2
            layers.Conv1D(filters[1], kernel_size, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(dropout),
            # Conv block 3
            layers.Conv1D(filters[2], kernel_size, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(dropout),
            # Dense
            layers.Dense(64, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(self.num_classes, activation='softmax')
        ], name='CNN')
        return self._compile_model(model)
    
    # ==================== HYBRID MODELS ====================
    
    def build_cnn_lstm(self, cnn_filters=64, lstm_units=64, dropout=0.3):
        """CNN-LSTM hybrid - spatial feature extraction + temporal modeling"""
        model = keras.Sequential([
            layers.Input(shape=(self.sequence_length, self.num_features)),
            # CNN feature extraction
            layers.Conv1D(cnn_filters, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(dropout),
            layers.Conv1D(cnn_filters*2, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Dropout(dropout),
            # LSTM temporal modeling
            layers.LSTM(lstm_units, return_sequences=True, dropout=dropout),
            layers.LSTM(lstm_units//2, dropout=dropout),
            layers.BatchNormalization(),
            # Dense
            layers.Dense(32, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(self.num_classes, activation='softmax')
        ], name='CNN_LSTM')
        return self._compile_model(model)
    
    def build_cnn_transformer(self, cnn_filters=64, num_heads=4, ff_dim=128, dropout=0.3):
        """
        CNN-Transformer hybrid - CNN for local features + Transformer for global attention.
        State-of-the-art for complex gait pattern recognition.
        """
        inputs = layers.Input(shape=(self.sequence_length, self.num_features))
        
        # CNN feature extraction
        x = layers.Conv1D(cnn_filters, 3, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(dropout)(x)
        
        x = layers.Conv1D(cnn_filters*2, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
        
        # Positional encoding (simple learned embeddings)
        seq_len = self.sequence_length // 2  # After pooling
        positions = tf.range(start=0, limit=seq_len, delta=1)
        pos_embedding = layers.Embedding(input_dim=seq_len, output_dim=cnn_filters*2)(positions)
        x = x + pos_embedding
        
        # Transformer encoder block
        # Multi-head attention
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=cnn_filters*2//num_heads, dropout=dropout
        )(x, x)
        x = layers.Add()([x, attn_output])
        x = layers.LayerNormalization()(x)
        
        # Feed-forward network
        ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(cnn_filters*2)
        ])
        ffn_output = ffn(x)
        x = layers.Add()([x, ffn_output])
        x = layers.LayerNormalization()(x)
        
        # Global pooling and classification
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='CNN_Transformer')
        return self._compile_model(model)
    
    def build_attention_lstm(self, lstm_units=64, dropout=0.3):
        """
        Attention-LSTM hybrid - Self-attention mechanism + LSTM.
        Attention helps focus on important frames in the gait cycle.
        """
        inputs = layers.Input(shape=(self.sequence_length, self.num_features))
        
        # First LSTM layer
        x = layers.LSTM(lstm_units, return_sequences=True, dropout=dropout)(inputs)
        x = layers.BatchNormalization()(x)
        
        # Self-attention mechanism
        attention = layers.MultiHeadAttention(
            num_heads=4, key_dim=lstm_units//4, dropout=dropout
        )(x, x)
        x = layers.Add()([x, attention])
        x = layers.LayerNormalization()(x)
        
        # Second LSTM layer
        x = layers.LSTM(lstm_units//2, dropout=dropout)(x)
        x = layers.BatchNormalization()(x)
        
        # Classification
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='Attention_LSTM')
        return self._compile_model(model)
    
    # ==================== UTILITY ====================
    
    def build_all_models(self):
        """Build all available model architectures"""
        models = {
            'LSTM': self.build_lstm(),
            'BiLSTM': self.build_bilstm(),
            'GRU': self.build_gru(),
            'CNN': self.build_cnn(),
            'CNN_LSTM': self.build_cnn_lstm(),
            'CNN_Transformer': self.build_cnn_transformer(),
            'Attention_LSTM': self.build_attention_lstm()
        }
        return models
    
    def get_model_summary(self):
        """Get summary of all model architectures"""
        models = self.build_all_models()
        print("\n" + "="*70)
        print("MODEL ARCHITECTURES SUMMARY")
        print("="*70)
        for name, model in models.items():
            params = model.count_params()
            print(f"{name:20} | Parameters: {params:,}")
        print("="*70)
        return models


# ==================== CONVENIENCE FUNCTIONS ====================

def create_model(model_type, sequence_length=64, num_features=70, num_classes=2):
    """
    Factory function to create a specific model.
    
    Args:
        model_type: One of 'LSTM', 'BiLSTM', 'GRU', 'CNN', 'CNN_LSTM', 
                   'CNN_Transformer', 'Attention_LSTM'
        sequence_length: Number of frames
        num_features: Features per frame
        num_classes: Output classes
    
    Returns:
        Compiled Keras model
    """
    builder = GaitModelBuilder(sequence_length, num_features, num_classes)
    
    model_map = {
        'LSTM': builder.build_lstm,
        'BiLSTM': builder.build_bilstm,
        'GRU': builder.build_gru,
        'CNN': builder.build_cnn,
        'CNN_LSTM': builder.build_cnn_lstm,
        'CNN_Transformer': builder.build_cnn_transformer,
        'Attention_LSTM': builder.build_attention_lstm
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available: {list(model_map.keys())}")
    
    return model_map[model_type]()


if __name__ == "__main__":
    # Demo: show all model architectures
    print("TensorFlow version:", tf.__version__)
    print("GPU Available:", tf.config.list_physical_devices('GPU'))
    
    builder = GaitModelBuilder(sequence_length=64, num_features=70, num_classes=2)
    models = builder.get_model_summary()
    
    # Show detailed summary for one model
    print("\n\nDetailed CNN-Transformer Architecture:")
    print("-" * 50)
    models['CNN_Transformer'].summary()
