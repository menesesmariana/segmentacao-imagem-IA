import os
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import rasterio
from tqdm import tqdm

def create_simple_unet(input_size=(256, 256, 3)):
    """
    Cria uma versão simplificada da arquitetura U-Net.
    
    Args:
        input_size (tuple): Tamanho da imagem de entrada (altura, largura, canais)
    
    Returns:
        Model: Modelo U-Net simplificado
    """
    inputs = layers.Input(input_size)
    
    # Encoder
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Middle
    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv3)
    
    # Decoder
    up1 = layers.UpSampling2D(size=(2, 2))(conv3)
    up1 = layers.concatenate([up1, conv2], axis=3)
    conv4 = layers.Conv2D(64, 3, activation='relu', padding='same')(up1)
    conv4 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv4)
    
    up2 = layers.UpSampling2D(size=(2, 2))(conv4)
    up2 = layers.concatenate([up2, conv1], axis=3)
    conv5 = layers.Conv2D(32, 3, activation='relu', padding='same')(up2)
    conv5 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv5)
    
    # Output
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv5)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def load_dataset(rgb_dir, mask_dir, batch_size=8):
    """
    Carrega e prepara o dataset para treinamento.
    
    Args:
        rgb_dir (str): Diretório com imagens RGB
        mask_dir (str): Diretório com máscaras
        batch_size (int): Tamanho do batch
    
    Returns:
        tuple: (dataset de treino, dataset de validação)
    """
    rgb_files = sorted(list(Path(rgb_dir).glob('*.png')))
    mask_files = sorted(list(Path(mask_dir).glob('mask_*.png')))
    
    # Verificar se os arquivos correspondem
    assert len(rgb_files) == len(mask_files), "Número de imagens RGB e máscaras não corresponde"
    
    # Dividir em treino e validação (80/20)
    split_idx = int(len(rgb_files) * 0.8)
    train_rgb = rgb_files[:split_idx]
    train_mask = mask_files[:split_idx]
    val_rgb = rgb_files[split_idx:]
    val_mask = mask_files[split_idx:]
    
    def load_and_preprocess(rgb_path, mask_path):
        # Carregar imagem RGB
        with rasterio.open(str(rgb_path)) as src:
            rgb = src.read()
            # Garantir que temos apenas 3 canais (RGB)
            if rgb.shape[0] > 3:
                rgb = rgb[:3]  # Pegar apenas os primeiros 3 canais
            rgb = np.transpose(rgb, (1, 2, 0))
            rgb = rgb.astype(np.float32) / 255.0
        
        # Carregar máscara
        with rasterio.open(str(mask_path)) as src:
            mask = src.read()
            # Transpor para (altura, largura, 1)
            mask = np.transpose(mask, (1, 2, 0))
            mask = mask.astype(np.float32) / 255.0
        
        return rgb, mask
    
    def train_generator():
        for rgb_path, mask_path in zip(train_rgb, train_mask):
            rgb, mask = load_and_preprocess(rgb_path, mask_path)
            yield rgb, mask
    
    def val_generator():
        for rgb_path, mask_path in zip(val_rgb, val_mask):
            rgb, mask = load_and_preprocess(rgb_path, mask_path)
            yield rgb, mask
    
    # Criar datasets
    train_dataset = tf.data.Dataset.from_generator(
        train_generator,
        output_signature=(
            tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(256, 256, 1), dtype=tf.float32)
        )
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_generator(
        val_generator,
        output_signature=(
            tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(256, 256, 1), dtype=tf.float32)
        )
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset

def main():
    parser = argparse.ArgumentParser(
        description='Treina um modelo U-Net para segmentação de vegetação'
    )
    parser.add_argument(
        '--rgb',
        required=True,
        help='Diretório com imagens RGB'
    )
    parser.add_argument(
        '--groundtruth',
        required=True,
        help='Diretório com máscaras de ground truth'
    )
    parser.add_argument(
        '--modelpath',
        required=True,
        help='Caminho para salvar o modelo treinado'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Tamanho do batch (padrão: 8)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Número de épocas (padrão: 50)'
    )
    
    args = parser.parse_args()
    
    # Criar diretório para o modelo se não existir
    os.makedirs(os.path.dirname(args.modelpath), exist_ok=True)
    
    # Criar modelo
    model = create_simple_unet()
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Carregar dataset
    print("Carregando dataset...")
    train_dataset, val_dataset = load_dataset(args.rgb, args.groundtruth, args.batch_size)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            args.modelpath,
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]
    
    # Treinar modelo
    print("Iniciando treinamento...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks
    )
    
    print(f"Modelo salvo em: {args.modelpath}")

if __name__ == '__main__':
    main() 