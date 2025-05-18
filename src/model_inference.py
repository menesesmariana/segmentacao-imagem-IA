import os
import argparse
import numpy as np
import tensorflow as tf
import rasterio
from pathlib import Path
from tqdm import tqdm
import sys
import tempfile
import shutil

def load_model(model_path):
    """
    Carrega o modelo com tratamento especial para caminhos com caracteres especiais
    """
    try:
        # Tenta criar um arquivo temporário sem caracteres especiais
        temp_dir = tempfile.gettempdir()
        temp_model_path = os.path.join(temp_dir, 'temp_model.h5')
        
        # Copia o modelo para o diretório temporário
        shutil.copy2(str(model_path), temp_model_path)
        
        # Carrega o modelo do arquivo temporário
        model = tf.keras.models.load_model(temp_model_path)
        
        # Remove o arquivo temporário
        try:
            os.remove(temp_model_path)
        except:
            pass
            
        return model
    except Exception as e:
        print(f"\nErro ao carregar o modelo: {str(e)}")
        print("\nSugestões:")
        print("1. Verifique se o caminho do modelo está correto")
        print("2. Tente mover o modelo para um caminho sem caracteres especiais")
        print("3. Verifique se o arquivo do modelo não está corrompido")
        sys.exit(1)

def process_image_in_blocks(image_path, model, block_size=256, overlap=32):
    """
    Processa uma imagem grande em blocos usando o modelo.
    
    Args:
        image_path (str): Caminho para a imagem
        model (Model): Modelo carregado
        block_size (int): Tamanho dos blocos
        overlap (int): Sobreposição entre blocos
    
    Returns:
        np.ndarray: Imagem segmentada
    """
    with rasterio.open(image_path) as src:
        # Obter dimensões da imagem
        height = src.height
        width = src.width
        
        # Criar matriz de saída
        output = np.zeros((height, width), dtype=np.uint8)
        count = np.zeros((height, width), dtype=np.uint8)
        
        # Calcular número de blocos
        stride = block_size - overlap
        n_blocks_h = (height - block_size) // stride + 1
        n_blocks_w = (width - block_size) // stride + 1
        
        # Processar cada bloco
        for i in tqdm(range(n_blocks_h), desc="Processando linhas"):
            for j in range(n_blocks_w):
                # Calcular índices do bloco
                y_start = i * stride
                x_start = j * stride
                
                # Ler o bloco
                window = rasterio.windows.Window(
                    x_start, y_start, block_size, block_size
                )
                block = src.read(window=window)
                block = np.transpose(block, (1, 2, 0))
                
                # Garantir que o bloco tenha exatamente 3 canais (RGB)
                if block.shape[2] > 3:
                    block = block[:, :, :3]
                elif block.shape[2] < 3:
                    # Preencher canais faltantes com zeros
                    zeros = np.zeros((block.shape[0], block.shape[1], 3 - block.shape[2]), dtype=block.dtype)
                    block = np.concatenate([block, zeros], axis=2)
                
                print(f"[LOG] Shape do bloco antes da predição: {block.shape}")
                
                # Normalizar
                block = block.astype(np.float32) / 255.0
                
                # Fazer predição
                pred = model.predict(block[np.newaxis, ...], verbose=0)
                pred = (pred[0, :, :, 0] > 0.5).astype(np.uint8) * 255
                
                # Adicionar ao resultado final
                output[y_start:y_start+block_size, x_start:x_start+block_size] += pred
                count[y_start:y_start+block_size, x_start:x_start+block_size] += 1
        
        # Média das sobreposições
        output = output / count
        output = (output > 127).astype(np.uint8) * 255
        
        return output

def main():
    print("[LOG] Iniciando script de inferência...")
    # Configurar codificação padrão para UTF-8
    if sys.platform.startswith('win'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    
    parser = argparse.ArgumentParser(
        description='Realiza inferência do modelo em uma imagem'
    )
    parser.add_argument(
        '--rgb',
        required=True,
        help='Caminho para a imagem RGB'
    )
    parser.add_argument(
        '--modelpath',
        required=True,
        help='Caminho para o modelo treinado'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Caminho para salvar a imagem segmentada'
    )
    parser.add_argument(
        '--block-size',
        type=int,
        default=256,
        help='Tamanho dos blocos (padrão: 256)'
    )
    parser.add_argument(
        '--overlap',
        type=int,
        default=32,
        help='Sobreposição entre blocos (padrão: 32)'
    )
    
    args = parser.parse_args()
    
    try:
        print(f"[LOG] Argumentos recebidos: rgb={args.rgb}, modelpath={args.modelpath}, output={args.output}, block_size={args.block_size}, overlap={args.overlap}")
        # Converte caminhos para objetos Path e resolve para caminhos absolutos
        rgb_path = Path(args.rgb).resolve()
        model_path = Path(args.modelpath).resolve()
        output_path = Path(args.output).resolve()
        
        print(f"[LOG] Caminho da imagem de entrada: {rgb_path}")
        print(f"[LOG] Caminho do modelo: {model_path}")
        print(f"[LOG] Caminho de saída: {output_path}")
        # Verificar arquivos
        if not rgb_path.exists():
            raise FileNotFoundError(f"Imagem de entrada não encontrada: {rgb_path}")
        
        # Criar diretório de saída se necessário
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print("[LOG] Carregando modelo...")
        model = load_model(str(model_path))
        print("[LOG] Modelo carregado com sucesso!")
        
        print(f"[LOG] Processando imagem: {rgb_path}")
        segmented = process_image_in_blocks(
            str(rgb_path),
            model,
            args.block_size,
            args.overlap
        )
        print("[LOG] Processamento da imagem concluído!")
        
        print("[LOG] Salvando resultado...")
        with rasterio.open(str(rgb_path)) as src:
            profile = src.profile
            profile.update(
                driver='GTiff',
                dtype=np.uint8,
                count=1,
                compress='lzw'
            )
            
            with rasterio.open(str(output_path), 'w', **profile) as dst:
                dst.write(segmented[np.newaxis, ...])
        print(f"[LOG] Imagem segmentada salva em: {output_path}")
        print("[LOG] Script finalizado com sucesso!")
        
    except Exception as e:
        print(f"\n[LOG][ERRO] Erro durante o processamento: {str(e)}")
        raise

if __name__ == '__main__':
    main() 