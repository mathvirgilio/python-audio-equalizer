"""
Ponto de entrada principal do Equalizador em Tempo Real
"""

import sys
import os

# Adiciona o diretório src ao path para permitir imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from realtime_equalizer import EqualizerGUI

def main():
    """Função principal."""
    import sys
    
    print("=" * 60)
    print("Equalizador em Tempo Real - 5 Bandas")
    print("=" * 60)
    print("\nFrequências das bandas:")
    print("  Banda 1: 100 Hz")
    print("  Banda 2: 330 Hz")
    print("  Banda 3: 1000 Hz")
    print("  Banda 4: 3300 Hz")
    print("  Banda 5: 10000 Hz")
    print("\nGanho em dB:")
    print("  0 dB = sem alteração")
    print("  +dB = amplificação")
    print("  -dB = atenuação")
    print("=" * 60)
    
    # Verifica se um arquivo foi passado como argumento
    audio_file = None
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        if not os.path.exists(audio_file):
            print(f"\nAviso: Arquivo '{audio_file}' não encontrado. Usando microfone.")
            audio_file = None
        else:
            print(f"\nUsando arquivo: {audio_file}")
    else:
        # Tenta usar o arquivo padrão
        default_file = "tracks/The Cure - In Between Days.mp3"
        if os.path.exists(default_file):
            audio_file = default_file
            print(f"\nUsando arquivo padrão: {audio_file}")
        else:
            print("\nNenhum arquivo especificado. Usando microfone.")
            print("Para usar um arquivo, execute: python realtime_equalizer.py <arquivo>")
    
    print("\nIniciando interface gráfica...")
    
    app = EqualizerGUI(audio_file=audio_file)
    app.run()


if __name__ == '__main__':
    main()
