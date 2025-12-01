"""
Exemplo simples de uso do equalizador de áudio.
Este script aplica um filtro de 100 Hz no arquivo MP3.
"""

from equalizer import process_audio

# Arquivo de entrada
arquivo_entrada = "The Cure - In Between Days.mp3"

# Aplica filtro passa-banda centrado em 100 Hz
print("=" * 50)
print("Filtro de Frequência - 100 Hz")
print("=" * 50)

process_audio(
    input_file=arquivo_entrada,
    output_file="The Cure - In Between Days_filtered_100Hz.wav",
    center_freq=100,      # Frequência central: 100 Hz
    bandwidth=50,         # Largura de banda: 50 Hz (filtra entre 75-125 Hz)
    filter_type='bandpass'
)

print("\nArquivo processado com sucesso!")
print("Arquivo de saída: The Cure - In Between Days_filtered_100Hz.wav")

