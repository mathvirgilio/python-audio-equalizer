"""
Exemplo de uso do equalizador de 5 bandas.
Este script demonstra como usar a classe MultiBandEqualizer para aplicar
filtros em múltiplas frequências simultaneamente.
"""

from multi_band_equalizer import MultiBandEqualizer

# Arquivo de entrada
arquivo_entrada = "The Cure - In Between Days.mp3"

# Cria o equalizador de 5 bandas
print("=" * 60)
print("Equalizador de 5 Bandas")
print("=" * 60)
print("\nFrequências das bandas:")
print("  Banda 1: 100 Hz")
print("  Banda 2: 330 Hz")
print("  Banda 3: 1 kHz")
print("  Banda 4: 3.3 kHz")
print("  Banda 5: 10 kHz")
print("\nFatores de ganho (0-100):")
print("  50 = sem alteração")
print("  < 50 = atenuação")
print("  > 50 = amplificação")
print("=" * 60)

# Cria o equalizador
eq = MultiBandEqualizer(bandwidth=50, filter_shape='gaussian')

# Exemplo 1: Equalização plana (sem alteração)
print("\n[Exemplo 1] Equalização plana (todos os fatores = 50)")
eq.set_all_gains([50, 50, 50, 50, 50])
eq.process_audio(arquivo_entrada, "output_flat.wav")

# Exemplo 2: Realçar graves (100 Hz) e agudos (10 kHz)
print("\n[Exemplo 2] Realçar graves e agudos")
eq.set_all_gains([100, 50, 50, 50, 100])  # Máxima amplificação em 100 Hz e 10 kHz
eq.process_audio(arquivo_entrada, "output_bass_treble.wav")

# Exemplo 3: Atenuar médios (1 kHz e 3.3 kHz)
print("\n[Exemplo 3] Atenuar médios")
eq.set_all_gains([50, 50, 0, 0, 50])  # Atenuação total em 1 kHz e 3.3 kHz
eq.process_audio(arquivo_entrada, "output_mid_cut.wav")

# Exemplo 4: Configuração personalizada
print("\n[Exemplo 4] Configuração personalizada")
eq.set_band_gain(0, 75)   # 100 Hz: leve amplificação
eq.set_band_gain(1, 60)   # 330 Hz: leve amplificação
eq.set_band_gain(2, 40)   # 1 kHz: atenuação
eq.set_band_gain(3, 30)   # 3.3 kHz: mais atenuação
eq.set_band_gain(4, 80)   # 10 kHz: amplificação
eq.process_audio(arquivo_entrada, "output_custom.wav")

# Mostra informações sobre as bandas
print("\n" + "=" * 60)
print("Informações das bandas configuradas:")
print("=" * 60)
for freq, factor in eq.get_band_info():
    print(f"  {freq:6.0f} Hz: fator = {factor:.1f}")

print("\nTodos os exemplos foram processados com sucesso!")

