"""
Exemplo de uso do equalizador em tempo real.
Este script demonstra como usar o RealTimeEqualizer para processar arquivos de áudio.
"""

from realtime_equalizer import RealTimeEqualizer
import librosa
import pyaudio
import numpy as np
import time
import os
import sys


def play_audio_with_equalizer(audio_file, gains_db=None):
    """
    Carrega e reproduz um arquivo de áudio aplicando o equalizador em tempo real.
    
    Args:
        audio_file: Caminho do arquivo de áudio
        gains_db: Lista com ganhos em dB para cada banda (opcional)
    """
    print("=" * 60)
    print("Equalizador em Tempo Real - Reprodução de Arquivo")
    print("=" * 60)
    
    # Verifica se o arquivo existe
    if not os.path.exists(audio_file):
        print(f"Erro: Arquivo '{audio_file}' não encontrado!")
        return
    
    print(f"\nCarregando arquivo: {audio_file}")
    
    # Carrega o áudio
    try:
        audio, sample_rate = librosa.load(audio_file, sr=None, mono=True)
        print(f"Taxa de amostragem: {sample_rate} Hz")
        print(f"Duração: {len(audio) / sample_rate:.2f} segundos")
    except Exception as e:
        print(f"Erro ao carregar arquivo: {e}")
        return
    
    # Cria o equalizador
    chunk_size = 1024
    eq = RealTimeEqualizer(sample_rate=sample_rate, chunk_size=chunk_size)
    
    # Define ganhos se fornecidos
    if gains_db is not None:
        print("\nConfigurando ganhos:")
        for i, gain_db in enumerate(gains_db):
            freq = eq.center_frequencies[i]
            freq_str = f"{freq} Hz" if freq < 1000 else f"{freq/1000:.1f} kHz"
            print(f"  Banda {i+1} ({freq_str}): {gain_db:+.1f} dB")
            eq.set_band_gain_db(i, gain_db)
    else:
        # Ganhos padrão
        print("\nUsando ganhos padrão:")
        eq.set_band_gain_db(0, 6.0)   # 100 Hz: +6 dB
        eq.set_band_gain_db(1, 0.0)   # 330 Hz: 0 dB
        eq.set_band_gain_db(2, -3.0)  # 1000 Hz: -3 dB
        eq.set_band_gain_db(3, 0.0)   # 3300 Hz: 0 dB
        eq.set_band_gain_db(4, 3.0)   # 10000 Hz: +3 dB
    
    # Mostra informações
    print("\nInformações das bandas:")
    for freq, gain_db in eq.get_band_info():
        gain_linear = 10 ** (gain_db / 20.0)
        freq_str = f"{freq} Hz" if freq < 1000 else f"{freq/1000:.1f} kHz"
        print(f"  {freq_str:8s}: {gain_db:+.1f} dB (A = {gain_linear:.3f})")
    
    # Inicializa PyAudio
    p = pyaudio.PyAudio()
    
    # Abre stream de saída
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=sample_rate,
        output=True,
        frames_per_buffer=chunk_size
    )
    
    print("\n" + "=" * 60)
    print("Reproduzindo áudio com equalizador...")
    print("Pressione Ctrl+C para parar")
    print("=" * 60)
    
    try:
        # Processa e reproduz o áudio em blocos
        audio_index = 0
        total_samples = len(audio)
        
        while audio_index < total_samples:
            # Pega um chunk do áudio
            chunk_end = min(audio_index + chunk_size, total_samples)
            audio_chunk = audio[audio_index:chunk_end]
            
            # Se o chunk for menor que chunk_size, preenche com zeros
            if len(audio_chunk) < chunk_size:
                padded_chunk = np.zeros(chunk_size, dtype=np.float32)
                padded_chunk[:len(audio_chunk)] = audio_chunk
                audio_chunk = padded_chunk
            
            # Processa o chunk com o equalizador
            processed_chunk = eq.process_chunk(audio_chunk.astype(np.float32))
            
            # Reproduz o chunk processado
            stream.write(processed_chunk.tobytes())
            
            # Avança para o próximo chunk
            audio_index += chunk_size
            
            # Pequeno delay para evitar buffer underrun
            time.sleep(0.001)
        
        print("\nReprodução concluída!")
        
    except KeyboardInterrupt:
        print("\n\nReprodução interrompida pelo usuário")
    except Exception as e:
        print(f"\nErro durante reprodução: {e}")
    finally:
        # Fecha o stream e PyAudio
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Stream de áudio fechado")


def exemplo_programatico():
    """
    Exemplo de uso programático do equalizador com microfone.
    """
    print("=" * 60)
    print("Exemplo: Equalizador em Tempo Real (Microfone)")
    print("=" * 60)
    
    # Cria o equalizador
    eq = RealTimeEqualizer(sample_rate=44100, chunk_size=1024)
    
    # Define ganhos em dB para cada banda
    print("\nConfigurando ganhos:")
    print("  Banda 1 (100 Hz): +6 dB")
    print("  Banda 2 (330 Hz): 0 dB")
    print("  Banda 3 (1000 Hz): -3 dB")
    print("  Banda 4 (3300 Hz): 0 dB")
    print("  Banda 5 (10000 Hz): +3 dB")
    
    eq.set_band_gain_db(0, 6.0)   # 100 Hz: +6 dB
    eq.set_band_gain_db(1, 0.0)   # 330 Hz: 0 dB
    eq.set_band_gain_db(2, -3.0)  # 1000 Hz: -3 dB
    eq.set_band_gain_db(3, 0.0)   # 3300 Hz: 0 dB
    eq.set_band_gain_db(4, 3.0)   # 10000 Hz: +3 dB
    
    # Mostra informações
    print("\nInformações das bandas:")
    for freq, gain_db in eq.get_band_info():
        gain_linear = 10 ** (gain_db / 20.0)
        print(f"  {freq:6.0f} Hz: {gain_db:+.1f} dB (A = {gain_linear:.3f})")
    
    # Inicia o processamento
    print("\nIniciando processamento em tempo real...")
    print("Pressione Ctrl+C para parar")
    
    try:
        eq.start_processing()
        
        # Processa por 30 segundos (ou até interromper)
        time.sleep(30)
        
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário")
    finally:
        eq.stop_processing()
        print("Processamento finalizado")


if __name__ == '__main__':
    # Arquivo padrão
    default_file = r"C:\Users\mathe\OneDrive\Documents\GitHub\python-audio-equalizer\The Cure - In Between Days.mp3"
    
    print("\n" + "=" * 60)
    print("Exemplo de Uso do Equalizador em Tempo Real")
    print("=" * 60)
    print("\nOpções:")
    print("  1. Reproduzir arquivo MP3 com equalizador")
    print("  2. Processar áudio do microfone")
    print("  3. Sair")
    
    escolha = input("\nEscolha uma opção (1/2/3): ").strip()
    
    if escolha == '1':
        # Usa o arquivo padrão ou permite especificar outro
        arquivo = input(f"\nArquivo de áudio (Enter para usar padrão): ").strip()
        if not arquivo:
            arquivo = default_file
        
        # Ganhos personalizados (opcional)
        print("\nDeseja usar ganhos personalizados? (Enter para usar padrão)")
        gains_input = input("Ganhos em dB (5 valores separados por espaço): ").strip()
        
        gains_db = None
        if gains_input:
            try:
                gains_list = [float(g) for g in gains_input.split()]
                if len(gains_list) == 5:
                    gains_db = gains_list
                else:
                    print("Aviso: Forneça exatamente 5 valores. Usando ganhos padrão.")
            except ValueError:
                print("Aviso: Valores inválidos. Usando ganhos padrão.")
        
        play_audio_with_equalizer(arquivo, gains_db)
        
    elif escolha == '2':
        exemplo_programatico()
    else:
        print("Saindo...")

