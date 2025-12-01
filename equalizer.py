import numpy as np
import librosa
import soundfile as sf
import argparse
import os


def create_frequency_filter(n_samples, sample_rate, center_freq, bandwidth=50, filter_shape='gaussian'):
    """
    Cria um filtro passa-banda no domínio da frequência.
    
    Args:
        n_samples: Número de amostras do sinal
        sample_rate: Taxa de amostragem (Hz)
        center_freq: Frequência central do filtro (Hz)
        bandwidth: Largura de banda do filtro (Hz)
        filter_shape: Forma do filtro ('gaussian' ou 'rectangular')
    
    Returns:
        Array com os valores do filtro no domínio da frequência
    """
    # Calcula as frequências correspondentes a cada bin da FFT
    freqs = np.fft.fftfreq(n_samples, 1.0 / sample_rate)
    freqs = np.abs(freqs)  # Apenas valores positivos
    
    # Calcula as frequências de corte
    low_freq = max(0, center_freq - bandwidth / 2)
    high_freq = min(sample_rate / 2, center_freq + bandwidth / 2)
    
    if filter_shape == 'gaussian':
        # Filtro gaussiano (transições suaves)
        # Usa desvio padrão baseado na largura de banda
        sigma = bandwidth / (2 * np.sqrt(2 * np.log(2)))  # FWHM = 2.355 * sigma
        filter_response = np.exp(-0.5 * ((freqs - center_freq) / sigma) ** 2)
    else:
        # Filtro retangular (transições abruptas)
        filter_response = np.zeros_like(freqs)
        mask = (freqs >= low_freq) & (freqs <= high_freq)
        filter_response[mask] = 1.0
    
    return filter_response


def apply_bandpass_filter(audio, sample_rate, center_freq, bandwidth=50, filter_shape='gaussian'):
    """
    Aplica um filtro passa-banda centrado em uma frequência específica usando FFT.
    
    Args:
        audio: Array numpy com o sinal de áudio
        sample_rate: Taxa de amostragem do áudio
        center_freq: Frequência central do filtro (Hz)
        bandwidth: Largura de banda do filtro (Hz)
        filter_shape: Forma do filtro ('gaussian' ou 'rectangular')
    
    Returns:
        Áudio filtrado
    """
    n_samples = len(audio)
    
    # Converte para o domínio da frequência
    audio_fft = np.fft.fft(audio)
    
    # Cria o filtro no domínio da frequência
    filter_response = create_frequency_filter(n_samples, sample_rate, center_freq, bandwidth, filter_shape)
    
    # Aplica o filtro multiplicando no domínio da frequência
    filtered_fft = audio_fft * filter_response
    
    # Converte de volta para o domínio do tempo
    filtered_audio = np.real(np.fft.ifft(filtered_fft))
    
    return filtered_audio


def apply_parametric_eq(audio, sample_rate, center_freq, gain_db=0, q=1.0):
    """
    Aplica um equalizador paramétrico (boost/cut) em uma frequência específica usando FFT.
    
    Args:
        audio: Array numpy com o sinal de áudio
        sample_rate: Taxa de amostragem do áudio
        center_freq: Frequência central (Hz)
        gain_db: Ganho em dB (positivo = boost, negativo = cut)
        q: Fator Q (largura do filtro, maior = mais estreito)
    
    Returns:
        Áudio filtrado
    """
    if gain_db == 0:
        return audio
    
    n_samples = len(audio)
    
    # Converte ganho de dB para linear
    gain_linear = 10 ** (gain_db / 20.0)
    
    # Calcula largura de banda a partir do Q
    bandwidth = center_freq / q
    
    # Converte para o domínio da frequência
    audio_fft = np.fft.fft(audio)
    
    # Cria o filtro paramétrico no domínio da frequência
    freqs = np.fft.fftfreq(n_samples, 1.0 / sample_rate)
    freqs = np.abs(freqs)
    
    # Filtro gaussiano para o equalizador paramétrico
    sigma = bandwidth / (2 * np.sqrt(2 * np.log(2)))
    filter_response = np.exp(-0.5 * ((freqs - center_freq) / sigma) ** 2)
    
    # Aplica o ganho apenas na faixa de frequências do filtro
    # Para boost: multiplica a resposta do filtro pelo ganho e adiciona ao sinal original
    # Para cut: reduz a resposta do filtro do sinal original
    if gain_db > 0:
        # Boost: adiciona o sinal filtrado amplificado
        eq_response = 1.0 + filter_response * (gain_linear - 1.0)
    else:
        # Cut: reduz o sinal filtrado
        eq_response = 1.0 - filter_response * (1.0 - gain_linear)
    
    # Aplica o equalizador
    filtered_fft = audio_fft * eq_response
    
    # Converte de volta para o domínio do tempo
    output = np.real(np.fft.ifft(filtered_fft))
    
    # Normaliza para evitar clipping
    max_val = np.max(np.abs(output))
    if max_val > 1.0:
        output = output / max_val
    
    return output


def process_audio(input_file, output_file=None, center_freq=100, bandwidth=50, 
                  filter_type='bandpass', gain_db=0, q=1.0, filter_shape='gaussian'):
    """
    Processa um arquivo de áudio aplicando um filtro de frequência.
    
    Args:
        input_file: Caminho do arquivo de entrada
        output_file: Caminho do arquivo de saída (opcional)
        center_freq: Frequência central do filtro (Hz)
        bandwidth: Largura de banda (apenas para filtro passa-banda)
        filter_type: Tipo de filtro ('bandpass' ou 'parametric')
        gain_db: Ganho em dB (apenas para equalizador paramétrico)
        q: Fator Q (apenas para equalizador paramétrico)
        filter_shape: Forma do filtro ('gaussian' ou 'rectangular')
    """
    print(f"Carregando arquivo: {input_file}")
    
    # Carrega o áudio
    audio, sample_rate = librosa.load(input_file, sr=None, mono=False)
    
    # Se o áudio for mono, converte para formato 2D
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)
    
    print(f"Taxa de amostragem: {sample_rate} Hz")
    print(f"Duração: {len(audio[0]) / sample_rate:.2f} segundos")
    print(f"Canais: {audio.shape[0]}")
    print(f"Aplicando filtro centrado em {center_freq} Hz...")
    
    # Processa cada canal
    filtered_channels = []
    for channel in audio:
        if filter_type == 'bandpass':
            filtered = apply_bandpass_filter(channel, sample_rate, center_freq, bandwidth, filter_shape)
        elif filter_type == 'parametric':
            filtered = apply_parametric_eq(channel, sample_rate, center_freq, gain_db, q)
        else:
            filtered = apply_bandpass_filter(channel, sample_rate, center_freq, bandwidth, filter_shape)
        
        filtered_channels.append(filtered)
    
    # Combina os canais
    filtered_audio = np.array(filtered_channels)
    
    # Se for mono, converte de volta para 1D
    if filtered_audio.shape[0] == 1:
        filtered_audio = filtered_audio[0]
    else:
        # Transpõe para formato (samples, channels)
        filtered_audio = filtered_audio.T
    
    # Salva o arquivo de saída
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_filtered_{center_freq}Hz.wav"
    
    print(f"Salvando arquivo filtrado: {output_file}")
    sf.write(output_file, filtered_audio, sample_rate)
    
    print("Processamento concluído!")
    return filtered_audio, sample_rate


def main():
    parser = argparse.ArgumentParser(description='Filtro de frequência para áudio')
    parser.add_argument('input_file', help='Arquivo de áudio de entrada')
    parser.add_argument('-o', '--output', help='Arquivo de saída (opcional)')
    parser.add_argument('-f', '--freq', type=float, default=100, 
                       help='Frequência central do filtro (Hz) - padrão: 100')
    parser.add_argument('-b', '--bandwidth', type=float, default=50,
                       help='Largura de banda (Hz) - padrão: 50')
    parser.add_argument('-t', '--type', choices=['bandpass', 'parametric'], 
                       default='bandpass', help='Tipo de filtro')
    parser.add_argument('-g', '--gain', type=float, default=0,
                       help='Ganho em dB (apenas para filtro paramétrico)')
    parser.add_argument('-q', '--q', type=float, default=1.0,
                       help='Fator Q (apenas para filtro paramétrico)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Erro: Arquivo '{args.input_file}' não encontrado!")
        return
    
    process_audio(
        args.input_file,
        args.output,
        center_freq=args.freq,
        bandwidth=args.bandwidth,
        filter_type=args.type,
        gain_db=args.gain,
        q=args.q
    )


if __name__ == '__main__':
    # Se executado sem argumentos, usa o arquivo padrão
    import sys
    if len(sys.argv) == 1:
        default_file = "The Cure - In Between Days.mp3"
        if os.path.exists(default_file):
            print("Executando com arquivo padrão...")
            process_audio(default_file, center_freq=100, bandwidth=50)
        else:
            print("Uso: python equalizer.py <arquivo_audio> [opções]")
            print("\nExemplo:")
            print("  python equalizer.py 'The Cure - In Between Days.mp3' -f 100")
            print("\nOpções:")
            print("  -f, --freq FLOAT     Frequência central (Hz) - padrão: 100")
            print("  -b, --bandwidth FLOAT Largura de banda (Hz) - padrão: 50")
            print("  -o, --output FILE    Arquivo de saída")
            print("  -t, --type TYPE      Tipo: 'bandpass' ou 'parametric'")
            print("  -g, --gain FLOAT     Ganho em dB (paramétrico)")
            print("  -q, --q FLOAT        Fator Q (paramétrico)")
    else:
        main()

