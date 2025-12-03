import numpy as np
import librosa
import soundfile as sf
import argparse
import os


def calculate_cutoff_frequencies(center_frequencies, sample_rate):
    """
    Calcula as frequências de corte que ficam no meio entre as frequências centrais das bandas.
    Conforme especificação: "As frequências de corte dos filtros devem estar no meio 
    entre as frequências centrais das bandas."
    
    Args:
        center_frequencies: Lista com as frequências centrais das bandas (Hz)
        sample_rate: Taxa de amostragem (Hz)
    
    Returns:
        Lista de tuplas (low_cutoff, high_cutoff) para cada banda
    """
    cutoff_freqs = []
    n_bands = len(center_frequencies)
    
    for i in range(n_bands):
        # Calcula a frequência de corte inferior
        if i == 0:
            # Primeira banda: corte inferior é 0 Hz ou ponto médio entre 0 e a primeira frequência central
            # Usamos um valor baixo (ex: 50 Hz) ou podemos usar 0
            low_cutoff = 0.0
        else:
            # Ponto médio entre a frequência central anterior e a atual
            low_cutoff = (center_frequencies[i-1] + center_frequencies[i]) / 2.0
        
        # Calcula a frequência de corte superior
        if i == n_bands - 1:
            # Última banda: corte superior é a frequência de Nyquist
            high_cutoff = sample_rate / 2.0
        else:
            # Ponto médio entre a frequência central atual e a próxima
            high_cutoff = (center_frequencies[i] + center_frequencies[i+1]) / 2.0
        
        cutoff_freqs.append((low_cutoff, high_cutoff))
    
    return cutoff_freqs


def create_bandpass_impulse_response(filter_length, sample_rate, low_cutoff, high_cutoff):
    """
    Cria a resposta ao impulso de um filtro passa-faixa usando a expressão matemática:
    h[n] = (sin(ω_c2 n) - sin(ω_c1 n)) / (nπ) quando n ≠ 0
    h[n] = (ω_c2 - ω_c1) / π quando n = 0
    
    Usando numpy.sinc para implementação eficiente.
    
    Args:
        filter_length: Comprimento do filtro (número de taps, deve ser ímpar)
        sample_rate: Taxa de amostragem (Hz)
        low_cutoff: Frequência de corte inferior (Hz)
        high_cutoff: Frequência de corte superior (Hz)
    
    Returns:
        Array com a resposta ao impulso do filtro
    """
    # Garante que o comprimento seja ímpar
    if filter_length % 2 == 0:
        filter_length += 1
    
    # Converte frequências de Hz para radianos por amostra
    omega_c1 = 2 * np.pi * low_cutoff / sample_rate
    omega_c2 = 2 * np.pi * high_cutoff / sample_rate
    
    # Cria índices n centrados em zero: [-M, ..., -1, 0, 1, ..., M]
    M = (filter_length - 1) // 2
    n = np.arange(-M, M + 1, dtype=np.float64)
    
    # Inicializa a resposta ao impulso
    h = np.zeros(filter_length, dtype=np.float64)
    
    # Caso n = 0
    h[M] = (omega_c2 - omega_c1) / np.pi
    
    # Caso n ≠ 0: usa numpy.sinc
    # sinc(x) = sin(πx) / (πx)
    # sin(ω_c2 n) / (nπ) = (ω_c2 / π) * sinc(ω_c2 n / π)
    # sin(ω_c1 n) / (nπ) = (ω_c1 / π) * sinc(ω_c1 n / π)
    mask = n != 0
    n_nonzero = n[mask]
    
    # Usa sinc para calcular sin(ω n) / (nπ)
    # sin(ω n) / (nπ) = (ω / π) * sinc(ω n / π)
    h[mask] = (omega_c2 / np.pi) * np.sinc(omega_c2 * n_nonzero / np.pi) - \
              (omega_c1 / np.pi) * np.sinc(omega_c1 * n_nonzero / np.pi)
    
    return h


def create_frequency_filter(n_samples, sample_rate, center_freq, bandwidth=50, 
                           low_cutoff=None, high_cutoff=None, filter_shape='gaussian'):
    """
    Cria um filtro passa-banda no domínio da frequência.
    
    Args:
        n_samples: Número de amostras do sinal
        sample_rate: Taxa de amostragem (Hz)
        center_freq: Frequência central do filtro (Hz)
        bandwidth: Largura de banda do filtro (Hz) - usado apenas se low_cutoff/high_cutoff não fornecidos
        low_cutoff: Frequência de corte inferior (Hz) - se None, calcula a partir de bandwidth
        high_cutoff: Frequência de corte superior (Hz) - se None, calcula a partir de bandwidth
        filter_shape: Forma do filtro ('gaussian' ou 'sinc')
    
    Returns:
        Array com os valores do filtro no domínio da frequência
    """
    # Calcula as frequências correspondentes a cada bin da FFT
    freqs = np.fft.fftfreq(n_samples, 1.0 / sample_rate)
    freqs = np.abs(freqs)  # Apenas valores positivos
    
    # Calcula as frequências de corte
    if low_cutoff is not None and high_cutoff is not None:
        # Usa as frequências de corte fornecidas
        low_freq = max(0, low_cutoff)
        high_freq = min(sample_rate / 2, high_cutoff)
        # Calcula bandwidth efetivo para filtro gaussiano
        effective_bandwidth = high_freq - low_freq
    else:
        # Usa o método antigo baseado em bandwidth
        low_freq = max(0, center_freq - bandwidth / 2)
        high_freq = min(sample_rate / 2, center_freq + bandwidth / 2)
        effective_bandwidth = bandwidth
    
    if filter_shape == 'sinc':
        # Filtro passa-faixa usando resposta ao impulso com sinc
        # Usa um comprimento de filtro baseado na resolução de frequência desejada
        # Fórmula: filter_length ≈ 4 * sample_rate / bandwidth (regra de ouro)
        filter_length = int(4 * sample_rate / max(effective_bandwidth, 1))
        # Limita o comprimento para não ser muito grande
        filter_length = min(filter_length, n_samples // 2)
        # Garante que seja ímpar e pelo menos 3
        if filter_length % 2 == 0:
            filter_length += 1
        filter_length = max(3, filter_length)
        
        # Cria a resposta ao impulso
        h = create_bandpass_impulse_response(filter_length, sample_rate, low_freq, high_freq)
        
        # Aplica uma janela (Hamming) para reduzir ringing
        window = np.hamming(filter_length)
        h = h * window
        
        # Converte para o domínio da frequência usando FFT
        # Preenche com zeros até n_samples para ter o mesmo tamanho do sinal
        h_padded = np.zeros(n_samples, dtype=np.float64)
        M = len(h) // 2
        # Coloca o filtro centralizado (h[0] no índice 0, h[-M:] no final para circularidade)
        h_padded[:M+1] = h[M:]
        if M > 0:
            h_padded[-M:] = h[:M]
        
        # Calcula a FFT do filtro
        filter_response = np.fft.fft(h_padded)
        # Usa apenas a magnitude (filtro passa-faixa ideal tem fase zero)
        filter_response = np.abs(filter_response)
        
        return filter_response
    
    elif filter_shape == 'gaussian':
        # Filtro gaussiano (transições suaves)
        # Para filtros com frequências de corte específicas, ajusta o sigma para que
        # o filtro tenha transição suave entre low_freq e high_freq
        # Usa desvio padrão baseado na largura de banda efetiva
        sigma = effective_bandwidth / (2 * np.sqrt(2 * np.log(2)))  # FWHM = 2.355 * sigma
        filter_response = np.exp(-0.5 * ((freqs - center_freq) / sigma) ** 2)
        
        # Aplica uma janela para garantir que o filtro seja zero fora do intervalo [low_freq, high_freq]
        # Isso garante que as frequências de corte sejam respeitadas
        window = np.ones_like(freqs)
        window[freqs < low_freq] = 0.0
        window[freqs > high_freq] = 0.0
        # Transição suave nas bordas (opcional, pode ser removido para transição mais abrupta)
        transition_width = effective_bandwidth * 0.1  # 10% da largura de banda para transição
        if transition_width > 0:
            # Transição suave na borda inferior
            transition_low = (freqs >= low_freq) & (freqs < low_freq + transition_width)
            window[transition_low] = (freqs[transition_low] - low_freq) / transition_width
            # Transição suave na borda superior
            transition_high = (freqs > high_freq - transition_width) & (freqs <= high_freq)
            window[transition_high] = (high_freq - freqs[transition_high]) / transition_width
        
        filter_response *= window
        return filter_response
    else:
        raise ValueError(f"filter_shape deve ser 'sinc' ou 'gaussian', recebido: {filter_shape}")


def apply_bandpass_filter(audio, sample_rate, center_freq, bandwidth=50, 
                          low_cutoff=None, high_cutoff=None, filter_shape='sinc'):
    """
    Aplica um filtro passa-banda centrado em uma frequência específica.
    
    Args:
        audio: Array numpy com o sinal de áudio
        sample_rate: Taxa de amostragem do áudio
        center_freq: Frequência central do filtro (Hz)
        bandwidth: Largura de banda do filtro (Hz) - usado apenas se low_cutoff/high_cutoff não fornecidos
        low_cutoff: Frequência de corte inferior (Hz) - se None, calcula a partir de bandwidth
        high_cutoff: Frequência de corte superior (Hz) - se None, calcula a partir de bandwidth
        filter_shape: Forma do filtro ('sinc' ou 'gaussian')
                     'sinc' usa a resposta ao impulso matemática com numpy.sinc (padrão)
    
    Returns:
        Áudio filtrado
    """
    n_samples = len(audio)
    
    # Converte para o domínio da frequência
    audio_fft = np.fft.fft(audio)
    
    # Cria o filtro no domínio da frequência
    filter_response = create_frequency_filter(n_samples, sample_rate, center_freq, bandwidth, 
                                             low_cutoff=low_cutoff, high_cutoff=high_cutoff, 
                                             filter_shape=filter_shape)
    
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
                  filter_type='bandpass', gain_db=0, q=1.0, filter_shape='sinc'):
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
        filter_shape: Forma do filtro ('sinc' ou 'gaussian')
                     'sinc' usa a resposta ao impulso matemática com numpy.sinc (padrão)
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
        default_file = "tracks/The Cure - In Between Days.mp3"
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

