import numpy as np
import librosa
import soundfile as sf
import os
from equalizer import create_frequency_filter


class MultiBandEqualizer:
    """
    Equalizador de 5 bandas que aplica filtros em frequências específicas
    e combina os sinais filtrados conforme a equação:
    y[n] = sum_{i=1}^{5} A_i * (x[n] * h_i[n]) = x[n] * (sum_{i=1}^{5} A_i * h_i[n])
    """
    
    def __init__(self, bandwidth=50, filter_shape='gaussian'):
        """
        Inicializa o equalizador de 5 bandas.
        
        Args:
            bandwidth: Largura de banda de cada filtro (Hz)
            filter_shape: Forma do filtro ('gaussian' ou 'rectangular')
        """
        # Frequências centrais das 5 bandas
        self.center_frequencies = [100, 330, 1000, 3300, 10000]  # Hz
        
        # Fatores de amplificação/atenuação para cada banda (0 a 100)
        # 50 = sem alteração, < 50 = atenuação, > 50 = amplificação
        self.amplification_factors = [100, 50, 25, 50, 50]  # Valores padrão (sem alteração)
        
        self.bandwidth = bandwidth
        self.filter_shape = filter_shape
        
    def set_band_gain(self, band_index, gain_factor):
        """
        Define o fator de amplificação/atenuação para uma banda específica.
        
        Args:
            band_index: Índice da banda (0-4 correspondendo a 100Hz, 330Hz, 1kHz, 3.3kHz, 10kHz)
            gain_factor: Fator de 0 a 100 (50 = sem alteração, < 50 = atenuação, > 50 = amplificação)
        """
        if band_index < 0 or band_index >= len(self.center_frequencies):
            raise ValueError(f"Índice de banda deve estar entre 0 e {len(self.center_frequencies) - 1}")
        
        if gain_factor < 0 or gain_factor > 100:
            raise ValueError("Fator de ganho deve estar entre 0 e 100")
        
        self.amplification_factors[band_index] = gain_factor
        
    def set_all_gains(self, gains):
        """
        Define os fatores de amplificação/atenuação para todas as bandas.
        
        Args:
            gains: Lista com 5 valores (0 a 100) para cada banda
        """
        if len(gains) != len(self.center_frequencies):
            raise ValueError(f"Deve fornecer exatamente {len(self.center_frequencies)} valores de ganho")
        
        for i, gain in enumerate(gains):
            self.set_band_gain(i, gain)
    
    def _create_combined_filter(self, n_samples, sample_rate):
        """
        Cria o filtro combinado h[n] = sum_{i=1}^{5} A_i * h_i[n] no domínio da frequência.
        
        Args:
            n_samples: Número de amostras do sinal
            sample_rate: Taxa de amostragem (Hz)
        
        Returns:
            Array com a resposta do filtro combinado no domínio da frequência
        """
        # Normaliza os fatores de 0-100 para valores lineares
        # 50 = 1.0 (sem alteração), 0 = 0.0 (atenuação total), 100 = 2.0 (amplificação máxima)
        normalized_factors = [factor / 50.0 for factor in self.amplification_factors]
        
        # Inicializa o filtro combinado
        combined_filter = np.zeros(n_samples, dtype=np.complex128)
        
        # Para cada banda, cria o filtro e adiciona ao filtro combinado
        for i, center_freq in enumerate(self.center_frequencies):
            # Cria o filtro passa-banda para esta frequência
            h_i = create_frequency_filter(n_samples, sample_rate, center_freq, 
                                         self.bandwidth, self.filter_shape)
            
            # Adiciona ao filtro combinado multiplicado pelo fator A_i
            combined_filter += normalized_factors[i] * h_i
        
        return combined_filter
    
    def process_audio(self, input_file, output_file=None):
        """
        Processa um arquivo de áudio aplicando o equalizador de 5 bandas.
        
        Args:
            input_file: Caminho do arquivo de áudio de entrada
            output_file: Caminho do arquivo de saída (opcional)
        
        Returns:
            Tuple (audio_filtrado, sample_rate)
        """
        print(f"Carregando arquivo: {input_file}")
        
        # Carrega o áudio
        audio, sample_rate = librosa.load(input_file, sr=None, mono=False)
        
        # Se o áudio for mono, converte para formato 2D
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        
        n_samples = audio.shape[1]
        
        print(f"Taxa de amostragem: {sample_rate} Hz")
        print(f"Duração: {n_samples / sample_rate:.2f} segundos")
        print(f"Canais: {audio.shape[0]}")
        print(f"Aplicando equalizador de 5 bandas...")
        print(f"Fatores de ganho: {self.amplification_factors}")
        
        # Cria o filtro combinado uma vez (é o mesmo para todos os canais)
        combined_filter = self._create_combined_filter(n_samples, sample_rate)
        
        # Processa cada canal
        filtered_channels = []
        for channel_idx, channel in enumerate(audio):
            print(f"Processando canal {channel_idx + 1}...")
            
            # Converte para o domínio da frequência
            audio_fft = np.fft.fft(channel)
            
            # Aplica o filtro combinado: y[n] = x[n] * h[n]
            # No domínio da frequência: Y[k] = X[k] * H[k]
            filtered_fft = audio_fft * combined_filter
            
            # Converte de volta para o domínio do tempo
            filtered_audio = np.real(np.fft.ifft(filtered_fft))
            
            filtered_channels.append(filtered_audio)
        
        # Combina os canais
        filtered_audio = np.array(filtered_channels)
        
        # Se for mono, converte de volta para 1D
        if filtered_audio.shape[0] == 1:
            filtered_audio = filtered_audio[0]
        else:
            # Transpõe para formato (samples, channels)
            filtered_audio = filtered_audio.T
        
        # Normaliza para evitar clipping
        max_val = np.max(np.abs(filtered_audio))
        if max_val > 1.0:
            print(f"Normalizando sinal (máximo: {max_val:.3f})")
            filtered_audio = filtered_audio / max_val
        
        # Salva o arquivo de saída
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            gains_str = "_".join([f"{int(g)}" for g in self.amplification_factors])
            output_file = f"{base_name}_equalized_{gains_str}.wav"
        
        print(f"Salvando arquivo equalizado: {output_file}")
        sf.write(output_file, filtered_audio, sample_rate)
        
        print("Processamento concluído!")
        return filtered_audio, sample_rate
    
    def get_band_info(self):
        """
        Retorna informações sobre as bandas do equalizador.
        
        Returns:
            Lista de tuplas (frequência, fator_atual)
        """
        return [(freq, factor) for freq, factor in 
                zip(self.center_frequencies, self.amplification_factors)]


def main():
    """
    Exemplo de uso do equalizador de 5 bandas.
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python multi_band_equalizer.py <arquivo_audio> [fatores_gain]")
        print("\nExemplo:")
        print("  python multi_band_equalizer.py 'audio.wav' 50 50 50 50 50")
        print("  python multi_band_equalizer.py 'audio.wav' 100 75 50 25 0")
        print("\nOs fatores devem estar entre 0 e 100:")
        print("  50 = sem alteração")
        print("  < 50 = atenuação")
        print("  > 50 = amplificação")
        print("\nOrdem das bandas: 100 Hz, 330 Hz, 1 kHz, 3.3 kHz, 10 kHz")
        return
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Erro: Arquivo '{input_file}' não encontrado!")
        return
    
    # Cria o equalizador
    eq = MultiBandEqualizer(bandwidth=50, filter_shape='gaussian')
    
    # Se foram fornecidos fatores de ganho, usa-os
    if len(sys.argv) > 2:
        try:
            gains = [float(g) for g in sys.argv[2:7]]
            if len(gains) == 5:
                eq.set_all_gains(gains)
            else:
                print("Aviso: Forneça exatamente 5 valores de ganho. Usando valores padrão.")
        except ValueError:
            print("Aviso: Valores de ganho inválidos. Usando valores padrão.")
    
    # Processa o áudio
    eq.process_audio(input_file)


if __name__ == '__main__':
    main()

