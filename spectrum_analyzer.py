"""
Analisador de Espectro de Barras
Implementa um analisador de espectro de barras com 10 bandas para visualização em tempo real.
"""

import numpy as np


class SpectrumAnalyzer:
    """
    Analisador de espectro de barras com 10 bandas para visualização em tempo real.
    """
    
    def __init__(self, sample_rate=44100, n_bands=10):
        """
        Inicializa o analisador de espectro.
        
        Args:
            sample_rate: Taxa de amostragem (Hz)
            n_bands: Número de bandas (padrão: 10)
        """
        self.sample_rate = sample_rate
        self.n_bands = n_bands
        
        # Define as frequências centrais das 10 bandas (distribuição logarítmica)
        # Cobre de ~20 Hz até Nyquist (sample_rate/2)
        nyquist = sample_rate / 2.0
        # Usa distribuição logarítmica para melhor representação do espectro de áudio
        self.band_centers = np.logspace(np.log10(20), np.log10(nyquist), n_bands)
        
        # Calcula as frequências de corte entre as bandas
        self.band_edges = []
        for i in range(n_bands):
            if i == 0:
                low_edge = 0
            else:
                low_edge = np.sqrt(self.band_centers[i-1] * self.band_centers[i])
            
            if i == n_bands - 1:
                high_edge = nyquist
            else:
                high_edge = np.sqrt(self.band_centers[i] * self.band_centers[i+1])
            
            self.band_edges.append((low_edge, high_edge))
        
        # Valores atuais de cada banda (para suavização)
        self.band_levels = np.zeros(n_bands, dtype=np.float32)
        self.band_peaks = np.zeros(n_bands, dtype=np.float32)
        
        # Parâmetros de suavização (decay para peaks)
        self.peak_decay = 0.95  # Fator de decaimento dos picos
        self.level_smoothing = 0.7  # Fator de suavização dos níveis
    
    def analyze(self, audio_chunk):
        """
        Analisa um chunk de áudio e retorna os níveis das 10 bandas.
        
        Args:
            audio_chunk: Array numpy com amostras de áudio
            
        Returns:
            Tupla (band_levels, band_peaks) com os níveis e picos de cada banda
        """
        if audio_chunk is None or len(audio_chunk) == 0:
            return self.band_levels.copy(), self.band_peaks.copy()
        
        # Calcula a FFT do chunk
        n_samples = len(audio_chunk)
        fft_size = n_samples
        
        # Aplica janela de Hamming para reduzir vazamento espectral
        windowed = audio_chunk * np.hamming(n_samples)
        
        # Calcula FFT
        fft = np.fft.fft(windowed, n=fft_size)
        
        # Calcula magnitude (espectro de potência)
        magnitude = np.abs(fft[:fft_size // 2])
        
        # Converte para dB (com proteção contra log(0))
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        
        # Calcula as frequências correspondentes aos bins da FFT
        freqs = np.fft.fftfreq(fft_size, 1.0 / self.sample_rate)[:fft_size // 2]
        
        # Calcula o nível de energia para cada banda
        new_levels = np.zeros(self.n_bands, dtype=np.float32)
        
        for i, (low_edge, high_edge) in enumerate(self.band_edges):
            # Encontra os bins da FFT que correspondem a esta banda
            mask = (freqs >= low_edge) & (freqs < high_edge)
            
            if np.any(mask):
                # Calcula a média da magnitude na banda (em dB)
                band_magnitude = np.mean(magnitude_db[mask])
                # Converte de dB para valor linear normalizado (0-1)
                # Normaliza considerando que o range típico é de -80 a 0 dB
                # Usa uma escala mais sensível para melhor visualização
                normalized_level = np.clip((band_magnitude + 80) / 80.0, 0.0, 1.0)
                new_levels[i] = normalized_level
            else:
                new_levels[i] = 0.0
        
        # Suaviza os níveis para evitar flickering
        self.band_levels = (self.level_smoothing * self.band_levels + 
                           (1 - self.level_smoothing) * new_levels)
        
        # Atualiza os picos (com decaimento)
        self.band_peaks = np.maximum(self.band_peaks * self.peak_decay, new_levels)
        
        return self.band_levels.copy(), self.band_peaks.copy()
    
    def get_band_frequencies(self):
        """
        Retorna as frequências centrais das bandas.
        
        Returns:
            Array com as frequências centrais em Hz
        """
        return self.band_centers.copy()

