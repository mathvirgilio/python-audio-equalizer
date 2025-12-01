"""
Equalizador em Tempo Real com Interface Gráfica
Processa áudio em tempo real aplicando filtros de 5 bandas com ganhos ajustáveis em dB.
"""

import numpy as np
import pyaudio
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from scipy import signal
from collections import deque
from equalizer import create_frequency_filter


class RealTimeEqualizer:
    """
    Equalizador de 5 bandas para processamento em tempo real.
    Cada banda é filtrada e amplificada/atenuada conforme ganho em dB.
    """
    
    def __init__(self, sample_rate=44100, chunk_size=1024):
        """
        Inicializa o equalizador em tempo real.
        
        Args:
            sample_rate: Taxa de amostragem (Hz) - padrão: 44100
            chunk_size: Tamanho do bloco de processamento - padrão: 1024
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
        # Frequências centrais das 5 bandas (Hz)
        self.center_frequencies = [100, 330, 1000, 3300, 10000]
        
        # Ganhos em dB para cada banda (inicialmente 0 dB = sem alteração)
        self.gains_db = [0.0] * len(self.center_frequencies)
        
        # Largura de banda de cada filtro (Hz)
        self.bandwidth = 50
        
        # Forma do filtro ('gaussian' ou 'rectangular')
        self.filter_shape = 'gaussian'
        
        # Pré-calcula os filtros no domínio da frequência para cada banda
        # Usa um tamanho de FFT maior que o chunk para melhor resolução
        self.fft_size = chunk_size * 2  # Tamanho da FFT (pode ser ajustado)
        self.band_filters_fft = []
        self._precompute_filters()
        
        # Buffer para processamento com overlap-add
        # Usa um buffer deslizante para acumular chunks antes de processar
        self.input_buffer = np.zeros(self.fft_size, dtype=np.float32)
        self.output_buffer = np.zeros(self.fft_size, dtype=np.float32)
        self.hop_size = chunk_size  # Tamanho do avanço (igual ao chunk)
        
        # Estado do processamento
        self.is_processing = False
        self.audio_stream = None
        
    def _precompute_filters(self):
        """
        Pré-calcula os filtros passa-banda no domínio da frequência usando create_frequency_filter.
        Usa a mesma função do equalizer.py, como em multi_band_equalizer.py.
        """
        self.band_filters_fft = []
        
        for center_freq in self.center_frequencies:
            # Usa create_frequency_filter do equalizer.py
            filter_response = create_frequency_filter(
                self.fft_size, 
                self.sample_rate, 
                center_freq, 
                self.bandwidth, 
                self.filter_shape
            )
            
            # Armazena o filtro no domínio da frequência
            self.band_filters_fft.append(filter_response)
        
        print(f"Filtros pré-calculados para {len(self.center_frequencies)} bandas")
        print(f"Frequências: {self.center_frequencies} Hz")
        print(f"Usando create_frequency_filter do equalizer.py (FFT size: {self.fft_size})")
    
    def set_band_gain_db(self, band_index, gain_db):
        """
        Define o ganho em dB para uma banda específica.
        
        Args:
            band_index: Índice da banda (0-4)
            gain_db: Ganho em dB (positivo = amplificação, negativo = atenuação)
        """
        if 0 <= band_index < len(self.center_frequencies):
            self.gains_db[band_index] = gain_db
        else:
            raise ValueError(f"Índice de banda deve estar entre 0 e {len(self.center_frequencies) - 1}")
    
    def _db_to_linear(self, gain_db):
        """
        Converte ganho em dB para amplificação linear.
        
        Fórmula: A = 10^(AdB/20)
        
        Args:
            gain_db: Ganho em dB
            
        Returns:
            Amplificação linear
        """
        return 10.0 ** (gain_db / 20.0)
    
    def process_chunk(self, audio_chunk):
        """
        Processa um bloco de áudio aplicando o equalizador usando FFT.
        
        Aplica a equação: y[n] = sum_{i=1}^{5} A_i * (x[n] * h_i[n])
        onde A_i é a amplificação linear calculada a partir do ganho em dB.
        A_i = 10^(AdB/20)
        
        Usa create_frequency_filter do equalizer.py, como em multi_band_equalizer.py.
        Processa em blocos usando FFT com overlap-add para processamento em tempo real.
        
        Args:
            audio_chunk: Array numpy com amostras de áudio (mono)
            
        Returns:
            Áudio processado
        """
        if len(audio_chunk) == 0:
            return audio_chunk
        
        # Adiciona o novo chunk ao buffer de entrada (deslizante)
        # Move o buffer para a esquerda e adiciona o novo chunk no final
        self.input_buffer[:-self.hop_size] = self.input_buffer[self.hop_size:]
        self.input_buffer[-self.hop_size:] = audio_chunk
        
        # Cria o filtro combinado no domínio da frequência
        # Similar ao multi_band_equalizer.py: h[n] = sum_{i=1}^{5} A_i * h_i[n]
        combined_filter = np.zeros(self.fft_size, dtype=np.complex128)
        
        for i, filter_response in enumerate(self.band_filters_fft):
            # Converte ganho em dB para amplificação linear: A_i = 10^(AdB/20)
            gain_linear = self._db_to_linear(self.gains_db[i])
            
            # Adiciona ao filtro combinado: A_i * h_i[n]
            combined_filter += gain_linear * filter_response
        
        # Converte o sinal do buffer para o domínio da frequência
        audio_fft = np.fft.fft(self.input_buffer)
        
        # Aplica o filtro combinado: Y[k] = X[k] * H[k]
        # onde H[k] = sum_{i=1}^{5} A_i * H_i[k]
        filtered_fft = audio_fft * combined_filter
        
        # Converte de volta para o domínio do tempo
        filtered_audio = np.real(np.fft.ifft(filtered_fft))
        
        # Aplica overlap-add: adiciona a parte de overlap do buffer de saída anterior
        output = filtered_audio[:self.hop_size] + self.output_buffer[:self.hop_size]
        
        # Salva a parte de overlap para o próximo chunk
        self.output_buffer[:-self.hop_size] = filtered_audio[self.hop_size:]
        self.output_buffer[-self.hop_size:] = 0
        
        # Normaliza para evitar clipping
        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output = output / max_val
        
        return output.astype(np.float32)
    
    def start_processing(self, input_device=None, output_device=None):
        """
        Inicia o processamento em tempo real.
        
        Args:
            input_device: Índice do dispositivo de entrada (None = padrão)
            output_device: Índice do dispositivo de saída (None = padrão)
        """
        if self.is_processing:
            return
        
        self.is_processing = True
        
        # Inicializa PyAudio
        self.p = pyaudio.PyAudio()
        
        # Abre stream de áudio
        self.audio_stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,  # Mono
            rate=self.sample_rate,
            input=True,
            output=True,
            frames_per_buffer=self.chunk_size,
            input_device_index=input_device,
            output_device_index=output_device,
            stream_callback=self._audio_callback
        )
        
        self.audio_stream.start_stream()
        print("Processamento em tempo real iniciado")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        Callback chamado pelo PyAudio para cada bloco de áudio.
        """
        if not self.is_processing:
            return (None, pyaudio.paComplete)
        
        # Converte bytes para array numpy
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        # Processa o bloco de áudio
        processed_audio = self.process_chunk(audio_data)
        
        # Converte de volta para bytes
        output_data = processed_audio.tobytes()
        
        return (output_data, pyaudio.paContinue)
    
    def stop_processing(self):
        """Para o processamento em tempo real."""
        if not self.is_processing:
            return
        
        self.is_processing = False
        
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        
        if self.p:
            self.p.terminate()
        
        print("Processamento em tempo real parado")
    
    def get_band_info(self):
        """
        Retorna informações sobre as bandas.
        
        Returns:
            Lista de tuplas (frequência, ganho_db)
        """
        return [(freq, gain) for freq, gain in 
                zip(self.center_frequencies, self.gains_db)]


class EqualizerGUI:
    """
    Interface gráfica para o equalizador em tempo real.
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Equalizador em Tempo Real - 5 Bandas")
        self.root.geometry("600x500")
        
        # Cria o equalizador
        self.equalizer = RealTimeEqualizer(sample_rate=44100, chunk_size=1024)
        
        # Variáveis para os sliders (em dB)
        self.slider_vars = []
        self.sliders = []
        
        # Cria a interface
        self._create_ui()
        
        # Inicia o processamento
        try:
            self.equalizer.start_processing()
        except Exception as e:
            print(f"Erro ao iniciar processamento: {e}")
            tk.messagebox.showerror("Erro", f"Não foi possível iniciar o processamento de áudio:\n{e}")
    
    def _create_ui(self):
        """Cria a interface gráfica."""
        # Título
        title_label = tk.Label(
            self.root, 
            text="EQUALIZADOR EM TEMPO REAL",
            font=("Arial", 16, "bold"),
            pady=10
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            self.root,
            text="5 Bandas - Ganho em dB",
            font=("Arial", 10),
            pady=5
        )
        subtitle_label.pack()
        
        # Frame para os sliders
        sliders_frame = tk.Frame(self.root, padx=20, pady=20)
        sliders_frame.pack(fill=tk.BOTH, expand=True)
        
        # Cria um slider para cada banda
        frequencies = self.equalizer.center_frequencies
        
        for i, freq in enumerate(frequencies):
            # Frame para cada banda
            band_frame = tk.Frame(sliders_frame)
            band_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
            
            # Label da frequência
            freq_label = tk.Label(
                band_frame,
                text=f"{freq} Hz" if freq < 1000 else f"{freq/1000:.1f} kHz",
                font=("Arial", 9, "bold")
            )
            freq_label.pack()
            
            # Variável do slider (em dB, de -12 a +12)
            var = tk.DoubleVar(value=0.0)
            self.slider_vars.append(var)
            
            # Slider vertical
            slider = tk.Scale(
                band_frame,
                from_=12,  # +12 dB
                to=-12,    # -12 dB
                resolution=0.5,  # Passo de 0.5 dB
                orient=tk.VERTICAL,
                length=300,
                variable=var,
                command=lambda val, idx=i: self._on_slider_change(idx, float(val)),
                tickinterval=6,
                showvalue=True,
                label="dB"
            )
            slider.pack()
            self.sliders.append(slider)
            
            # Label do valor atual
            value_label = tk.Label(
                band_frame,
                text="0.0 dB",
                font=("Arial", 8),
                fg="blue"
            )
            value_label.pack()
            self.slider_vars[i].trace('w', lambda *args, idx=i, lbl=value_label: 
                                     self._update_value_label(idx, lbl))
        
        # Frame para controles
        controls_frame = tk.Frame(self.root, pady=10)
        controls_frame.pack()
        
        # Botão para resetar todos os sliders
        reset_button = tk.Button(
            controls_frame,
            text="Resetar Todos (0 dB)",
            command=self._reset_all,
            font=("Arial", 10),
            padx=10,
            pady=5
        )
        reset_button.pack(side=tk.LEFT, padx=5)
        
        # Label de status
        self.status_label = tk.Label(
            controls_frame,
            text="● Processando",
            font=("Arial", 9),
            fg="green"
        )
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Handler para fechar a janela
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _on_slider_change(self, band_index, gain_db):
        """Callback quando um slider é movido."""
        self.equalizer.set_band_gain_db(band_index, gain_db)
    
    def _update_value_label(self, band_index, label):
        """Atualiza o label do valor do slider."""
        value = self.slider_vars[band_index].get()
        label.config(text=f"{value:.1f} dB")
    
    def _reset_all(self):
        """Reseta todos os sliders para 0 dB."""
        for var in self.slider_vars:
            var.set(0.0)
        
        # Atualiza o equalizador
        for i in range(len(self.equalizer.center_frequencies)):
            self.equalizer.set_band_gain_db(i, 0.0)
    
    def _on_closing(self):
        """Handler para fechar a janela."""
        self.equalizer.stop_processing()
        self.root.destroy()
    
    def run(self):
        """Inicia a interface gráfica."""
        self.root.mainloop()


def main():
    """Função principal."""
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
    print("\nIniciando interface gráfica...")
    
    app = EqualizerGUI()
    app.run()


if __name__ == '__main__':
    main()

