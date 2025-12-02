"""
Equalizador em Tempo Real com Interface Gráfica
Processa áudio em tempo real aplicando filtros de 5 bandas com ganhos ajustáveis em dB.
Suporta entrada de microfone ou arquivo de áudio.
"""

import numpy as np
import pyaudio
import threading
import time
import tkinter as tk
from tkinter import messagebox, filedialog
from equalizer import create_frequency_filter, calculate_cutoff_frequencies
import librosa
import os


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
        # Usa largura de banda proporcional para melhor cobertura
        # Para cada frequência, usa aproximadamente 2/3 de oitava para efeitos mais perceptíveis
        # Isso garante que cada banda tenha uma cobertura adequada
        self.bandwidth_factor = 0.6  # Fator para calcular largura de banda proporcional (maior = mais larga)
        
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
        As frequências de corte são calculadas no meio entre as frequências centrais das bandas,
        conforme especificação do projeto.
        """
        self.band_filters_fft = []
        self.bandwidths = []  # Armazena a largura de banda de cada filtro
        self.cutoff_frequencies = []  # Armazena as frequências de corte
        
        # Calcula as frequências de corte que ficam no meio entre as frequências centrais
        cutoff_freqs = calculate_cutoff_frequencies(self.center_frequencies, self.sample_rate)
        
        for i, center_freq in enumerate(self.center_frequencies):
            low_cutoff, high_cutoff = cutoff_freqs[i]
            self.cutoff_frequencies.append((low_cutoff, high_cutoff))
            
            # Calcula largura de banda efetiva
            bandwidth = high_cutoff - low_cutoff
            self.bandwidths.append(bandwidth)
            
            # Usa create_frequency_filter do equalizer.py com frequências de corte específicas
            filter_response = create_frequency_filter(
                self.fft_size, 
                self.sample_rate, 
                center_freq, 
                bandwidth=bandwidth,  # Mantido para compatibilidade
                low_cutoff=low_cutoff,
                high_cutoff=high_cutoff,
                filter_shape=self.filter_shape
            )
            
            # Armazena o filtro no domínio da frequência
            self.band_filters_fft.append(filter_response)
        
        print(f"Filtros pré-calculados para {len(self.center_frequencies)} bandas")
        print(f"Frequências centrais: {self.center_frequencies} Hz")
        print(f"Frequências de corte (low, high):")
        for i, (low, high) in enumerate(self.cutoff_frequencies):
            print(f"  Banda {i+1} ({self.center_frequencies[i]} Hz): {low:.1f} - {high:.1f} Hz")
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
        
        # Cria o filtro combinado usando abordagem paramétrica
        # Quando gain_db = 0, o filtro não altera o sinal (resposta = 1.0)
        # Quando gain_db > 0, amplifica aquela banda
        # Quando gain_db < 0, atenua aquela banda
        # O filtro é real (não complexo) pois os filtros de frequência são reais
        combined_filter = np.ones(self.fft_size, dtype=np.float64)
        
        for i, filter_response in enumerate(self.band_filters_fft):
            # Converte ganho em dB para amplificação linear: A_i = 10^(AdB/20)
            gain_db = self.gains_db[i]
            
            # Se o ganho for 0 dB, não altera nada (pula esta banda)
            if abs(gain_db) < 0.001:  # Praticamente zero
                continue
            
            gain_linear = self._db_to_linear(gain_db)
            
            # Aplica o ganho usando abordagem paramétrica
            # Para boost: adiciona o sinal filtrado amplificado
            # Para cut: reduz o sinal filtrado
            # Quando gain_db é muito negativo (ex: -30 dB), gain_linear ≈ 0.032
            # Isso resulta em atenuação de ~97%, tornando a banda praticamente inaudível
            if gain_db > 0:
                # Boost: eq_response = 1.0 + filter_response * (gain_linear - 1.0)
                combined_filter += filter_response * (gain_linear - 1.0)
            else:
                # Cut: eq_response = 1.0 - filter_response * (1.0 - gain_linear)
                # Para valores muito negativos, isso resulta em atenuação quase total
                combined_filter -= filter_response * (1.0 - gain_linear)
        
        # Converte o sinal do buffer para o domínio da frequência
        audio_fft = np.fft.fft(self.input_buffer)
        
        # Aplica o filtro combinado: Y[k] = X[k] * H[k]
        # onde H[k] é o filtro paramétrico combinado
        # Multiplica elemento por elemento (NumPy faz casting automático de real para complex)
        filtered_fft = audio_fft * combined_filter
        
        # Debug: mostra estatísticas do filtro (apenas ocasionalmente para não poluir o console)
        if np.random.random() < 0.01:  # 1% das vezes
            print(f"Filtro - Min: {np.min(combined_filter):.3f}, Max: {np.max(combined_filter):.3f}, Mean: {np.mean(combined_filter):.3f}")
        
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
    
    def start_processing(self, input_device=None, output_device=None, audio_file=None):
        """
        Inicia o processamento em tempo real.
        
        Args:
            input_device: Índice do dispositivo de entrada (None = padrão, ignorado se audio_file fornecido)
            output_device: Índice do dispositivo de saída (None = padrão)
            audio_file: Caminho do arquivo de áudio para reproduzir (None = usa microfone)
        """
        if self.is_processing:
            return
        
        self.is_processing = True
        self.audio_file = audio_file
        self.audio_data = None
        self.audio_index = 0
        self.audio_lock = threading.Lock()
        
        # Se um arquivo foi fornecido, carrega o áudio
        if audio_file:
            try:
                print(f"Carregando arquivo: {audio_file}")
                self.audio_data, file_sr = librosa.load(audio_file, sr=None, mono=True)
                
                # Se a taxa de amostragem for diferente, resampleia
                if file_sr != self.sample_rate:
                    print(f"Resampleando de {file_sr} Hz para {self.sample_rate} Hz")
                    self.audio_data = librosa.resample(self.audio_data, orig_sr=file_sr, target_sr=self.sample_rate)
                
                self.audio_index = 0
                print(f"Áudio carregado: {len(self.audio_data) / self.sample_rate:.2f} segundos")
            except Exception as e:
                print(f"Erro ao carregar arquivo: {e}")
                self.is_processing = False
                raise
        
        # Inicializa PyAudio
        self.p = pyaudio.PyAudio()
        
        # Abre stream de áudio
        # Se usar arquivo, não precisa de input
        self.audio_stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,  # Mono
            rate=self.sample_rate,
            input=audio_file is None,  # Só usa input se não houver arquivo
            output=True,
            frames_per_buffer=self.chunk_size,
            input_device_index=input_device if audio_file is None else None,
            output_device_index=output_device,
            stream_callback=self._audio_callback if audio_file is None else None
        )
        
        # Inicia o stream
        self.audio_stream.start_stream()
        
        if audio_file:
            # Para arquivo, usa thread separada para alimentar o stream
            self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
            self.playback_thread.start()
        
        print("Processamento em tempo real iniciado")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        Callback chamado pelo PyAudio para cada bloco de áudio (apenas para microfone).
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
    
    def _playback_loop(self):
        """
        Loop de reprodução para arquivo de áudio (executado em thread separada).
        """
        try:
            while self.is_processing and self.audio_data is not None:
                with self.audio_lock:
                    # Pega um chunk do áudio
                    chunk_end = min(self.audio_index + self.chunk_size, len(self.audio_data))
                    audio_chunk = self.audio_data[self.audio_index:chunk_end]
                    
                    # Se o chunk for menor que chunk_size, preenche com zeros ou reinicia
                    if len(audio_chunk) < self.chunk_size:
                        padded_chunk = np.zeros(self.chunk_size, dtype=np.float32)
                        padded_chunk[:len(audio_chunk)] = audio_chunk
                        audio_chunk = padded_chunk
                        # Reinicia o áudio quando chegar ao fim
                        self.audio_index = 0
                    else:
                        self.audio_index = chunk_end
                
                # Processa o chunk com o equalizador
                processed_chunk = self.process_chunk(audio_chunk.astype(np.float32))
                
                # Escreve no stream de saída
                self.audio_stream.write(processed_chunk.tobytes())
                
                # Pequeno delay para evitar buffer underrun
                time.sleep(0.001)
                
        except Exception as e:
            print(f"Erro no loop de reprodução: {e}")
            self.is_processing = False
    
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
    
    def restart_audio(self):
        """Reinicia a reprodução do áudio do início."""
        if self.audio_data is not None:
            with self.audio_lock:
                self.audio_index = 0
    
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
    
    def __init__(self, audio_file=None):
        self.root = tk.Tk()
        self.root.title("Equalizador em Tempo Real - 5 Bandas")
        self.root.geometry("700x600")
        
        # Cria o equalizador
        self.equalizer = RealTimeEqualizer(sample_rate=44100, chunk_size=1024)
        
        # Variáveis para os sliders (em dB)
        self.slider_vars = []
        self.sliders = []
        
        # Arquivo de áudio
        self.audio_file = audio_file
        
        # Cria a interface
        self._create_ui()
        
        # Inicia o processamento
        try:
            self.equalizer.start_processing(audio_file=self.audio_file)
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
        
        # Frame para seleção de arquivo
        file_frame = tk.Frame(self.root, pady=5)
        file_frame.pack()
        
        # Label do arquivo (será atualizado)
        self.file_label = tk.Label(
            file_frame,
            text=f"Arquivo: {os.path.basename(self.audio_file)}" if self.audio_file else "Entrada: Microfone",
            font=("Arial", 9),
            fg="blue" if self.audio_file else "green"
        )
        self.file_label.pack(side=tk.LEFT, padx=5)
        
        # Botão de reiniciar (só aparece se houver arquivo)
        self.restart_button = None
        if self.audio_file:
            self.restart_button = tk.Button(
                file_frame,
                text="Reiniciar",
                command=self._restart_audio,
                font=("Arial", 8),
                padx=5,
                pady=2
            )
            self.restart_button.pack(side=tk.LEFT, padx=5)
        
        load_button = tk.Button(
            file_frame,
            text="Carregar Arquivo",
            command=self._load_audio_file,
            font=("Arial", 8),
            padx=5,
            pady=2
        )
        load_button.pack(side=tk.LEFT, padx=5)
        
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
            
            # Variável do slider (em dB, de -30 a +12)
            # -30 dB resulta em atenuação de ~97% (praticamente inaudível)
            var = tk.DoubleVar(value=0.0)
            self.slider_vars.append(var)
            
            # Slider vertical
            slider = tk.Scale(
                band_frame,
                from_=12,  # +12 dB (amplificação)
                to=-30,    # -30 dB (praticamente inaudível)
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
        # Debug: mostra os ganhos atuais
        print(f"Ganhos atualizados: {[f'{g:.1f}' for g in self.equalizer.gains_db]} dB")
    
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
    
    def _load_audio_file(self):
        """Carrega um novo arquivo de áudio."""
        filename = filedialog.askopenfilename(
            title="Selecionar arquivo de áudio",
            filetypes=[
                ("Arquivos de áudio", "*.mp3 *.wav *.flac *.ogg *.m4a"),
                ("Todos os arquivos", "*.*")
            ]
        )
        
        if filename:
            # Para o processamento atual
            self.equalizer.stop_processing()
            
            # Atualiza o arquivo
            self.audio_file = filename
            
            # Reinicia com o novo arquivo
            try:
                self.equalizer.start_processing(audio_file=self.audio_file)
                # Atualiza o label do arquivo
                self.file_label.config(
                    text=f"Arquivo: {os.path.basename(self.audio_file)}",
                    fg="blue"
                )
                # Adiciona botão de reiniciar se não existir
                if self.restart_button is None:
                    self.restart_button = tk.Button(
                        self.file_label.master,
                        text="Reiniciar",
                        command=self._restart_audio,
                        font=("Arial", 8),
                        padx=5,
                        pady=2
                    )
                    self.restart_button.pack(side=tk.LEFT, padx=5, before=self.file_label.master.winfo_children()[-1])
            except Exception as e:
                tk.messagebox.showerror("Erro", f"Erro ao carregar arquivo:\n{e}")
    
    def _restart_audio(self):
        """Reinicia a reprodução do áudio do início."""
        if self.audio_file:
            self.equalizer.restart_audio()
    
    def _on_closing(self):
        """Handler para fechar a janela."""
        self.equalizer.stop_processing()
        self.root.destroy()
    
    def run(self):
        """Inicia a interface gráfica."""
        self.root.mainloop()


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

