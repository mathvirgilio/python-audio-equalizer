# python-audio-equalizer

Equalizador de áudio em Python que permite filtrar arquivos de áudio por frequência.

## Instalação

Instale as dependências necessárias:

```bash
pip install -r requirements.txt
```

## Uso

### Uso Básico

Para filtrar o arquivo MP3 com um filtro centrado em 100 Hz:

```bash
python equalizer.py "tracks/The Cure - In Between Days.mp3" -f 100
```

Ou simplesmente execute sem argumentos (usa o arquivo padrão):

```bash
python equalizer.py
```

### Usando Outras Músicas

Para usar suas próprias músicas com o equalizador, siga estes passos:

1. **Adicione sua música à pasta `tracks`**:
   - Coloque seu arquivo de áudio (MP3, WAV, FLAC, etc.) na pasta `tracks/`
   - Exemplo: `tracks/MinhaMusica.mp3`

2. **Use o equalizador com sua música**:
   
   **Opção A - Especificando o caminho completo:**
   ```bash
   python equalizer.py "tracks/MinhaMusica.mp3" -f 100
   ```
   
   **Opção B - Usando o equalizador em tempo real:**
   ```bash
   python realtime_equalizer.py "tracks/MinhaMusica.mp3"
   ```
   
   **Opção C - Carregando pela interface gráfica:**
   - Execute `python realtime_equalizer.py`
   - Clique no botão "Carregar Arquivo" na interface
   - Selecione sua música da pasta `tracks/`

3. **Formatos suportados**:
   - MP3, WAV, FLAC, OGG, M4A e outros formatos suportados pelo librosa

**Dica:** Se você quiser que sua música seja o arquivo padrão, edite os arquivos `equalizer.py` ou `realtime_equalizer.py` e altere o caminho na variável `default_file`.

### Exemplo Simples

Execute o script de exemplo:

```bash
python exemplo_uso.py
```

### Opções Disponíveis

- `-f, --freq FLOAT`: Frequência central do filtro em Hz (padrão: 100)
- `-b, --bandwidth FLOAT`: Largura de banda em Hz (padrão: 50)
- `-o, --output FILE`: Nome do arquivo de saída (opcional)
- `-t, --type TYPE`: Tipo de filtro - `bandpass` ou `parametric` (padrão: bandpass)
- `-g, --gain FLOAT`: Ganho em dB para filtro paramétrico (padrão: 0)
- `-q, --q FLOAT`: Fator Q para filtro paramétrico (padrão: 1.0)

**Nota:** O filtro passa-banda usa por padrão a implementação matemática com `numpy.sinc` baseada na resposta ao impulso do filtro passa-faixa ideal.

### Exemplos

**Filtro passa-banda em 100 Hz:**
```bash
python equalizer.py "arquivo.mp3" -f 100 -b 50
```

**Equalizador paramétrico (boost de 6 dB em 100 Hz):**
```bash
python equalizer.py "arquivo.mp3" -f 100 -t parametric -g 6 -q 2.0
```

**Filtro com largura de banda personalizada:**
```bash
python equalizer.py "arquivo.mp3" -f 100 -b 30
```

## Implementação do Filtro Passa-Faixa

O filtro passa-banda implementa a resposta ao impulso matemática de um filtro passa-faixa ideal usando `numpy.sinc`:

### Resposta ao Impulso

A resposta ao impulso `h[n]` é definida como:

```
h[n] = (sin(ω_c2 n) - sin(ω_c1 n)) / (nπ)  quando n ≠ 0
h[n] = (ω_c2 - ω_c1) / π                    quando n = 0
```

Onde:
- `ω_c1 = 2π f_c1 / fs` é a frequência de corte inferior normalizada (radianos por amostra)
- `ω_c2 = 2π f_c2 / fs` é a frequência de corte superior normalizada (radianos por amostra)
- `f_c1` e `f_c2` são as frequências de corte em Hz
- `fs` é a taxa de amostragem

### Implementação com numpy.sinc

A implementação utiliza `numpy.sinc` para calcular eficientemente:

```
sin(ω n) / (nπ) = (ω / π) * sinc(ω n / π)
```

Onde `sinc(x) = sin(πx) / (πx)` é a função sinc normalizada.

### Processamento

1. A resposta ao impulso é calculada usando a fórmula matemática acima
2. Uma janela Hamming é aplicada para reduzir artefatos de ringing
3. O filtro é convertido para o domínio da frequência via FFT
4. O sinal é filtrado multiplicando no domínio da frequência

## Funcionalidades

- Suporta arquivos MP3, WAV e outros formatos suportados pelo librosa
- **Filtro passa-banda (bandpass)** - isola uma faixa de frequências usando resposta ao impulso matemática com `numpy.sinc` (padrão)
- **Filtros alternativos** - suporta também filtros gaussianos e retangulares
- **Equalizador paramétrico** - permite boost/cut em frequências específicas
- Preserva múltiplos canais (estéreo)
- Saída em formato WAV de alta qualidade

## Equalizador em Tempo Real

O projeto inclui um equalizador gráfico em tempo real com interface gráfica:

```bash
python realtime_equalizer.py
```

### Características do Equalizador em Tempo Real

- **5 Bandas**: 100 Hz, 330 Hz, 1 kHz, 3.3 kHz, 10 kHz
- **Ganho em dB**: Sliders de -12 dB a +12 dB
- **Processamento em tempo real**: Captura e reproduz áudio em tempo real
- **Filtros pré-calculados**: Filtros passa-banda Butterworth otimizados
- **Conversão dB → Linear**: Usa a fórmula A = 10^(AdB/20)

### Equação Implementada

O equalizador implementa a equação:
```
y[n] = sum_{i=1}^{5} A_i * (x[n] * h_i[n])
```

Onde:
- `x[n]` é o sinal de entrada
- `h_i[n]` é o filtro passa-banda da banda i
- `A_i = 10^(AdB/20)` é a amplificação linear calculada a partir do ganho em dB
- `y[n]` é o sinal de saída equalizado

### Exemplo Programático

```python
from realtime_equalizer import RealTimeEqualizer

# Cria o equalizador
eq = RealTimeEqualizer(sample_rate=44100, chunk_size=1024)

# Define ganhos em dB
eq.set_band_gain_db(0, 6.0)   # 100 Hz: +6 dB
eq.set_band_gain_db(2, -3.0)  # 1000 Hz: -3 dB

# Inicia processamento
eq.start_processing()
# ... processa áudio ...
eq.stop_processing()
```

## Estrutura do Código

- `equalizer.py`: Script principal com filtros de frequência
  - Implementa filtro passa-faixa usando resposta ao impulso com `numpy.sinc`
  - Suporta filtros gaussianos e retangulares alternativos
  - Equalizador paramétrico para boost/cut
- `realtime_equalizer.py`: Equalizador em tempo real com interface gráfica
  - 5 bandas equalizáveis (100 Hz, 330 Hz, 1 kHz, 3.3 kHz, 10 kHz)
  - Interface gráfica com sliders para ajuste de ganho
- `requirements.txt`: Dependências do projeto