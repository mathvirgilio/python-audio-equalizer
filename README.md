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
python equalizer.py "The Cure - In Between Days.mp3" -f 100
```

Ou simplesmente execute sem argumentos (usa o arquivo padrão):

```bash
python equalizer.py
```

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

## Funcionalidades

- Suporta arquivos MP3, WAV e outros formatos suportados pelo librosa
- Filtro passa-banda (bandpass) - isola uma faixa de frequências
- Equalizador paramétrico - permite boost/cut em frequências específicas
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
- `multi_band_equalizer.py`: Equalizador de 5 bandas para arquivos
- `realtime_equalizer.py`: Equalizador em tempo real com interface gráfica
- `exemplo_uso.py`: Exemplo simples de uso
- `exemplo_multi_band.py`: Exemplo do equalizador multi-banda
- `exemplo_realtime.py`: Exemplo programático do equalizador em tempo real
- `requirements.txt`: Dependências do projeto