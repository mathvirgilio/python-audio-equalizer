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

## Estrutura do Código

- `equalizer.py`: Script principal com todas as funcionalidades
- `exemplo_uso.py`: Exemplo simples de uso
- `requirements.txt`: Dependências do projeto