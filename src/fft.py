"""
Implementação da FFT (Fast Fourier Transform) compatível com numpy.fft.fft
Usa o algoritmo Cooley-Tukey para calcular a Transformada de Fourier Discreta (DFT)
"""

import numpy as np
from typing import Union, Optional


def fft(x: np.ndarray, n: Optional[int] = None, axis: int = -1, norm: Optional[str] = None) -> np.ndarray:
    """
    Calcula a FFT (Fast Fourier Transform) de um array.
    
    Compatível com numpy.fft.fft
    
    Parâmetros:
    -----------
    x : array_like
        Array de entrada (pode ser real ou complexo)
    n : int, opcional
        Tamanho da FFT. Se n < x.shape[axis], x é truncado.
        Se n > x.shape[axis], x é preenchido com zeros.
        Se None, usa x.shape[axis]
    axis : int, opcional
        Eixo ao longo do qual calcular a FFT. Padrão é -1 (último eixo)
    norm : str, opcional
        Normalização: 'ortho' para normalização ortogonal, None para padrão
        
    Retorna:
    --------
    out : ndarray
        Array complexo com a FFT do sinal de entrada
    """
    x = np.asarray(x)
    
    # Se n não foi especificado, usa o tamanho do array ao longo do axis
    if n is None:
        n = x.shape[axis] if axis >= 0 else x.shape[x.ndim + axis]
    
    # Prepara o array: truncar ou preencher com zeros se necessário
    if axis == -1 or axis == x.ndim - 1:
        # Caso mais comum: último eixo
        if x.shape[-1] < n:
            # Preenche com zeros
            pad_shape = list(x.shape)
            pad_shape[-1] = n - x.shape[-1]
            x = np.concatenate([x, np.zeros(pad_shape, dtype=x.dtype)], axis=-1)
        elif x.shape[-1] > n:
            # Trunca
            x = x[..., :n]
    else:
        # Para outros eixos, precisa de manipulação mais complexa
        # Por simplicidade, vamos mover o eixo para o final, processar e mover de volta
        x = np.moveaxis(x, axis, -1)
        if x.shape[-1] < n:
            pad_shape = list(x.shape)
            pad_shape[-1] = n - x.shape[-1]
            x = np.concatenate([x, np.zeros(pad_shape, dtype=x.dtype)], axis=-1)
        elif x.shape[-1] > n:
            x = x[..., :n]
        x = np.moveaxis(x, -1, axis)
    
    # Converte para complexo se necessário
    if not np.iscomplexobj(x):
        x = x.astype(np.complex128)
    
    # Aplica a FFT ao longo do eixo especificado
    if axis == -1 or axis == x.ndim - 1:
        result = _fft_1d(x)
    else:
        # Para outros eixos, move para o final, processa e move de volta
        x_moved = np.moveaxis(x, axis, -1)
        result_moved = _fft_1d(x_moved)
        result = np.moveaxis(result_moved, -1, axis)
    
    # Aplica normalização se especificada
    if norm == 'ortho':
        result = result / np.sqrt(n)
    
    return result


def ifft(x: np.ndarray, n: Optional[int] = None, axis: int = -1, norm: Optional[str] = None) -> np.ndarray:
    """
    Calcula a IFFT (Inverse Fast Fourier Transform) de um array.
    
    Compatível com numpy.fft.ifft
    
    Parâmetros:
    -----------
    x : array_like
        Array complexo de entrada
    n : int, opcional
        Tamanho da IFFT
    axis : int, opcional
        Eixo ao longo do qual calcular a IFFT
    norm : str, opcional
        Normalização: 'ortho' para normalização ortogonal, None para padrão
        
    Retorna:
    --------
    out : ndarray
        Array complexo com a IFFT do sinal de entrada
    """
    x = np.asarray(x)
    
    if n is None:
        n = x.shape[axis] if axis >= 0 else x.shape[x.ndim + axis]
    
    # IFFT é calculada como: IFFT(x) = conj(FFT(conj(x))) / n
    # Ou: IFFT(x) = FFT(conj(x)) / n (e depois tomar o conjugado)
    x_conj = np.conj(x)
    
    # Prepara o array similar à FFT
    if axis == -1 or axis == x.ndim - 1:
        if x_conj.shape[-1] < n:
            pad_shape = list(x_conj.shape)
            pad_shape[-1] = n - x_conj.shape[-1]
            x_conj = np.concatenate([x_conj, np.zeros(pad_shape, dtype=x_conj.dtype)], axis=-1)
        elif x_conj.shape[-1] > n:
            x_conj = x_conj[..., :n]
    else:
        x_conj = np.moveaxis(x_conj, axis, -1)
        if x_conj.shape[-1] < n:
            pad_shape = list(x_conj.shape)
            pad_shape[-1] = n - x_conj.shape[-1]
            x_conj = np.concatenate([x_conj, np.zeros(pad_shape, dtype=x_conj.dtype)], axis=-1)
        elif x_conj.shape[-1] > n:
            x_conj = x_conj[..., :n]
        x_conj = np.moveaxis(x_conj, -1, axis)
    
    if not np.iscomplexobj(x_conj):
        x_conj = x_conj.astype(np.complex128)
    
    # Calcula FFT do conjugado
    if axis == -1 or axis == x_conj.ndim - 1:
        result = _fft_1d(x_conj)
    else:
        x_moved = np.moveaxis(x_conj, axis, -1)
        result_moved = _fft_1d(x_moved)
        result = np.moveaxis(result_moved, -1, axis)
    
    # Aplica normalização
    if norm == 'ortho':
        result = np.conj(result) / np.sqrt(n)
    else:
        result = np.conj(result) / n
    
    return result


def _fft_1d(x: np.ndarray) -> np.ndarray:
    """
    Calcula a FFT 1D usando o algoritmo Cooley-Tukey (recursivo).
    
    Esta função processa arrays multi-dimensionais aplicando FFT
    ao longo da última dimensão.
    
    Parâmetros:
    -----------
    x : ndarray
        Array complexo (pode ser multi-dimensional)
        
    Retorna:
    --------
    out : ndarray
        Array complexo com a FFT aplicada ao longo da última dimensão
    """
    # Se o array é multi-dimensional, aplica recursivamente
    if x.ndim > 1:
        # Aplica FFT a cada "fatia" ao longo da última dimensão
        result = np.zeros_like(x, dtype=np.complex128)
        for idx in np.ndindex(x.shape[:-1]):
            result[idx] = _fft_1d_recursive(x[idx])
        return result
    else:
        return _fft_1d_recursive(x)


def _fft_1d_recursive(x: np.ndarray) -> np.ndarray:
    """
    Implementação recursiva da FFT 1D usando o algoritmo Cooley-Tukey.
    
    Parâmetros:
    -----------
    x : ndarray
        Array 1D complexo
        
    Retorna:
    --------
    out : ndarray
        Array 1D complexo com a FFT
    """
    n = len(x)
    
    # Caso base: se n é 1, retorna o próprio valor
    if n == 1:
        return x.copy()
    
    # Se n não é potência de 2, usa DFT direta (mais lento)
    # ou pode usar zero-padding para potência de 2
    if n & (n - 1) != 0:
        # Não é potência de 2 - usa implementação iterativa mais eficiente
        return _fft_radix2_iterative(x)
    
    # Divide e conquista: separa em pares e ímpares
    even = _fft_1d_recursive(x[::2])   # FFT dos elementos pares
    odd = _fft_1d_recursive(x[1::2])   # FFT dos elementos ímpares
    
    # Combina os resultados
    # W_n^k = exp(-2πik/n) são os fatores de rotação (twiddle factors)
    t = np.exp(-2j * np.pi * np.arange(n // 2) / n) * odd
    
    # Combina: resultado = [even + t, even - t]
    result = np.zeros(n, dtype=np.complex128)
    result[:n//2] = even + t
    result[n//2:] = even - t
    
    return result


def _fft_radix2_iterative(x: np.ndarray) -> np.ndarray:
    """
    Implementação iterativa da FFT usando radix-2.
    Mais eficiente para arrays que não são potência de 2.
    
    Parâmetros:
    -----------
    x : ndarray
        Array 1D complexo
        
    Retorna:
    --------
    out : ndarray
        Array 1D complexo com a FFT
    """
    n = len(x)
    
    # Se n é pequeno, usa DFT direta
    if n <= 8:
        return _dft_direct(x)
    
    # Encontra a próxima potência de 2 maior ou igual a n
    next_power_of_2 = 1
    while next_power_of_2 < n:
        next_power_of_2 <<= 1
    
    # Preenche com zeros até a próxima potência de 2
    if next_power_of_2 > n:
        x_padded = np.zeros(next_power_of_2, dtype=np.complex128)
        x_padded[:n] = x
        # Calcula FFT da versão preenchida
        fft_padded = _fft_1d_recursive(x_padded)
        # Retorna apenas os primeiros n elementos
        return fft_padded[:n]
    else:
        # Já é potência de 2
        return _fft_1d_recursive(x)


def _dft_direct(x: np.ndarray) -> np.ndarray:
    """
    Implementação direta da DFT (Discrete Fourier Transform).
    Usada para arrays pequenos ou quando a FFT não é aplicável.
    
    Parâmetros:
    -----------
    x : ndarray
        Array 1D complexo
        
    Retorna:
    --------
    out : ndarray
        Array 1D complexo com a DFT
    """
    n = len(x)
    k = np.arange(n)
    result = np.zeros(n, dtype=np.complex128)
    
    # DFT: X[k] = sum(x[n] * exp(-2πikn/N))
    for i in range(n):
        result[i] = np.sum(x * np.exp(-2j * np.pi * i * k / n))
    
    return result


def fftfreq(n: int, d: float = 1.0) -> np.ndarray:
    """
    Retorna as frequências da FFT.
    
    Compatível com numpy.fft.fftfreq
    
    Parâmetros:
    -----------
    n : int
        Tamanho da janela
    d : float
        Espaçamento da amostra (inverso da taxa de amostragem)
        
    Retorna:
    --------
    f : ndarray
        Array com as frequências
    """
    val = 1.0 / (n * d)
    results = np.empty(n, dtype=np.float64)
    N = (n - 1) // 2 + 1
    p1 = np.arange(0, N, dtype=np.int32)
    results[:N] = p1
    p2 = np.arange(-(n // 2), 0, dtype=np.int32)
    results[N:] = p2
    return results * val

