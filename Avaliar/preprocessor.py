"""
preprocessor.py
---------------
Pipeline de pré-processamento de imagem para o fikzPy.

Objetivo: transformar qualquer imagem rasterizada (foto, scan, line-art leve)
em uma máscara binária de boa qualidade antes de esqueletizar.

Problema resolvido aqui:
  - Threshold global (cv2.threshold com valor fixo) descarta pixels "cinza
    claro" que representam traços com pouca tinta ou originados de scans de
    baixo contraste.  A solução é um pipeline adaptativo que primeiro melhora
    o contraste local (CLAHE) e depois aplica um threshold adaptativo por
    bloco, que decide o limiar de forma independente em cada região da imagem.

Estratégias disponíveis
-----------------------
preprocess_adaptive  – recomendada para line-art, esboços e scans.
preprocess_classic   – equivalente ao modo anterior; threshold global simples.
preprocess_canny     – detecta bordas; útil para imagens fotográficas / sólidas.

Todos os métodos retornam um array numpy uint8 com shape (H, W), valores
0 (fundo) ou 255 (traço).
"""

from __future__ import annotations

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _to_gray(image: np.ndarray) -> np.ndarray:
    """Converte BGR, BGRA ou já-gray para grayscale uint8."""
    if image.ndim == 2:
        return image.astype(np.uint8)
    if image.shape[2] == 4:
        # canal alfa: compõe sobre fundo branco antes de converter
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        rgb = image[:, :, :3].astype(np.float32)
        white = np.ones_like(rgb) * 255.0
        composed = (alpha * rgb + (1 - alpha) * white).astype(np.uint8)
        return cv2.cvtColor(composed, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _safe_block_size(h: int, w: int, requested: int) -> int:
    """
    cv2.adaptiveThreshold exige block_size ímpar e >= 3.
    Garante que nunca exceda a menor dimensão da imagem.
    """
    bs = min(requested, min(h, w) - 1)
    if bs < 3:
        bs = 3
    if bs % 2 == 0:
        bs += 1
    return bs


# ---------------------------------------------------------------------------
# Pipeline adaptativo  (modo principal)
# ---------------------------------------------------------------------------

def preprocess_adaptive(
    image: np.ndarray,
    *,
    # CLAHE – equalização de histograma local
    clahe_clip: float = 3.0,          # clip limit; >4 começa a amplificar ruído
    clahe_tile: int = 8,              # grade NxN de tiles
    # Denoising leve antes do threshold
    denoise_h: float = 7.0,           # força do filtro Non-Local Means (3-10)
    denoise_template: int = 7,        # tamanho do patch (deve ser ímpar)
    denoise_search: int = 21,         # janela de busca (deve ser ímpar)
    # Threshold adaptativo
    block_size: int = 31,             # tamanho do bloco (ímpar); maior = mais tolerante a variação de iluminação
    C: float = 8,                     # constante subtraída da média local; maior = mais agressivo
    # Morfologia pós-threshold
    close_ksize: int = 3,             # fechamento para reconectar traços quebrados (0 = desativado)
    open_ksize: int = 2,              # abertura para remover ruído fino (0 = desativado)
    invert: bool = True,              # True = traços escuros sobre fundo claro (maioria dos casos)
) -> np.ndarray:
    """
    Pipeline adaptativo recomendado para line-art, esboços e scans.

    Etapas:
      1. Converte para escala de cinza.
      2. Aplica CLAHE para equalizar contraste local → traços leves tornam-se visíveis.
      3. Denoising Non-Local Means para suavizar ruído sem borrar contornos.
      4. Threshold adaptativo Gaussiano → binário robusto a iluminação não-uniforme.
      5. Morphological closing para reconectar traços fragmentados.
      6. Morphological opening para eliminar ruído de ponto único.

    Retorna máscara binária uint8 (255 = traço, 0 = fundo).
    """
    gray = _to_gray(image)
    h, w = gray.shape

    # --- 1. CLAHE -----------------------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
    enhanced = clahe.apply(gray)

    # --- 2. Denoising -------------------------------------------------------
    # fastNlMeansDenoising é eficaz mas pesado em imagens grandes.
    # Para imagens > 2048 px, reduzimos a força automaticamente.
    max_dim = max(h, w)
    h_param = denoise_h if max_dim < 2048 else denoise_h * 0.6
    denoised = cv2.fastNlMeansDenoising(
        enhanced,
        None,
        h=h_param,
        templateWindowSize=denoise_template,
        searchWindowSize=denoise_search,
    )

    # --- 3. Threshold adaptativo -------------------------------------------
    bs = _safe_block_size(h, w, block_size)
    # THRESH_BINARY_INV: pixels mais escuros que a média local → 255
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY,
        bs,
        C,
    )

    # --- 4. Closing (fecha buracos e reconecta traços próximos) ------------
    if close_ksize > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_ksize, close_ksize)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)

    # --- 5. Opening (remove pequenos artefatos isolados) -------------------
    if open_ksize > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_ksize, open_ksize)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)

    return binary


# ---------------------------------------------------------------------------
# Pipeline clássico  (compatibilidade com o modo anterior)
# ---------------------------------------------------------------------------

def preprocess_classic(
    image: np.ndarray,
    *,
    ink_threshold: int = 128,   # pixels mais escuros que este valor → traço
    invert: bool = True,
) -> np.ndarray:
    """
    Threshold global simples – equivalente ao comportamento original.

    Mantido para compatibilidade e como caminho de rollback.
    """
    gray = _to_gray(image)
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, ink_threshold, 255, flag)
    return binary


# ---------------------------------------------------------------------------
# Pipeline Canny  (detecção de bordas – útil para imagens fotográficas)
# ---------------------------------------------------------------------------

def preprocess_canny(
    image: np.ndarray,
    *,
    blur_ksize: int = 3,
    canny_low: int = 50,
    canny_high: int = 150,
    dilate_ksize: int = 1,   # dilata levemente as bordas para facilitar o trace
) -> np.ndarray:
    """
    Detecção de bordas com Canny.

    Útil para imagens com preenchimento sólido, fotografias ou geometrias
    onde o contorno é mais informativo do que o interior do traço.
    """
    gray = _to_gray(image)
    blurred = cv2.GaussianBlur(gray, (blur_ksize | 1, blur_ksize | 1), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    if dilate_ksize > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_ksize * 2 + 1, dilate_ksize * 2 + 1)
        )
        edges = cv2.dilate(edges, k)
    return edges


# ---------------------------------------------------------------------------
# API unificada
# ---------------------------------------------------------------------------

def preprocess(
    image: np.ndarray,
    mode: str = "adaptive",
    **kwargs,
) -> np.ndarray:
    """
    Ponto de entrada único para o pré-processamento.

    Parâmetros
    ----------
    image : np.ndarray
        Imagem lida com cv2.imread (BGR) ou carregada de outro modo.
    mode : str
        "adaptive"  – pipeline robusto para traços leves (padrão).
        "classic"   – threshold global simples (modo anterior).
        "canny"     – detecção de bordas Canny.
    **kwargs
        Parâmetros repassados à função correspondente.

    Retorna
    -------
    np.ndarray uint8, shape (H, W), valores 0 ou 255.
    """
    mode = mode.lower().strip()
    if mode == "adaptive":
        return preprocess_adaptive(image, **kwargs)
    if mode == "classic":
        return preprocess_classic(image, **kwargs)
    if mode == "canny":
        return preprocess_canny(image, **kwargs)
    raise ValueError(f"Modo desconhecido: {mode!r}. Use 'adaptive', 'classic' ou 'canny'.")
