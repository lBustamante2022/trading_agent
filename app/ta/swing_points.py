import numpy as np
import pandas as pd

def find_swing_highs(highs: np.ndarray) -> list[int]:
    """
    Swing high clásico: high[i] > high[i-1] y high[i] > high[i+1]
    """
    idx = []
    for i in range(1, len(highs) - 1):
        if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
            idx.append(i)
    return idx


def find_swing_lows(lows: np.ndarray) -> list[int]:
    """
    Swing low clásico: low[i] < low[i-1] y low[i] < low[i+1]
    """
    idx = []
    for i in range(1, len(lows) - 1):
        if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
            idx.append(i)
    return idx

# import numpy as np

# def find_swing_highs(
#     highs: np.ndarray,
#     left: int = 2,
#     right: int = 2,
#     min_prominence_pct: float = 0.0,
#     min_separation: int = 3,
# ) -> list[int]:
#     """
#     Swing high mejorado:

#       - high[i] es el máximo en la ventana [i-left, i+right]
#       - opcional: prominencia mínima vs vecinos (min_prominence_pct)
#       - opcional: distancia mínima en velas entre swings (min_separation)
#     """
#     idx: list[int] = []
#     n = len(highs)
#     if n == 0:
#         return idx

#     last_idx = -10_000  # algo bien lejos

#     for i in range(left, n - right):
#         window = highs[i - left : i + right + 1]
#         center = highs[i]

#         # 1) máximo local en esa ventana
#         if center != window.max():
#             continue

#         # 2) prominencia mínima (opcional)
#         if min_prominence_pct > 0:
#             neigh_min = min(window[0], window[-1])
#             diff = center - neigh_min
#             if diff / center < min_prominence_pct:
#                 continue

#         # 3) separación mínima en barras (para evitar swings pegados)
#         if i - last_idx < min_separation:
#             # si este es más alto que el último swing, reemplazamos
#             if idx and center > highs[idx[-1]]:
#                 idx[-1] = i
#                 last_idx = i
#             continue

#         idx.append(i)
#         last_idx = i

#     return idx


# def find_swing_lows(
#     lows: np.ndarray,
#     left: int = 2,
#     right: int = 2,
#     min_prominence_pct: float = 0.0,
#     min_separation: int = 3,
# ) -> list[int]:
#     """
#     Swing low mejorado:

#       - low[i] es el mínimo en la ventana [i-left, i+right]
#       - opcional: prominencia mínima vs vecinos
#       - opcional: distancia mínima en velas entre swings
#     """
#     idx: list[int] = []
#     n = len(lows)
#     if n == 0:
#         return idx

#     last_idx = -10_000

#     for i in range(left, n - right):
#         window = lows[i - left : i + right + 1]
#         center = lows[i]

#         # 1) mínimo local en esa ventana
#         if center != window.min():
#             continue

#         # 2) prominencia mínima (opcional)
#         if min_prominence_pct > 0:
#             neigh_max = max(window[0], window[-1])
#             diff = neigh_max - center
#             if diff / neigh_max < min_prominence_pct:
#                 continue

#         # 3) separación mínima entre swings
#         if i - last_idx < min_separation:
#             # si este es más bajo que el último swing, lo reemplazamos
#             if idx and center < lows[idx[-1]]:
#                 idx[-1] = i
#                 last_idx = i
#             continue

#         idx.append(i)
#         last_idx = i

#     return idx