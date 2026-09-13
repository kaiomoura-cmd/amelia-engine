"""
Pacote de Interpretação — A.M.E.L.I.A.

Separa os módulos de:
- Transcricao (STT):  Whisper local + fallback Google
- GeracaoVoz (TTS):  XTTSv2 (subprocesso) + fallback Edge-TTS
"""

from .transcricao import Transcricao
from .geracao_voz import GeracaoVoz

__all__ = ["Transcricao", "GeracaoVoz"]
