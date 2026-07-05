"""
torchaudio_compat.py — restore torchaudio.AudioMetaData for pyannote 3.1.1
==========================================================================
WhisperX 3.3.1 pins pyannote.audio 3.1.1, whose ``core/io.py`` references
``torchaudio.AudioMetaData`` in a return-type annotation that is evaluated at
import time. Recent torchaudio (the Colab default) relocated this class out of
the top-level namespace, so importing pyannote raises:

    AttributeError: module 'torchaudio' has no attribute 'AudioMetaData'

This shim re-attaches the class (at whichever internal location it now lives,
or a stub as a last resort) BEFORE pyannote is imported. It is idempotent and
a no-op on torchaudio versions that still expose the attribute (e.g. the local
2.1.x pin). Critically, it changes NO installed package — the pinned
whisperx / transformers / autoawq / torch / torchaudio versions are untouched,
so it cannot reintroduce a resolver conflict or break the AWQ kernels.

Usage — call apply() before the first `import whisperx` / `import pyannote`:

    import torchaudio_compat
    torchaudio_compat.apply()
    import whisperx
"""

import importlib


def apply() -> bool:
    """
    Ensure ``torchaudio.AudioMetaData`` exists.

    Returns True if a patch was applied, False if it was already present or
    torchaudio is unavailable.
    """
    try:
        import torchaudio
    except Exception:
        return False

    if hasattr(torchaudio, "AudioMetaData"):
        return False

    # The class still ships with torchaudio — only its public location moved
    # across the backend-dispatcher refactors. Try the known homes in order.
    audio_meta = None
    for modpath in ("torchaudio.backend.common", "torchaudio._backend.common"):
        try:
            audio_meta = importlib.import_module(modpath).AudioMetaData
            break
        except Exception:
            continue

    if audio_meta is None:
        # pyannote 3.1.1 only uses it as a type annotation, so a bare stub
        # satisfies the attribute lookup without affecting runtime behaviour.
        class AudioMetaData:  # noqa: D401 - minimal placeholder
            pass
        audio_meta = AudioMetaData

    torchaudio.AudioMetaData = audio_meta
    print("  [torchaudio_compat] patched torchaudio.AudioMetaData for pyannote 3.1.1")
    return True
