"""Owner-only listening: a small classic voiceprint, no neural embeddings.

A speaker is verified by the shape of their voice, learned from a few
enrolled clips and compared like a fingerprint. The features are MFCCs --
the same mel-cepstrum classic speech systems have used for decades -- kept
deliberately small (13 coefficients, mean and spread over the clip, one
unit-normalised vector). Two recordings of one speaker sit close together;
two people sit apart. Cosine similarity over that vector, against a
threshold, is the whole decision.

This is not a security boundary and is not sold as one: a close imitator or
a good recording can pass it. It is a politeness gate -- the room's voices
should not wake the agent -- built to add nothing to install. It pulls no
new dependency and runs on the CPU beside the rest of the line. A neural
embedding would separate speakers far better, and is exactly the hundreds
of megabytes this stack refuses to carry.

Pure numpy, no I/O except the profile file, so it is testable on synthetic
audio with no mic.
"""
import json
import os
import time

import numpy as np

from backtalk.vlog import log

RATE = 16000
WIN_MS = 25
HOP_MS = 10
N_FFT = 512
N_MELS = 26
N_CEPS = 13          # includes c0, which is dropped before comparing
PREEMPH = 0.97
FMIN = 50.0
FMAX = 8000.0
PROFILE_VERSION = 1
DEFAULT_THRESHOLD = 0.75

# The phrases read during enrollment. Ordinary sentences, varied vowels,
# so the print sees the speaker's real mouth rather than one held note.
ENROLL_PROMPTS = [
    "Please say: the quick brown fox jumps over the lazy dog.",
    "Please say: I am the one who talks to you, and no one else is.",
    "Please say: today is a good day to get some work done.",
]
MIN_CLIPS = 2


def _hz2mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel2hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _mel_filterbank(n_mels=N_MELS, n_fft=N_FFT, rate=RATE,
                    fmin=FMIN, fmax=FMAX):
    """Triangular mel filters over the FFT bins. Built once at import."""
    edges = _mel2hz(np.linspace(_hz2mel(fmin), _hz2mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * edges / rate).astype(int)
    n_bins = n_fft // 2 + 1
    fb = np.zeros((n_mels, n_bins))
    for i in range(n_mels):
        lo, mid, hi = bins[i], bins[i + 1], bins[i + 2]
        if hi <= lo:
            continue
        for k in range(lo, mid):
            fb[i, k] = (k - lo) / (mid - lo + 1e-9)
        for k in range(mid, hi):
            fb[i, k] = (hi - k) / (hi - mid + 1e-9)
    return fb


def _dct_matrix(n, k):
    """DCT-II whose rows are the cepstral bases, orthonormal."""
    i = np.arange(k)[:, None]
    j = np.arange(n)[None, :]
    m = np.cos(np.pi * (j + 0.5) * i / n)
    m[0] *= np.sqrt(1.0 / n)
    m[1:] *= np.sqrt(2.0 / n)
    return m


_MEL = _mel_filterbank()
_DCT = _dct_matrix(N_MELS, N_MELS)[:N_CEPS]


def _frames(x, rate=RATE, win_ms=WIN_MS, hop_ms=HOP_MS):
    win = int(rate * win_ms / 1000)
    hop = int(rate * hop_ms / 1000)
    if x.size < win:
        return np.empty((0, win))
    n = 1 + (x.size - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def mfcc(pcm, rate=RATE) -> np.ndarray:
    """Mel-cepstral coefficients for one clip: frames x N_CEPS."""
    x = np.asarray(pcm, dtype=np.float64).ravel()
    if x.size == 0:
        return np.zeros((0, N_CEPS))
    x = np.append(x[0], x[1:] - PREEMPH * x[:-1])       # pre-emphasis
    fr = _frames(x, rate)
    if fr.shape[0] == 0:
        return np.zeros((0, N_CEPS))
    fr = fr * np.hamming(fr.shape[1])
    power = np.abs(np.fft.rfft(fr, n=N_FFT)) ** 2
    mel = power[:, :_MEL.shape[1]] @ _MEL.T
    ceps = np.log(mel + 1e-10) @ _DCT.T
    return ceps


def feature(pcm, rate=RATE) -> np.ndarray | None:
    """One unit-normalised voice vector for a clip, or None if too short.

    The shape of the cepstrum over time -- the mean of each coefficient and
    how it spreads -- is the print. c0 (frame energy) is dropped because it
    tracks loudness, not the voice, and would make the print follow the
    room instead of the person."""
    c = mfcc(pcm, rate)[:, 1:]
    if c.shape[0] < 4:
        return None
    v = np.concatenate([c.mean(axis=0), c.std(axis=0)])
    return v / float(np.linalg.norm(v))


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two prints. Both are unit vectors, so this is
    their dot product; kept as a named call for the threshold to read
    against."""
    return float(np.dot(a, b))


def combine(vectors) -> np.ndarray | None:
    """Average enrolled prints into one, unit-normalised. None if there are
    too few good clips to trust the result."""
    good = [v for v in vectors if v is not None]
    if len(good) < MIN_CLIPS:
        return None
    m = np.mean(np.stack(good), axis=0)
    n = float(np.linalg.norm(m))
    return m / n if n > 0 else None


def default_profile() -> str:
    """Where a print lives when the config names no file: beside the other
    user state, not in the repo."""
    return os.path.join(os.path.expanduser("~"), ".backtalk", "voiceprint.json")


def save_profile(path: str, vector: np.ndarray, rate=RATE):
    """Write the print. Never raises: a voice line that cannot save its
    print is worse off than one that silently does not gate."""
    d = {"version": PROFILE_VERSION, "rate": rate,
         "created": time.time(), "vector": np.asarray(vector).tolist()}
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w") as f:
            json.dump(d, f)
        return True
    except OSError as e:
        log(f"[voiceprint] could not save {path}: {e}")
        return False


def load_vector(path: str) -> np.ndarray | None:
    """Read a saved print, or None if it is missing or malformed."""
    try:
        with open(path) as f:
            d = json.load(f)
        if int(d.get("version", 0)) != PROFILE_VERSION:
            log(f"[voiceprint] {path} is an old print; enroll again")
            return None
        v = np.asarray(d["vector"], dtype=np.float64)
        return v if v.ndim == 1 and v.size else None
    except (OSError, ValueError, TypeError, KeyError):
        return None


def enroll(record_clip, prompts=ENROLL_PROMPTS) -> np.ndarray | None:
    """Build a print from enrollment clips.

    record_clip is called once per prompt and must return a clip (int16
    mono at RATE) or None. Injectable so the flow is tested without a mic.
    Returns None if too few clips were usable."""
    vectors = []
    for _ in prompts:
        pcm = record_clip()
        if pcm is not None:
            vectors.append(feature(pcm))
    return combine(vectors)


class Speaker:
    """A loaded print and the threshold it is judged against."""

    def __init__(self, vector: np.ndarray, threshold: float = DEFAULT_THRESHOLD):
        self.vector = vector
        self.threshold = threshold
        self.last_score = 0.0

    def score(self, pcm) -> float | None:
        f = feature(pcm)
        if f is None:
            return None
        self.last_score = similarity(self.vector, f)
        return self.last_score

    def check(self, pcm) -> bool:
        """True if the clip sounds like the enrolled speaker. A clip too
        short to print is not the owner (fail closed on the gate)."""
        s = self.score(pcm)
        return s is not None and s >= self.threshold


def load_speaker(cfg) -> Speaker | None:
    """The enrolled speaker named by the config, or None if there is no
    print yet (in which case nothing gates)."""
    path = cfg.get("speaker_profile") or default_profile()
    v = load_vector(path)
    if v is None:
        return None
    try:
        thr = float(cfg.get("speaker_threshold", DEFAULT_THRESHOLD))
    except (TypeError, ValueError):
        thr = DEFAULT_THRESHOLD
    return Speaker(v, thr)
