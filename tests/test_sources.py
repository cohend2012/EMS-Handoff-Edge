"""FileAudioSource: a WAV clip delivered like the mic, at mic pace, with an end signal."""

import threading
import time
import wave

import numpy as np
import pytest

from conftest import wait_for
from ems_handoff.audio.sources import AudioFileError, FileAudioSource, load_wav
from fakes import RATE, tone


def write_wav(path, audio, rate=RATE, channels=1):
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    if channels == 2:
        pcm = np.column_stack([pcm, pcm])
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return path


@pytest.fixture
def clip(tmp_path):
    return write_wav(tmp_path / "clip.wav", tone(1.0))


def drain(q):
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    return items


# -- delivery ---------------------------------------------------------------
def test_blocks_match_the_mic_and_reach_every_subscriber(clip):
    src = FileAudioSource(clip)
    q1, q2 = src.subscribe(), src.subscribe()
    src.start()
    assert wait_for(lambda: src.ended, timeout=3)
    audio = src.stop()
    b1, b2 = drain(q1), drain(q2)
    assert b1[-1] is None and b2[-1] is None, "listeners are told the stream ended"
    blocks = b1[:-1]
    assert len(blocks) == 10 and all(len(b) == 1600 for b in blocks)   # 1.0 s in 100 ms blocks
    assert len(b2) == len(b1)
    assert src.blocks_sent == 10
    np.testing.assert_array_equal(audio, load_wav(clip))                # the full recording is kept
    assert abs(src.seconds - 1.0) < 1e-6
    assert src.level > 0.01
    assert src.device_name == "file: clip.wav"


def test_last_block_is_shorter_when_the_clip_does_not_divide(tmp_path):
    src = FileAudioSource(write_wav(tmp_path / "odd.wav", tone(0.25)), pace=0)
    q = src.subscribe()
    src.start()
    assert wait_for(lambda: src.ended)
    src.stop()
    lengths = [len(b) for b in drain(q) if b is not None]
    assert lengths == [1600, 1600, 800]


def test_blocks_are_released_at_real_time_pace(clip):
    src = FileAudioSource(clip)
    q = src.subscribe()
    arrivals = []

    def collect():
        while True:
            b = q.get()
            if b is None:
                return
            arrivals.append(time.monotonic())

    t = threading.Thread(target=collect, daemon=True)
    t.start()
    t0 = time.monotonic()
    src.start()
    assert wait_for(lambda: src.ended, timeout=3)
    src.stop()
    t.join(timeout=2)
    total = arrivals[-1] - t0
    assert 0.85 <= total <= 1.3, f"10 blocks of 100 ms took {total:.2f} s"
    gaps = np.diff(arrivals)
    assert abs(float(np.median(gaps)) - 0.1) < 0.03
    assert gaps.max() < 0.25


def test_pace_zero_plays_as_fast_as_possible(clip):
    src = FileAudioSource(clip, pace=0)
    src.subscribe()
    t0 = time.monotonic()
    src.start()
    assert wait_for(lambda: src.ended)
    src.stop()
    assert time.monotonic() - t0 < 0.5


# -- the end signal ---------------------------------------------------------
def test_on_end_fires_once_after_the_last_block(clip):
    ends = []
    src = FileAudioSource(clip, pace=0)
    src.set_on_end(lambda: ends.append(src.blocks_sent))
    q = src.subscribe()
    src.start()
    assert wait_for(lambda: ends)
    assert ends == [10], "every block was delivered before the end signal"
    src.stop()
    src.stop()                                       # idempotent
    assert ends == [10]
    assert [b for b in drain(q) if b is None] == [None]


def test_stop_before_the_end_cuts_playback_and_skips_on_end(tmp_path):
    ends = []
    src = FileAudioSource(write_wav(tmp_path / "long.wav", tone(3.0)), on_end=lambda: ends.append(1))
    q = src.subscribe()
    src.start()
    assert wait_for(lambda: src.blocks_sent >= 2)
    audio = src.stop()
    assert not src.ended and ends == []
    assert 2 <= len(audio) / 1600 < 30
    assert drain(q)[-1] is None


# -- file handling ----------------------------------------------------------
def test_other_sample_rates_are_resampled_to_16k(tmp_path):
    path = write_wav(tmp_path / "48k.wav", tone(1.5), rate=48000)   # 24000 frames = 0.5 s at 48 kHz
    src = FileAudioSource(path, pace=0)
    src.start()
    assert wait_for(lambda: src.ended)
    assert len(src.stop()) == 8000
    assert src.duration == 0.5


def test_stereo_and_unreadable_files_are_refused(tmp_path):
    with pytest.raises(AudioFileError, match="must be mono"):
        FileAudioSource(write_wav(tmp_path / "st.wav", tone(0.2), channels=2)).start()
    with pytest.raises(AudioFileError, match="Cannot read"):
        FileAudioSource(tmp_path / "missing.wav").start()
    (tmp_path / "junk.wav").write_bytes(b"not a wav")
    with pytest.raises(AudioFileError, match="Cannot read"):
        FileAudioSource(tmp_path / "junk.wav").start()
