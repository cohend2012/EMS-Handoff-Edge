"""
audio — everything that touches the microphone.

    capture.py    one mic stream -> 16 kHz blocks -> subscribers + full recording
    listeners.py  live captions (Vosk, or a Whisper fallback) and the stop-phrase spotter
    session.py    one recording from START to DONE / Cancel, with its threads
    sources.py    other audio sources: a WAV clip played at mic pace (replay)
"""
