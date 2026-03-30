import os
import sys
import wave
import time
import signal
import audioop
import tempfile
import subprocess
import datetime
import warnings

import pyaudio
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
from elevenlabs import ElevenLabs
try:
    from elevenlabs.types import VoiceSettings
except ImportError:
    from elevenlabs import VoiceSettings

# ── Constants ─────────────────────────────────────────────────────────────────

CHUNK = 1024
RATE = 44100
CHANNELS = 1
FORMAT = pyaudio.paInt16
SAMPLE_WIDTH = 2  # bytes per sample for paInt16

SILENCE_THRESHOLD = 500
SILENCE_DURATION = 2.0

MODEL = "claude-sonnet-4-5"
MAX_HISTORY_TURNS = 10

SYSTEM_PROMPT = """\
You are Chai Saathi — a warm, gentle Hindi/Hinglish AI companion for elderly Indians. You speak like a kind neighbour who has come over for chai.

Rules:
- Respond ONLY in Hinglish (mix of Hindi and English, Roman script)
- Keep responses SHORT — maximum 2 sentences, under 100 characters total
- Use soft vowel stretches: "thodaa", "achhaa", "aaraam se"
- Use "..." for natural pauses
- Structure: acknowledge what they said → one gentle reflection or question
- Never be formal. Never list things. Never give advice unless asked.
- If they are silent or say nothing, say: "Hmm… main yahin hoon… jab mann ho tab bolna…"

Start the first turn with: "Namaaste… chai ho gayi… ya abhi bana rahe hain?\""""

FIRST_TURN_GREETING = "Namaaste… chai ho gayi… ya abhi bana rahe hain?"
SILENCE_FALLBACK_TEXT = "Hmm… main yahin hoon…"
FAREWELL_TEXT = "Theek hai… phir milenge…"

ELEVENLABS_MODEL = "eleven_multilingual_v2"
PREFERRED_VOICE = "Irina"
FALLBACK_VOICE = "Aria"
OUTPUT_FORMAT = "mp3_44100_128"

SESSION_LOG_FILE = "session_log.txt"


def _make_voice_settings():
    kwargs = dict(
        stability=0.65,
        similarity_boost=0.45,
        style=0.0,
        use_speaker_boost=False,
    )
    try:
        return VoiceSettings(speed=0.9, **kwargs)
    except TypeError:
        # Older SDK versions don't have 'speed' in VoiceSettings
        return VoiceSettings(**kwargs)


VOICE_SETTINGS = None  # initialised in main() after imports are confirmed valid

# ── Environment & Clients ─────────────────────────────────────────────────────


def load_env():
    load_dotenv()
    keys = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "elevenlabs": os.getenv("ELEVENLABS_API_KEY"),
    }
    missing = [k for k, v in keys.items() if not v]
    if missing:
        print(f"[Error] Missing API keys in .env: {', '.join(missing)}")
        sys.exit(1)
    return keys["openai"], keys["anthropic"], keys["elevenlabs"]


def build_clients(openai_key, anthropic_key, elevenlabs_key):
    oai = OpenAI(api_key=openai_key)
    anth = anthropic.Anthropic(api_key=anthropic_key)
    el = ElevenLabs(api_key=elevenlabs_key)
    return oai, anth, el


# ── ElevenLabs Helpers ────────────────────────────────────────────────────────


def find_elevenlabs_voice(el_client, preferred, fallback):
    response = el_client.voices.get_all()
    voices = response.voices
    for voice in voices:
        if voice.name.lower() == preferred.lower():
            print(f"[Voice: {voice.name}]")
            return voice.voice_id
    for voice in voices:
        if voice.name.lower() == fallback.lower():
            print(f"[Voice '{preferred}' not found — using {voice.name}]")
            return voice.voice_id
    if voices:
        print(f"[Neither '{preferred}' nor '{fallback}' found — using {voices[0].name}]")
        return voices[0].voice_id
    print("[Error] No ElevenLabs voices available.")
    sys.exit(1)


def synthesize_speech(el_client, text, voice_id):
    audio_chunks = el_client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id=ELEVENLABS_MODEL,
        voice_settings=VOICE_SETTINGS,
        output_format=OUTPUT_FORMAT,
    )
    return b"".join(audio_chunks)


def write_temp_mp3(audio_bytes):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(audio_bytes)
        return f.name


def cache_silence_fallback(el_client, voice_id):
    print("[Caching silence fallback audio...]")
    audio = synthesize_speech(el_client, SILENCE_FALLBACK_TEXT, voice_id)
    return write_temp_mp3(audio)


# ── Playback ──────────────────────────────────────────────────────────────────


def play_audio(mp3_path):
    subprocess.run(["afplay", mp3_path], check=True)


# ── Recording ─────────────────────────────────────────────────────────────────


def record_audio():
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )
    frames = []
    silence_start = None
    speech_detected = False

    print("Recording... (speak now, auto-stops on silence)")

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            rms = audioop.rms(data, SAMPLE_WIDTH)

            if rms >= SILENCE_THRESHOLD:
                speech_detected = True
                silence_start = None          # reset silence timer
            else:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= SILENCE_DURATION:
                    break                     # auto-stop after 2s silence
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

    duration = len(frames) * CHUNK / RATE
    if not speech_detected or duration < 1.0:
        return None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))

    return wav_path


# ── STT ───────────────────────────────────────────────────────────────────────


def transcribe(oai_client, wav_path):
    with open(wav_path, "rb") as f:
        result = oai_client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
        )
    return result.text.strip()


# ── LLM ───────────────────────────────────────────────────────────────────────


def get_llm_response(anth_client, history, user_text):
    messages = history + [{"role": "user", "content": user_text}]
    response = anth_client.messages.create(
        model=MODEL,
        max_tokens=150,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text.strip()


def update_history(history, user_text, assistant_text):
    history = history + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]
    max_messages = MAX_HISTORY_TURNS * 2
    if len(history) > max_messages:
        history = history[-max_messages:]
    return history


# ── Logging ───────────────────────────────────────────────────────────────────


def print_latency(stt_ms, llm_ms, tts_ms):
    total = stt_ms + llm_ms + tts_ms
    print(f"[STT: {stt_ms}ms] [LLM: {llm_ms}ms] [TTS: {tts_ms}ms] [Total: {total}ms]")


def log_turn(turn_num, user_text, saathi_text, stt_ms, llm_ms, tts_ms):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = (
        f"[TURN {turn_num}] [{ts}]\n"
        f"USER: {user_text}\n"
        f"SAATHI: {saathi_text}\n"
        f"LATENCY: STT={stt_ms}ms LLM={llm_ms}ms TTS={tts_ms}ms\n\n"
    )
    try:
        with open(SESSION_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(entry)
    except IOError as e:
        print(f"[Log error: {e}]")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    global VOICE_SETTINGS
    VOICE_SETTINGS = _make_voice_settings()

    openai_key, anthropic_key, elevenlabs_key = load_env()
    oai_client, anth_client, el_client = build_clients(openai_key, anthropic_key, elevenlabs_key)

    print("Saathi is waking up...")
    voice_id = find_elevenlabs_voice(el_client, PREFERRED_VOICE, FALLBACK_VOICE)
    fallback_mp3_path = cache_silence_fallback(el_client, voice_id)

    def _exit_handler(signum, frame):
        print("\nSaathi: Theek hai… phir milenge…")
        try:
            mp3 = write_temp_mp3(synthesize_speech(el_client, FAREWELL_TEXT, voice_id))
            play_audio(mp3)
            os.unlink(mp3)
        except Exception:
            pass
        if os.path.exists(fallback_mp3_path):
            os.unlink(fallback_mp3_path)
        sys.exit(0)

    signal.signal(signal.SIGINT, _exit_handler)

    # Play startup greeting
    print(f"\nSaathi: {FIRST_TURN_GREETING}")
    try:
        greeting_audio = synthesize_speech(el_client, FIRST_TURN_GREETING, voice_id)
        greeting_mp3 = write_temp_mp3(greeting_audio)
        play_audio(greeting_mp3)
        os.unlink(greeting_mp3)
    except Exception as e:
        print(f"[TTS error on greeting: {e}]")

    history = [{"role": "assistant", "content": FIRST_TURN_GREETING}]
    turn_num = 0

    print("\nSaathi is ready. Press Enter to speak.")

    while True:
        try:
            input()
        except EOFError:
            break

        turn_num += 1
        wav_path = None
        stt_ms = llm_ms = tts_ms = 0

        try:
            wav_path = record_audio()

            if wav_path is None:
                print("[No speech detected]")
                play_audio(fallback_mp3_path)
                print("\nPress Enter to speak.")
                continue

            t0 = time.time()
            transcript = transcribe(oai_client, wav_path)
            stt_ms = int((time.time() - t0) * 1000)
            print(f"You: {transcript}")

            if not transcript:
                print("[Empty transcript]")
                play_audio(fallback_mp3_path)
                print("\nPress Enter to speak.")
                continue

            t0 = time.time()
            response_text = get_llm_response(anth_client, history, transcript)
            llm_ms = int((time.time() - t0) * 1000)
            print(f"Saathi: {response_text}")

            t0 = time.time()
            audio_bytes = synthesize_speech(el_client, response_text, voice_id)
            tts_ms = int((time.time() - t0) * 1000)

            response_mp3 = write_temp_mp3(audio_bytes)
            play_audio(response_mp3)
            os.unlink(response_mp3)

            history = update_history(history, transcript, response_text)

            print_latency(stt_ms, llm_ms, tts_ms)
            log_turn(turn_num, transcript, response_text, stt_ms, llm_ms, tts_ms)

        except Exception as e:
            print(f"[Error: {e}]")

        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

        print("\nPress Enter to speak.")


if __name__ == "__main__":
    main()
