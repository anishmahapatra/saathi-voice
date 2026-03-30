# Saathi Voice Loop — Project Context

## What This Is
Phase 0 validation of a voice-first AI companion for lonely elderly Indians. Goal: prove the core interaction feels human before building an app. This is not a product yet — it is a truth test.

## Core Emotional Principle
"Continuation, not companionship" — the app's job is to be the next thing that appears in someone's day when next things have stopped coming. Not a chatbot. A daily ritual.

## Persona: Chai Saathi
Warm, Hinglish, short responses. Speaks like a kind neighbour. Never clinical, never formal, never lists things. Soft vowel stretches. Natural pauses with "...". Acknowledges before it reflects.

## Product Name
Working name: Saathi. Final brand name TBD — Vayu and Praan are under consideration.

## Stack
- STT: OpenAI Whisper (whisper-1)
- LLM: Anthropic Claude API (claude-sonnet-4-5)
- TTS: ElevenLabs (target: Irina, fallback: Aria)
- Runtime: Python on Mac (M2)
- Future app: React Native (Expo), Android first

## Architecture Notes
- Silence detection replaces keypress-to-stop (zsh Ctrl+O conflict on Mac)
- Common responses (silence fallback) pre-generated and cached at startup
- Session logs written per turn for qualitative review

## Key Constraints
- NEVER use the word "AI" in any user-facing string
- Responses must be under 100 characters
- Total latency target: under 3 seconds end-to-end
- Silence threshold: 500 rms, 2 second duration

## Files
- saathi_loop.py — main voice loop (single file)
- session_log.txt — conversation logs (gitignored)
- .env — API keys (gitignored, never commit)
- .env.example — placeholder keys (committed)
- requirements.txt — all dependencies
- CLAUDE.md — this file

## What To Never Break
- Silence detection fallback (pre-cached audio)
- Session logging (every turn must be written)
- Graceful Ctrl+C exit with farewell line
- Latency logging (STT / LLM / TTS split)

## Phase Status
Phase 0 — voice loop only. No app, no database, no auth, no payments.
Next phase: React Native Expo Android app once interaction quality is validated.

## Metrics That Matter Right Now
- End-to-end latency (target: <3000ms total)
- Does it feel human after 5 minutes of talking?
- Does the user want to come back tomorrow?
