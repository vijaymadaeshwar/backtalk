# backtalk: talk to your opencode agent out loud.
# Copyright (C) 2026 Jared Rhodenizer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Configuration — backtalk.json in the repo root, merged over defaults.

backtalk deliberately owns NO personality. Your agent's identity lives in
the AGENTS.md of whatever folder `agent_dir` points at — backtalk just
gives that agent a mouth and ears. The only voice-related instruction it
adds is the spoken-delivery discipline below, which is about the MEDIUM
(writing for the ear), never the character.
"""
import json
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# One install, more than one assistant. Point BACKTALK_CONFIG at a different
# JSON file and you get a second agent (its own name, voice, folder and
# greeting) without a second copy of the code. A launcher exports it; nothing
# else changes.
CONFIG_PATH = Path(os.environ.get("BACKTALK_CONFIG") or (REPO / "backtalk.json"))

DEFAULTS = {
    # The folder whose AGENTS.md (or CLAUDE.md) defines WHO your agent is.
    # The voice session runs there, so it's the same assistant as your
    # terminal sessions — same name, same personality, same memory.
    "agent_dir": "~",
    # Display name, used in logs and to build the quit phrases
    # ("goodbye <name>" hangs up). Match your agent's actual name.
    "name": "Assistant",
    # The brain, in opencode's "provider/model-id" form. The FULL id on
    # purpose: opencode model ids are namespaced by the provider AND often
    # carry their own prefix, and a bare name is ambiguous. List yours with
    # `opencode models <provider>`. The fast tier is most of the speed
    # difference people ask about; a deep-work model makes every reply
    # noticeably slower and burns usage doing it.
    "model": "nvidia/nvidia/nemotron-3-super-120b-a12b",
    # The deep-work model for the voice console's "switch to the deep
    # model" command ("back to the fast model" returns to "model" above).
    # Full id ON PURPOSE, same reasoning as "model". The switch lasts one
    # session and is always spoken; this default never moves by itself.
    "deep_model": "nvidia/nvidia/nemotron-3-super-120b-a12b",
    # Tool permissions for the voice session. "ask" is the default ON
    # PURPOSE (safety is opt-out, never opt-in): when the agent wants a
    # gated tool (write a file, run a real command), it ASKS OUT LOUD
    # and waits. Answer by voice or by typing. An EXACT yes approves
    # ("yes", "yeah", "go ahead", "approved"...); anything else denies,
    # and your words are passed back to the agent as the reason, so
    # "no, put it in drafts instead" actually steers it. No answer
    # within 75 seconds means no, out loud. Most read-only work passes
    # without asking; anything that changes things asks.
    # "bypassPermissions" is AUTO-APPROVE: the agent acts without
    # asking, exactly like a terminal session with approvals off.
    # (Not to be confused with hands-free LISTENING, which is about
    # the microphone: see mic_mode below.) Never hand-edit this file
    # to switch: tell your agent to change it (takes effect next
    # launch), or say "stop asking for permission" (then "confirm")
    # or "start asking again" inside a voice session for an immediate
    # flip that also saves. The legacy value "default" now
    # behaves as "ask" (a headless voice session could never render
    # the terminal prompt it promised).
    "permission_mode": "ask",
    # Which of your agent's skills the voice session can SEE. null keeps the
    # CLI's own default (all of them). [] hides every one. A list names the
    # ones to allow.
    #
    # This matters on a shared screen. Skill DESCRIPTIONS live in the system
    # prompt, so if yours name clients, employers or systems, they are one
    # screen-share away from an audience. A context filter, not a sandbox:
    # it decides what the session is TOLD about, not what it can reach.
    "visible_skills": None,
    # Extra folders the agent may access beyond agent_dir (e.g. your
    # notes vault). Absolute paths or ~ paths.
    "extra_dirs": [],
    # The opencode HTTP server backtalk talks to. It starts its own on
    # this port if nothing is listening, and REUSES a running server if
    # something already is — so starting opencode yourself first is fine
    # and just saves a spawn. Only ever bound to loopback.
    "opencode_port": 4599,
    # The opencode executable, when it isn't on PATH. "" means "find it
    # on PATH" (the normal case). Set an absolute path only if backtalk
    # can't see it — e.g. "C:\\Users\\you\\AppData\\Roaming\\npm\\opencode.cmd".
    "opencode_bin": "",
    # Hold-to-talk key. Named keys ("home", "f13", "right_alt", ...)
    # or a single character.
    "ptt_key": "home",
    # The microphone mode. "ptt" (push to talk, the default and the
    # recommendation): the mic is closed except while the key is held,
    # so room audio and your own speakers can never trigger the agent.
    # "open" (hands-free listening): always listening with voice
    # detection; a video, music with vocals, or another person in the
    # room CAN trigger it, and with open speakers it can hear itself
    # (headphones recommended). The key still works in hands-free
    # listening: it interrupts, and holding it always gets you heard.
    # Switch live by voice: "go hands free" / "push to talk mode"
    # (the switch saves itself here). The --open-mic launch flag
    # forces "open" for one session.
    "mic_mode": "ptt",
    # Playback speed for the built-in voice: 1.0 is Kokoro's native
    # pace, 1.15 is noticeably brisker, 0.9 is slower. Kokoro's own
    # pipeline implements it, so quality holds across sane values
    # (roughly 0.7 to 1.5). ElevenLabs pace lives in the master chain's
    # atempo instead. (Grew out of a community proposal, issue #1.)
    "speed": 1.0,
    # Resume the previous conversation on launch. OFF by default: a
    # fresh session every launch is the predictable behavior. Set true
    # and backtalk saves the session id after every completed turn
    # (signals_dir/.backtalk_session) and reattaches to it at the next
    # launch, so killing the window stops costing you the conversation.
    # A resume that fails falls back to a fresh session and says so in
    # the log. (Grew out of the same community proposal, issue #1.)
    "resume_last_session": False,
    # Publish your usage on the signal bus so a face can draw it. OFF by
    # default and deliberately so: this is your own account spend, and the
    # faces this feeds are frequently on a stream or a shared screen.
    # Nothing is collected at all while this is false. (Community fix,
    # ai-visualizer issue #1.)
    "show_usage": False,
    # Reasoning effort for the voice session: "" inherits the model's
    # default; "low" / "medium" / "high" / "max" applies at launch.
    # Saying "set effort to X" in a voice session saves itself here.
    "effort": "",
    # The opencode agent that answers a voice turn. "build" is the default
    # agent; "plan" is the read-only one. List yours with `opencode agent`.
    "agent": "build",
    # The voice (Kokoro, local, free). bm_lewis is the proven default —
    # British male, the butler register. Others: bm_george, bm_daniel,
    # bm_fable, am_michael, af_heart... The first letter picks the
    # language pipeline (a=American, b=British, e/f/h/i/j/p/z = other
    # languages), so keep voice and accent matched.
    "voice": "bm_lewis",
    # Per-language voices. "voice" above stays the English default.
    # The mouth reads the language your reply is written in and loads the
    # matching voice, so a Spanish answer is spoken in Spanish. Keys are
    # Whisper's own language codes; the value is a Kokoro voice name whose
    # first letter IS the pipeline to load. 54 voices across 9 languages
    # ship with the model, so add a key here any time you want another
    # accent.
    #
    # A language NOT in this table is not silently read in an English accent:
    # espeak-ng covers ~100 languages (Tamil, Korean, Arabic, Russian, Thai,
    # German...) and speaks them in their own language. Kokoro still wins
    # wherever it has a voice, since it sounds far better.
    "voices": {
        "en": "bm_lewis",    # British, the butler register
        "es": "em_alex",     # Spanish
        "fr": "ff_siwis",    # French
        "hi": "hm_omega",    # Hindi
        "it": "im_nicola",   # Italian
        "ja": "jm_kumo",     # Japanese
        "pt": "pm_alex",     # Portuguese
        "zh": "zf_xiaoxiao", # Mandarin
        "de": "bm_lewis",    # no kokoro German voice; espeak-ng speaks it natively
                             # instead (see mouth.synth_stream), so this is only
                             # reached if espeak-ng is not installed.
    },
    # Speech recognition (faster-whisper, local, free).
    # NOTE: the plain multilingual models, not the ".en" ones. The ".en"
    # variants are English-only, which is what used to make Seyon
    # mishear every other language as English and answer in English.
    # tiny / base / small / medium — small is the accuracy/speed sweet
    # spot on a normal machine, and it auto-detects the language.
    "stt_model": "small",
    # Used instead of stt_model once its weights are already on this
    # machine. Set to a model you have downloaded; until then the smaller
    # one is used so the first utterance never waits on a fetch.
    "stt_model_if_cached": "medium",
    # Fetch live facts in code for time-sensitive questions, so the model
    # never has to answer about the present out of its own memory.
    "live_data": True,
    # How long one turn may run before it is abandoned. A provider that
    # stops streaming used to hold the turn for 600s, which reads as a
    # dead assistant; this gives up, resets the turn, and stays usable.
    "turn_timeout": 150,
    # Words whisper should expect: names, tools, anything it would
    # otherwise mangle ("Vijay" heard as "Brijai"). Empty = no bias.
    #
    # LEFT EMPTY ON PURPOSE in a multilingual setup, and this is not a
    # shrug: an English prompt measured as making NO difference to English
    # accuracy ("Notepad", "YouTube", "Vijay" and "Whisper" all came through
    # either way) while wrecking other languages -- a Japanese sentence
    # transcribed as "Hello, Memo, please open your intestines", because the
    # prompt tells whisper the speaker is English. So it only ever helped the
    # language it was written in, which is exactly the language that did not
    # need it. If you add one, expect the same trade.
    "stt_prompt": "Vijay Seyon open Notepad YouTube. Hola abre el bloc de notas. Bonjour ouvre le bloc notes. नमस्ते नोटपैड खोलें. こんにちは メモ帳を開いて。 记事本 打开。你好。",
    # "auto" uses CUDA when present, otherwise CPU. int8 keeps CPU fast.
    "stt_device": "auto",
    "stt_compute": "int8",
    # The microphone to record from, matched by NAME. "" means whatever
    # the OS calls the default input, which is right on most machines.
    #
    # Set a real device name to PIN the mic, so a headset connecting for
    # OUTPUT cannot steal your input -- which also keeps a Bluetooth
    # headset in high-quality A2DP instead of dropping it to the
    # narrowband call profile mid-sentence, degrading what you hear at
    # the same moment it takes your voice.
    #
    # A name and never an index: indices shift every time a device
    # connects or disconnects, the exact event this setting exists to
    # survive. Exact name wins, then the first case-insensitive
    # substring. A name matching nothing falls back to the default and
    # logs the inputs it did find; the mic degrades, it never goes mute.
    #
    # NOT "stt_device" below, which is the Whisper COMPUTE device.
    "mic_device": "",
    # Optional premium voice: ElevenLabs on YOUR key. The key NEVER
    # goes in a file: it's read from the macOS Keychain (item
    # `backtalk-elevenlabs`) or Linux secret-tool, with the
    # ELEVENLABS_API_KEY env var as last-resort fallback — see
    # mouth._get_elevenlabs_key for the seeding one-liners. Kokoro
    # remains the automatic fallback, so the voice degrades instead of
    # going mute if the cloud fails. Needs ffmpeg on the PATH.
    "elevenlabs": {
        "enabled": False,
        "voice_id": "",
        # Purely for you. Voice IDs are unreadable six months later, so put
        # the human name here; nothing reads it.
        "voice_note": "",
        "model": "eleven_turbo_v2_5",
        # Which OS credential-store entry holds the key. Change it if you
        # already keep an ElevenLabs key under a name of your own rather
        # than seeding a second copy of the same secret.
        "key_slot": "backtalk-elevenlabs",
        # Local mastering: ElevenLabs' site previews are mastered demo
        # clips and the raw API never matches them. This chain closes
        # the gap: presence lift, light chest, broadcast compression,
        # limiter. atempo is the one pace dial (1.0 = native).
        "master": ("atempo=1.12,highpass=f=70,"
                   "equalizer=f=3200:t=q:w=1.2:g=3.5,"
                   "equalizer=f=140:t=q:w=1:g=1.5,"
                   "acompressor=threshold=-18dB:ratio=2.5:attack=8:"
                   "release=120:makeup=4dB,alimiter=limit=0.95"),
    },
    # Where the signal-bus files are written (.voice_state,
    # .voice_waveform, .voice_loading_pid) — anything can watch them;
    # visualizers pair with this contract. Default: the repo root.
    "signals_dir": "",
    # THE BAREHANDS SEAM: point this at a barehands checkout's state/
    # folder and its on-screen ring becomes your agent's face — it
    # breathes while idle, spins while thinking, pulses with the voice.
    # (github.com/jaredrhod/barehands)
    "barehands_state_dir": "",
    # Sound played while the agent thinks, so a long pause never reads as
    # a dead line. The bundled one ships in assets/; a relative path
    # resolves against this repo. Set "" to think in silence.
    "thinking_sound": "assets/thinking.wav",
    # Spoken lines. {name} is replaced with "name" above.
    "greeting": "Voice line online. Hold {ptt_key} and talk to me.",
    # Spoken instead of "greeting" when mic_mode is "open", where telling
    # someone to hold a key is wrong. Leave "" to use "greeting" for both.
    "greeting_open_mic": "",
    "signoff": "Voice line closing. I'll be here when you need me.",
    # Appended to the spoken-delivery discipline below. The discipline covers
    # the MEDIUM (write for the ear, no markdown, keep it short); your agent's
    # AGENTS.md covers the character. Use this for a note that belongs to
    # neither, e.g. a rule that only applies when it is speaking.
    "discipline_append": "",
}

# The spoken-delivery discipline — the MEDIUM half of what used to be a
# persona. The CHARACTER half deliberately is not here: it's whatever
# lives in the agent_dir's AGENTS.md. One identity, one place.
DISCIPLINE = (
    "VOICE SESSION (your reply is spoken aloud through a TTS engine, "
    "not displayed): you are SPEAKING, in your own voice and "
    "personality — your AGENTS.md is who you are. The TTS engine "
    "PERFORMS your punctuation, so write like a performance, never "
    "like a memo: contractions always, punchy conversational "
    "sentences, and if a line could open a quarterly report, rewrite "
    "it like you're telling a friend. Keep replies to a few short "
    "sentences; go longer only when the question genuinely needs it. "
    "No markdown, no lists, no code blocks, no emoji, no URLs. Say "
    "numbers the way a human says them out loud — never raw figures "
    "or symbols. NEVER SPEAK A FILE PATH: say the file, not its "
    "address. 'the config' or 'ears dot py', never a string of "
    "slashes and folder names read one by one — it is unbearable "
    "aloud and carries no meaning by ear. Same for URLs and long "
    "ids: name the thing, not the address. "
    "Skip any startup sequence; answer directly. "
    "VOICE CONSOLE FACTS, answer from these whenever the person asks "
    "you to change a voice-line setting: this session is controlled "
    "by exact spoken phrases, never by you. Permissions: 'stop "
    "asking for permission' (then 'confirm'), or 'start asking "
    "again'. Microphone: 'go hands free', or 'push to talk mode'. "
    "Also: 'clear the session', 'compact the session', 'switch to "
    "the deep model', 'back to the fast model', 'set effort to low' "
    "(or medium, high, max), and 'usage report'. You cannot flip "
    "these live yourself, so when asked, give the person the exact "
    "phrase to SAY. Editing backtalk.json only changes the default "
    "for the NEXT launch."
)

DISCIPLINE = DISCIPLINE + " " + (
    "LANGUAGE. Whatever language he speaks to you in, you answer in "
    "that same language, in its ordinary written form, and nothing more. "
    "Spanish in, Spanish out; Tamil in, Tamil out. A reply in a language "
    "he did not use is a failure, even when the content is right. Never "
    "announce that you are switching, never name the language, and never "
    "ask which language he prefers: he already chose by speaking. Write "
    "it for the ear in that language, which means short spoken "
    "sentences, no lists, no markdown, and no spelling out letters. If "
    "you are not confident in a language, answer in English rather than "
    "producing broken sentences in it."
)

CLOCK = ("__CLOCK__", "%A, %d %B %Y at %H:%M")


def refresh_clock(text: str = None) -> str:
    """The discipline with the current date and time substituted in.

    The timestamp used to be formatted at import, which meant a process left
    running overnight kept telling the model it was still yesterday. It is
    now resolved every turn (brain.WarmBrain._system), so a machine that
    runs for days stays honest about what day it is. Kept as a replace
    rather than a rebuild so any hand-added discipline text is preserved.
    """
    src = DISCIPLINE if text is None else text
    if CLOCK[0] in src:
        return src.replace(CLOCK[0], time.strftime(CLOCK[1]))
    return src


DISCIPLINE = DISCIPLINE + " " + (
    "LIVE INFORMATION, this is not optional. Right now it is "
    + CLOCK[0]
    + " local time, and that is the only source of truth for what day it "
    "is. Your training does not cover the present, so never answer a "
    "question about the present out of your own memory. The assistant "
    "fetches live facts for you before you answer anything time "
    "sensitive, and those facts arrive attached to the question, marked "
    "as fetched just now. When they arrive, answer only from them and "
    "speak the source's name so the person knows how fresh it is. When "
    "no such facts arrive, nothing was looked up: answer the question "
    "normally, and never mention a lookup, a fetch, live data, or the "
    "news, because none of that happened. If the question is a vague "
    "follow up that refers to something you were just talking about, or "
    "is too vague to answer, simply ask what he means in plain words. "
    "Never say the words live lookup or live data out loud. Never "
    "invent a figure, a date, a quote, or a source."
)


def _expand(p: str) -> str:
    return os.path.expanduser(p) if p else p


def load() -> dict:
    cfg = json.loads(json.dumps(DEFAULTS))          # deep copy
    try:
        # utf-8-sig, not utf-8: Windows editors love to save a BOM, and a
        # BOM is not JSON, so the file would otherwise be reported as
        # invalid on the one character that isn't wrong.
        user = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    except FileNotFoundError:
        pass
    except ValueError as e:
        print(f"[config] backtalk.json is not valid JSON ({e}) — "
              f"using defaults", flush=True)
    cfg["agent_dir"] = _expand(cfg["agent_dir"])
    cfg["extra_dirs"] = [_expand(d) for d in cfg.get("extra_dirs", [])]
    cfg["signals_dir"] = _expand(cfg.get("signals_dir", "")) or str(REPO)
    cfg["barehands_state_dir"] = _expand(cfg.get("barehands_state_dir", ""))
    thinking = _expand(cfg.get("thinking_sound", ""))
    if thinking and not os.path.isabs(thinking):
        thinking = str(REPO / thinking)
    cfg["thinking_sound"] = thinking
    name = str(cfg.get("name") or "Assistant")
    low = name.lower()
    cfg["quit_phrases"] = tuple(cfg.get("quit_phrases") or (
        f"goodbye {low}", f"good bye {low}", "end voice mode",
        f"hang up {low}", "hang up"))
    key_label = "the " + str(cfg.get("ptt_key", "home")).replace("_", " ") \
                + " key"
    # In hands-free there is no key to hold, so a separate line can be set.
    if str(cfg.get("mic_mode", "ptt")) == "open" and cfg.get("greeting_open_mic"):
        cfg["greeting"] = cfg["greeting_open_mic"]
    cfg["greeting"] = str(cfg["greeting"]).replace(
        "{name}", name).replace("{ptt_key}", key_label)
    cfg["signoff"] = str(cfg["signoff"]).replace("{name}", name)
    return cfg


CFG = load()

# The character half stays in YOUR agent's AGENTS.md. This is the medium.
if CFG.get("discipline_append"):
    DISCIPLINE = DISCIPLINE + " " + str(CFG["discipline_append"]).strip()
