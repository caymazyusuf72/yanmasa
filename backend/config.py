"""Ayarlar — `.env` dosyasından, ortamdan yedekle.

Anahtarlar yalnızca burada okunur. Başka hiçbir modül `os.environ`'a
dokunmuyor ki bir anahtarın nereden geldiği tek yerden görülebilsin.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

MODEL = "claude-opus-5"
MAX_TOKENS = 32000
EFFORT = "high"


@dataclass
class Config:
    anthropic_api_key: str = ""
    llm_provider: str = "anthropic"  # "anthropic" | "openai"
    anthropic_model: str = "claude-opus-5"
    openai_api_key: str = ""
    openai_base_url: str = ""
    openai_model: str = "gpt-4o"
    elevenlabs_keys: list[str] = field(default_factory=list)
    elevenlabs_voice_id: str = ""
    elevenlabs_model: str = "eleven_flash_v2_5"
    tts_backend: str = "elevenlabs"
    stt_model: str = "small"
    stt_language: str = "tr"
    #: Uzak makine panelinin ön dolduracağı varsayılanlar. Depoda gerçek
    #: bir adres durmasın diye: bir IP, kullanıcı adı ve SSH portu tek
    #: başına parola değil ama "şu adreste root, şu portta" demek, kaba
    #: kuvvet denemesi için hazır bir hedef listesi vermek demek.
    ssh_alias: str = ""
    ssh_host: str = ""
    ssh_user: str = "root"
    ssh_port: int = 22

    @property
    def active_model(self) -> str:
        """Kullanılan model adı."""
        if self.llm_provider == "openai":
            return self.openai_model
        return self.anthropic_model

    @classmethod
    def load(cls) -> Config:
        load_dotenv(REPO_ROOT / ".env")

        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
        openai_base_url = os.environ.get("OPENAI_BASE_URL", "").strip()

        provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
        if not provider:
            if anthropic_key:
                provider = "anthropic"
            elif openai_key or openai_base_url:
                provider = "openai"
            else:
                provider = "anthropic"

        generic_model = os.environ.get("MODEL", "").strip()
        anthropic_model = generic_model or os.environ.get(
            "ANTHROPIC_MODEL", "claude-opus-5"
        ).strip()
        openai_model = generic_model or os.environ.get(
            "OPENAI_MODEL", "gpt-4o"
        ).strip()

        if provider == "anthropic":
            if not anthropic_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is missing. Copy `.env.example` to `.env` and "
                    "fill it in, or set OPENAI_API_KEY / OPENAI_BASE_URL to use OpenAI."
                )
        elif provider == "openai":
            if not openai_key and not openai_base_url:
                raise RuntimeError(
                    "OPENAI_API_KEY or OPENAI_BASE_URL is missing. Set OPENAI_API_KEY "
                    "or OPENAI_BASE_URL in `.env`."
                )
            if not openai_key and openai_base_url:
                # Yerel sunucularda (Ollama, LM Studio vb.) anahtar şart değil
                openai_key = "dummy-key"
        else:
            raise RuntimeError(
                f"Unknown LLM_PROVIDER: {provider!r}. Must be 'anthropic' or 'openai'."
            )

        raw_keys = os.environ.get("ELEVENLABS_KEYS", "")
        return cls(
            anthropic_api_key=anthropic_key,
            llm_provider=provider,
            anthropic_model=anthropic_model,
            openai_api_key=openai_key,
            openai_base_url=openai_base_url,
            openai_model=openai_model,
            elevenlabs_keys=[k.strip() for k in raw_keys.split(",") if k.strip()],
            elevenlabs_voice_id=os.environ.get("ELEVENLABS_VOICE_ID", "").strip(),
            elevenlabs_model=os.environ.get("ELEVENLABS_MODEL", "eleven_flash_v2_5"),
            tts_backend=os.environ.get("TTS_BACKEND", "elevenlabs"),
            stt_model=os.environ.get("STT_MODEL", "small"),
            stt_language=os.environ.get("STT_LANGUAGE", "tr"),
            ssh_alias=os.environ.get("SSH_ALIAS", "").strip(),
            ssh_host=os.environ.get("SSH_HOST", "").strip(),
            ssh_user=os.environ.get("SSH_USER", "root").strip(),
            ssh_port=int(os.environ.get("SSH_PORT", "22") or 22),
        )
