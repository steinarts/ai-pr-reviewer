# ai-pr-reviewer

Lokalt AI-basert code review-verktøy (fase 1 og 2 skjelett).

## Krav

- Python 3.14.x
- Git tilgjengelig i PATH

## Installasjon

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e .[dev]
```

## Kjøring

### Fake LLM (standard, ingen avhengigheter)

For testing og utvikling uten å kreve en LLM-server:

```bash
python review.py --base main --head <feature-branch> --dry-run
python review.py --base HEAD~1 --head HEAD --dry-run
python review.py --base main --head HEAD --output review-result.json --dry-run
```

### Ollama (lokal LLM)

Hvis du har Ollama installert lokalt:

1. **Installer Ollama**

   Last ned fra https://ollama.ai og installer.

2. **Hent en modell**

   ```bash
   ollama pull qwen2.5-coder:7b
   ```

   (Andre gode modeller: `mistral`, `neural-chat`, etc.)

3. **Start Ollama-serveren**

   ```bash
   ollama serve
   ```

   Standard host er `http://localhost:11434`.

4. **Kjør ai-pr-reviewer med Ollama**

   ```bash
   python review.py --base main --head HEAD \
     --provider ollama \
     --model qwen2.5-coder:7b \
     --output review-result.json
   ```

   Eller med custom Ollama-host:

   ```bash
   python review.py --base main --head HEAD \
     --provider ollama \
     --model qwen2.5-coder:7b \
     --ollama-host http://192.168.1.100:11434 \
     --output review-result.json
   ```

## CLI-argumenter

```
--base COMMIT          : Base commit (required)
--head COMMIT          : Head commit (required)
--provider {fake,ollama} : LLM provider (default: fake)
--model MODEL          : Model name (required for ollama)
--ollama-host HOST     : Ollama server host (default: http://localhost:11434)
--dry-run              : Use fake LLM (same as --provider fake)
--output FILE          : JSON output file (default: review-result.json)
--max-files N          : Max files to review (default: 30)
--max-diff-lines N     : Max diff lines (default: 3000)
--max-published N      : Max findings to publish (default: 3)
--min-confidence CONF  : Min confidence (0-1, default: 0.85)
--severities LEVELS    : Severity levels to accept (default: high critical)
```

## Hva som er implementert i fase 1-2

- CLI med argparse
- Git-diff lesing med grenser (maks filer og diff-linjer)
- Endrede filer og endrede linjer
- Reviewer-profiler (bug/reliability/security)
- FakeLLMClient uten API-kall
- **OllamaLLMClient for lokal LLM-kjøring**
- LLM-klient-fabrikk med provider-valg
- Strukturert system/user-prompt-grensesnitt
- JSON-output med metadata, accepted og rejected findings
- Feilhåndtering for Ollama-forbindelse og manglende modeller

## Ikke implementert ennå

- Ekte OpenAI-klient
- Avansert guard-vask og deduplisering (fase 3+)
- GitHub-integrasjon
- Streaming-output
- Automatisk modellnedlasting
- Plugin-system

## Python-versjon i prosjektet

Prosjektet er konfigurert med:

`requires-python = ">=3.14,<3.15"`
