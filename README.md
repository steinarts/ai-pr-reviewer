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
       --review-mode consolidated \
       --llm-timeout 180 \
       --max-prompt-tokens 3500 \
       --test-max-prompt-tokens 2500 \
       --max-review-seconds 900 \
       --llm-max-output-tokens 700 \
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
--review-mode MODE     : consolidated|separate (default: consolidated)
--llm-timeout SEC      : Per-request hard wall-clock timeout (default: 180.0)
--max-review-seconds N : Total review wall-clock budget (default: 900)
--max-prompt-tokens N  : Approximate max prompt tokens per request (default: 3500)
--test-max-prompt-tokens N : Approximate max prompt tokens for test chunks (default: 2500)
--llm-max-output-tokens N : Approximate max output tokens per request (default: 700)
--max-findings-per-chunk N : Max findings accepted per request (default: 3)
--dry-run              : Use fake LLM (same as --provider fake)
--output FILE          : JSON output file (default: review-result.json)
--max-files N          : Max files to review (default: 30)
--max-diff-lines N     : Max diff lines (default: 3000)
--max-published N      : Max findings to publish (default: 3)
--min-confidence CONF  : Min confidence (0-1, default: 0.85)
--severities LEVELS    : Severity levels to accept (default: high critical)
```

## Chunked Reviews and Partial Results

- Reviews are chunked by approximate token budget (`--max-prompt-tokens`) per request.
- Default strategy is consolidated mode: one LLM request per chunk covering bug/reliability/security.
- Separate mode is available with `--review-mode separate`.
- Non-reviewable chunks (documentation/config/generated/comment-only) are skipped before LLM calls.
- Findings from all successful chunks are merged before deduplication and guards.
- If some chunks fail (for example timeout), the run continues and still writes JSON.
- Total review scheduling stops when `--max-review-seconds` is exhausted.
- Exit code is `0` for partial success, and `2` only when all LLM requests fail.

Metadata in JSON now includes:

- `chunk_count`
- `completed_requests`
- `failed_requests`
- `planned_requests`
- `skipped_requests`
- `reviewable_chunks`
- `skipped_chunks`
- `reviewer_failures[]` with reviewer/chunk/error details
- `reviewer_skips[]` with skip reasons (e.g. `no_reviewable_code`, `total_time_budget_exceeded`)

## Troubleshooting Slow Local Models

If `qwen2.5-coder:7b` is slow or times out:

1. Keep hard timeout conservative: `--llm-timeout 180`
2. Keep prompt budget conservative: `--max-prompt-tokens 3500`
3. Use debug telemetry:

```bash
python review.py --base main --head HEAD \
   --provider ollama \
   --model qwen2.5-coder:7b \
   --review-mode consolidated \
   --llm-timeout 180 \
   --max-review-seconds 900 \
   --max-prompt-tokens 3500 \
   --test-max-prompt-tokens 2500 \
   --llm-max-output-tokens 700 \
   --llm-debug \
   --llm-debug-log llm-debug.jsonl \
   --output review-result.json
```

Timeout enforcement:

- Each request is executed behind an application-level hard wall-clock timeout.
- On timeout, the worker process is terminated and control returns immediately.
- If installed `ollama` Python package cannot expose timeout support, startup fails fast with an actionable error.
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
