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

```bash
python review.py --base master --head feature/test --fake-llm
python review.py --base HEAD~1 --head HEAD --dry-run
python review.py --base master --head feature/test --output review-result.json --fake-llm
```

## Hva som er implementert i fase 1-2

- CLI med argparse
- Git-diff lesing med grenser (maks filer og diff-linjer)
- Endrede filer og endrede linjer
- Reviewer-profiler (bug/reliability/security)
- FakeLLMClient uten API-kall
- LLM-klient bak et lite Protocol-interface
- JSON-output med metadata, accepted og rejected findings

## Ikke implementert ennå

- Ekte OpenAI-klient
- Avansert guard-vask og deduplisering (fase 3+)
- GitHub-integrasjon

## Python-versjon i prosjektet

Prosjektet er konfigurert med:

`requires-python = ">=3.14,<3.15"`
