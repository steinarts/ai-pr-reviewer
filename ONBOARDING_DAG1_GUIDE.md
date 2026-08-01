# Onboarding og Dag 1 i ai-pr-reviewer

Denne guiden samler to ting:
- hva onboarding betyr i praksis i dette repoet
- hvordan en konkret dag 1 kan gjennomfores trygt og effektivt

## Kort forskjell

- Onboarding: forsta hvordan systemet fungerer.
- Dag 1: levere en liten, trygg endring med testdekning.

En enkel huskeregel:
- Onboarding svarer pa: "Hvordan fungerer dette?"
- Dag 1 svarer pa: "Kan jeg levere en trygg endring her?"

## Del 1: Onboarding

### Maling

Etter onboarding skal du kunne:
1. Forklare hele flyten fra CLI til publisert resultat.
2. Peke ut hvilke moduler som styrer diff, prompts, verifier og guard.
3. Lese benchmark-output og skille candidate, verified og published metrics.

### Forslaatt lese-rekkefolge

Les i denne rekkefolgen for raskest mulig helhetsforstaelse:
1. README.md
2. review.py
3. reviewer/cli.py
4. reviewer/models.py
5. reviewer/git_diff.py
6. reviewer/context_builder.py
7. reviewer/scouts.py
8. reviewer/deduplicator.py
9. reviewer/verifier.py
10. reviewer/guard.py
11. reviewer/output.py
12. benchmarks/run_benchmarks.py
13. benchmarks/schema.py
14. benchmarks/scoring.py
15. tests/test_verifier.py
16. tests/test_benchmarks.py

### Kjernespor du bor kunne forklare

1. Review-spor:
- parse args
- collect diff
- build context
- run reviewers
- deduplicate findings
- optional verifier
- guard
- write output

2. Verifier-spor:
- candidate finding inn
- valid/invalid/uncertain + confidence
- contradiction_code + evidenslinjer
- verification_rejected_findings vs verified_findings

3. Benchmark-spor:
- candidate-findings-input replay
- before/after metrics
- per-finding effects
- replay expected vs actual verifier verdict

### Praktisk onboarding-sjekkliste

Kjor disse kommandoene lokalt for baseline:

```bash
. .venv/Scripts/activate
pytest -q
python benchmarks/run_benchmarks.py --provider fake --case clean_fake_llm_client --output benchmarks/results/onboarding-smoke.json
```

Noter:
1. Antall passerte tester.
2. Hvor i JSON du finner candidate_findings, verified_findings og accepted_findings.
3. Hvilke metadata-felter som viser verifier-status.

## Del 2: Konkret Dag 1-plan

Dette er en detaljert plan for en normal arbeidsdag (ca. 7-8 timer).

### 09:00-09:30: Miljo og baseline

Maal:
- Bekrefte at alt er kjoreklart lokalt.

Aktiviteter:
1. Aktiver virtualenv.
2. Kjor full test-suite.
3. Kjor en rask benchmark-smoke.
4. Skriv ned baseline (teststatus + varighet).

Forslag til kommandoer:

```bash
. .venv/Scripts/activate
pytest -q
python benchmarks/run_benchmarks.py --provider fake --case clean_fake_llm_client --output benchmarks/results/day1-smoke.json
```

### 09:30-10:30: Flytforstaelse i runtime

Maal:
- Forsta hva som skjer i praksis nar review.py kjores.

Aktiviteter:
1. Les review.py og reviewer/cli.py.
2. Marker i notater hvor hvert delsteg i pipelinen trigges.
3. Finn hvilke CLI-flagg som paavirker review vs verifier.

### 10:30-11:30: Datamodell og kontrakter

Maal:
- Forsta hvilke felter som flyter gjennom systemet.

Aktiviteter:
1. Les reviewer/models.py.
2. Kartlegg Finding-felter og verification-felter.
3. Kartlegg ReviewResult-strukturen.

Sjekkpunkt:
- Du skal kunne forklare forskjellen pa:
  - candidate_findings
  - verified_findings
  - verification_rejected_findings
  - accepted_findings

### 11:30-12:15: Diff og context

Maal:
- Forsta hvordan input-data til LLM bygges.

Aktiviteter:
1. Les reviewer/git_diff.py.
2. Les reviewer/context_builder.py.
3. Verifiser hvordan changed lines og snippets beregnes.

### 12:15-13:00: Lunsj

### 13:00-14:30: Reviewer-laget

Maal:
- Forsta prompting, chunking og parsefeil-haandtering.

Aktiviteter:
1. Les reviewer/scouts.py.
2. Les prompts/bug_reviewer.md, prompts/reliability_reviewer.md og prompts/security_reviewer.md.
3. Forsta consolidated vs separate mode.
4. Forsta hvordan non-reviewable chunks hoppes over.

### 14:30-15:30: Verifier og guard

Maal:
- Forsta hvordan funn valideres og publiseres.

Aktiviteter:
1. Les reviewer/verifier.py.
2. Les reviewer/guard.py.
3. Koble policy-felter til endelig status.

Sjekkpunkt:
- Du skal kunne forklare hvorfor et funn blir rejected i verifier,
  og hvorfor et funn senere blir rejected i guard.

### 15:30-16:30: Lever en liten trygg endring

Maal:
- Gjore en avgrenset forbedring med test.

Trygge kandidater:
1. Liten forbedring i CLI-valideringsfeilmelding i reviewer/cli.py + test i tests/test_cli_fake.py.
2. Liten verifier-regresjonstest i tests/test_verifier.py.
3. Liten output-forbedring i reviewer/output.py + oppdatert test.

Krav til endringen:
1. En liten scope, ingen bred refaktorering.
2. Minst en test som dokumenterer atferd.
3. Relevante tester + full suite skal passere.

### 16:30-17:00: Oppsummering og dag 2-klargjoring

Maal:
- Avslutte med tydelig status og neste steg.

Aktiviteter:
1. Kjor full suite en gang til.
2. Noter hva du endret, hvorfor det er trygt, og hvilke tester som dekker det.
3. Noter ett anbefalt fokus for dag 2.

## Done-kriterier for Dag 1

Dag 1 er ferdig nar alle punkter under er oppfylt:
1. Du kan forklare runtime-flyten uten a gjette.
2. Du har gjort minst en liten kodeendring.
3. Endringen har testdekning.
4. Full test-suite er gronn.
5. Du kan forklare risiko og hvorfor endringen er trygg.

## Forslag til Dag 2

Velg ett av disse sporene:
1. Verifier-kvalitet: stram inn eller forenkle en konkret gate med test-forst.
2. Guard-kvalitet: forbedre filtrering som reduserer stoy uten recall-tap.
3. Benchmark-kvalitet: legg til et nytt kontrollcase for en kjent feilmodus.

For dag 2 gjelder samme regel: start med tester, endre lite, maaleffekt i benchmark.
