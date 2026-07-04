# Data Plan and Governance

## 1. Principles

Model quality cannot be separated from data quality. Every dataset entering training or evaluation has a manifest, provenance, license review, permitted uses, processing lineage, quality report, split policy, and content-risk assessment.

Data is never described only as “web,” “code,” or “reasoning.” The mixture and transformations must be reconstructable.

## 2. Dataset Manifest

Each immutable dataset version records:

- dataset ID, version, and content digest;
- source locations and retrieval dates;
- license and attribution obligations;
- collection method and applicable consent basis;
- languages, domains, formats, and time range;
- raw and retained sample/token counts;
- deduplication and filtering versions;
- safety, privacy, and quality flags;
- train, validation, test, and quarantine split digests;
- known contamination and limitations;
- responsible maintainer and approval status.

Raw sources, normalized examples, tokenized shards, and mixtures are different artifacts with lineage edges.

## 3. Initial Data Tracks

### Pipeline Validation

Use tiny, legally clear corpora and generated sequences to verify tokenization, packing, loss reduction, checkpoint resume, and overfitting. These runs do not support broad capability claims.

### Synthetic Algorithmic Tasks

Generate controlled tasks for copying, associative recall, variable binding, rule chains, graph reachability, state tracking, arithmetic, and instruction composition. Generators reserve held-out values, graph sizes, chain lengths, vocabulary permutations, and templates to measure extrapolation rather than template memorization.

Every generator is seeded and versioned and can emit the latent solution trace for evaluation. Solution traces are labels, not necessarily desired natural-language chain-of-thought targets.

### Language Pretraining

Start only after a license-compatible mixture is selected. Candidate domains include prose, documentation, permissively licensed code, educational math/science text, dialogue-like instruction data, and multilingual text if multilingual evaluation is funded. Mixture weights are based on target behavior and measured quality rather than raw availability.

### Post-Training

Instruction and preference data require documented annotator guidelines, disagreement handling, privacy controls, quality audits, and separation from test prompts. Preference pairs should vary correctness, calibration, relevance, style, safety, and tool behavior so the model does not learn a shallow length or tone proxy.

## 4. Processing Pipeline

```text
source acquisition
 -> quarantine and malware/type checks
 -> parsing and normalization
 -> language/domain classification
 -> exact and fuzzy deduplication
 -> quality and risk filtering
 -> sensitive-data handling
 -> benchmark-contamination scan
 -> immutable split assignment
 -> tokenization and packing
 -> shard validation and publication
```

Filters never overwrite raw artifacts. Each stage emits counts and reason codes so removals can be audited.

## 5. Deduplication and Contamination

Apply exact hashing before fuzzy or semantic methods. Split assignment should happen at document or source-group level before windowing so near-neighbor chunks cannot cross splits. Benchmark prompts, solutions, translations, and common paraphrases are scanned against training data where feasible.

Contamination analysis is imperfect. Reports describe method, threshold, coverage, and residual risk. A contaminated benchmark may be retained for regression testing but cannot support a generalization claim.

## 6. Privacy and Sensitive Content

Do not intentionally train on secrets, private repositories, leaked data, direct identifiers, or personal communications without lawful authorization and a documented need. Automated PII filters require sampling audits because both false negatives and false positives matter.

Dataset access is role-scoped and logged. Public model release requires a separate memorization and extraction assessment; deleting a source after training does not remove learned influence from an existing checkpoint.

## 7. Safety and Quality Filtering

Track rather than conflate:

- malformed or low-information data;
- spam and duplicated templates;
- harmful instructions or dangerous domain content;
- harassment and sexual content;
- bias and representational imbalance;
- unverifiable or obsolete factual claims;
- machine-generated text;
- policy-violating personal data.

Filtering everything difficult can reduce robustness and domain knowledge. Inclusion decisions depend on purpose, safeguards, and evaluation, and are documented by category.

## 8. Tokenizer

The initial tokenizer evaluation compares at least one practical subword scheme on target languages and code. Report vocabulary size, bytes per token by domain/language, unknown or byte fallback behavior, normalization, whitespace/code preservation, training cost, and compatibility policy.

Tokenizer artifacts are immutable dependencies of checkpoints. Changing a tokenizer creates a new model lineage unless a tested conversion method exists.

## 9. Data Quality Gates

Before a training run, automated validation checks schema, checksums, shard readability, token ranges, document boundaries, leakage, split overlap, source mixture, abnormal repetition, and sample decoding. Human spot checks cover every major source and filter-reason bucket.

## 10. Release Documentation

Every released checkpoint references a data card stating what is known and unknown about sources, languages, licenses, filtering, contamination, personal data, synthetic content, annotators, and intended use. If source disclosure is constrained, the limitation must be explicit rather than replaced by vague “high-quality data” language.
