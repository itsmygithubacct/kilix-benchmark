# kilix-benchmark

JSON suites for local language models. A suite names the questions, the
decoding, and the required JSON shape. It does not know which machine runs
the model.

## college-mcq

`src/kilix_benchmark/suites/college_mcq.json` is a 24-question college
questionnaire. Each item has four choices. The reply must be one JSON object
whose only key is `choice` and whose value is `A`, `B`, `C`, or `D`.

Scoring drops one leading `<think>` block and one markdown fence, then parses
the JSON. An extra key, a bare letter, or the wrong letter fails.

`src/kilix_benchmark/suites/college_models.json` pins the two GGUF files and
the llama.cpp b10174 engine builds used to serve them. The lab queue reads
that file. It is not a score by itself.

```sh
python -m kilix_benchmark.mcq --suite src/kilix_benchmark/suites/college_mcq.json \
  --model name --results results.json
```

`python -m kilix_benchmark.codex_exec` asks `codex exec` for each item and
writes a reference file. The prompt is the suite prompt. stdin is closed, the
sandbox is read-only, and the last message is the recorded reply.

```sh
python -m kilix_benchmark.codex_exec \
  --suite src/kilix_benchmark/suites/college_mcq.json \
  --model gpt-6-luna --effort high \
  --output src/kilix_benchmark/references/college_mcq/gpt-6-luna-high.json
```
