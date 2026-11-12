# kilix-benchmark

JSON suites for local language models, and a scripted Kilix 95 session run
in Kilix tabs. A suite names the work and the required shape of the answer.
It does not know which machine runs it.

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

## kilix95-tab

`src/kilix_benchmark/suites/kilix95_tab.json` is one scripted Kilix 95
session: the Start menu, Notepad opened by keyboard, 90 paced keystrokes, a
410-key burst, caret moves, 120 pointer motions and a window drag, then the
desktop's own quit key. It compares two Kilix 95 checkouts in tabs of the
running Kilix.

`python -m kilix_benchmark.tab_bench` opens a tab per run without taking
focus, runs `tab_probe.py` there on one checkout, and writes the suite's keys
and pointer reports to that tab through one `kitten @ send-text --stdin`. The
keys are encoded as the terminal encodes them under the keyboard flags the
desktop asks for, so they arrive as key presses, not as a paste. Each arm
gets one warm-up run, which also takes a `scrot` screenshot of its tab; then
pairs follow with the leading arm alternating.

```sh
PYTHONPATH=src python -m kilix_benchmark.tab_bench \
  --arm regular=/path/to/kilix-95 --arm neuronte=/path/to/kilix-95-neuronte \
  --pairs 10 --out results/ --scratch /tmp/kilix95-tab \
  --kilix-home /path/to/kilix --state-library /path/to/libkilix-state.so \
  --neuronte-home /path/to/neuronte --kitten /path/to/kitten
PYTHONPATH=src python -m kilix_benchmark.tab_stats --out results/ \
  --reference regular --candidate neuronte
```

`--scratch` holds each run's private storage tree and must be under `/tmp`
or the terminal's session directory: the terminal only deletes frame files
there, and the deletion is how a run sees the terminal take each frame.

What a run records, all on the host's monotonic clock:

- in the desktop (`tab_probe.py`): each drawn frame's render, compose and
  send times, the time spent drawing and measuring text since the previous
  frame, Neuronte mask calls, the frame file, every parsed input batch, the
  startup steps, and CPU time and peak memory at exit;
- in the driver (`tab_bench.py`): every write, every frame file written and
  deleted (inotify), the desktop's CPU ticks and memory every 200 ms, load
  every second, and load, battery and GPU state before and after.

The run's environment switches off sound, the host clipboard bridge, the
model wizard, D-Bus accessibility, X access and remote control of the live
Kilix. The default voice install at startup is a no-op. Nothing writes to the
live Kilix or Kilix 95 data.

`tab_stats` checks each run (clean exit, every key received as a key, no
paste input, every frame taken by the terminal, Notepad opened), pairs the
measured runs in order, and writes `analysis.json` and `SUMMARY.md`. A
difference is called only when the sign test over pairs gives p <= 0.05, the
bootstrap 95% interval of the median paired difference excludes zero, and the
difference is at least 5% of the reference median.

The numbers are application-side. They do not include the terminal drawing
the tab, which measured runs never show, or photon latency.
