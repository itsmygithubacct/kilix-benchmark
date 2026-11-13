# kilix-benchmark

JSON suites for local language models, a scripted Kilix 95 session run in
Kilix tabs, and one video played by two terminal players side by side. A
suite names the work and the required shape of the answer. It does not know
which machine runs it.

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

`src/kilix_benchmark/references/kilix95_tab/` holds recorded comparisons.
`2026-11-12.json` is the regular Kilix 95 against the build that draws its
text with Neuronte: one warm-up run per arm and 10 pairs on one laptop. It
keeps the commits, the host description, every run's checks and metrics and
the paired comparison, and no paths. A new record is written with

```sh
PYTHONPATH=src python -m kilix_benchmark.tab_stats --out results/ \
  --reference regular --candidate neuronte \
  --campaign results/campaign-01.json --host host.json \
  --public src/kilix_benchmark/references/kilix95_tab/DATE.json
```

where `host.json` describes the machine in words; `--public` refuses any
string that looks like a local path.

## video side by side

`python -m kilix_benchmark.video_bench` plays one file with kilix-mpv
(`--vo=kitty`, pictures through shared memory, the line the Kilix launcher
uses) and with the Neuronte player (`tools/neuronte-play.py` in a Neuronte
checkout). Its purpose is to show where each player spends its time, so that
the source of either can be improved; it is not a quality comparison.

Every run opens its own tab without taking focus and splits it into two
panes of equal size. A solo run plays the file in one pane while the other
idles. A together run plays it in both. Each kind starts with a warm-up run;
order and sides then alternate. A player is measured from the moment the
terminal takes its first picture until `--seconds` later, when its quit key
is written to its pane.

```sh
PYTHONPATH=src python -m kilix_benchmark.video_bench \
  --video clip.mp4 --start 20 --seconds 20 --pairs 3 \
  --seek 8:right --seek 14:left \
  --neuronte-home /path/to/neuronte --kitten /path/to/kitten --out results/
PYTHONPATH=src python -m kilix_benchmark.video_stats results/
```

`--kitten` is the running terminal's own `kitten`. `--seek SECONDS:KEY`
writes an arrow key to each player that many seconds into its run; both
players seek 5 s on left and right. `--mpv-option` and `--neuronte-option`
pass one extra argument to a player, which is how a suspected fix is tried
against the same measurements.

What a run records, all on the host's monotonic clock:

- each picture a player hands the terminal: when its shared-memory object
  was created, when the player closed it, when the terminal unlinked it (the
  terminal unlinks a picture once it has read it), and its size;
- CPU and memory of each player's whole process tree by program, of
  kilix-mpv's threads by name, and of the terminal by thread, with the bytes
  the terminal passed to write calls;
- kilix-mpv's position, dropped and delayed frame counts and A/V difference,
  and its seek and restart events, over its IPC socket;
- the Neuronte player's own line per source frame: reading, fitting,
  drawing, packing and writing times, how far the picture was behind the
  sound, and frames it dropped or skipped;
- when each seek key was written, and when the first picture from the new
  place reached the terminal;
- load, battery and GPU state before and after.

`video_stats` writes `metrics.json` and `REPORT.md`: one table per kind of
run with the median over measured runs and the range, then each player's CPU
by program and thread and its own counters.

Sound goes to a null output in both players, so both keep time by an audio
clock and nothing is heard.

Measured runs are never shown, so the numbers stop where the terminal has
read a picture. They do not include the terminal drawing it. `--watch N`
adds runs that are brought to the front and captured with `scrot` about five
times a second; a capture in which the middle of a pane is one flat colour
is counted for the player on that side. Those runs take the screen and the
keyboard for their length, disturb the timing, and are reported apart. Pick
a clip without flat frames for them.
