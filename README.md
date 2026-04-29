# Music Recommender: Music Theory

**Project 3 was a simulation. Music Theory is the machine.**

---

## 1. Base Project Identification

This project extends the Codepath AI110 Module 3 project: **Music Recommender Simulation**.

The original system simulated how platforms like Spotify predict user preferences. It operated on a static, hand-coded 20-song catalog and matched listener profiles to songs using a weighted scoring function that combined genre, mood, and energy. That system demonstrated recommendation logic in a controlled environment — it could not retrieve live data, adapt to feedback, or explain its reasoning. Every output was deterministic and pre-seeded.

Music Theory replaces the simulation with a working machine. The catalog is live. The reasoning is observable. The recommendations explain themselves.

---

## 2. What This System Does

Music Theory accepts a listener profile — four audio dimension values and a set of genre or mood tags — and returns a prioritized, explained five-song trajectory.

The system retrieves songs from two live data sources simultaneously:

<img src="assets/lastfm.png" alt="Last.fm" height="28" style="vertical-align:middle"/> Last.fm (track-level data) &nbsp;&nbsp; <img src="assets/radio_browser.png" alt="Radio Browser" height="28" style="vertical-align:middle"/> Radio Browser (live station directory) A cosine similarity engine scores every retrieved song against the listener's profile vector. A language model generates a Glass Box explanation for each top candidate — an explanation that names the specific dimensions that drove the match, states the numerical score, and lists which tags overlapped. A critique agent evaluates the explanation set and requests a second retrieval pass if the quality is insufficient. A final ranking agent selects five songs and arranges them as a listening trajectory.

Every intermediate step is visible in the terminal. Every recommendation earns its place with evidence.

### MasterMix — Precision Tempo Alignment

MasterMix is an optional beat-matching mode that adds a third live data source and full BPM awareness to the scoring and selection pipeline.

```bash
python main.py --profile afrobeats --mastermix            # uses profile's target_bpm
python main.py --profile afrobeats --mastermix --bpm 105  # override to any tempo (0–300)
python main.py --profile jazz --mastermix --bpm 80        # activate on any profile
```

The `--bpm` flag sets or overrides the profile's `target_bpm` at runtime — no file editing required. `--mastermix` without a declared BPM (in the profile or via `--bpm`) is rejected at the CLI with a plain-language error.

**What changes when MasterMix is active:**

Tempo introduces BPM as a fifth cosine dimension. The listener's target tempo is Min-Max normalized against the catalog range and scored alongside energy, valence, danceability, and acousticness. Maestro applies a hard ±5 BPM proximity filter to the explained candidate pool before its selection step — tracks inside the window are preferred, tracks with no BPM data are always included as neutral. The BPM column appears in the Top Scored Songs table, the Final Trajectory table, and each track card in the output — the filter is visible, not hidden.

**The BPM pipeline — three phases:**

After Misty completes dual-source retrieval, MasterMix triggers a three-phase BPM enrichment via the [MeloData API](https://melodata.voltenworks.com/):

1. **Track enrichment** — each Last.fm track is searched by title and artist to resolve its ISRC. Resolved ISRCs are submitted in a batch features call to populate `SongFeature.bpm` with high-accuracy values from real audio analysis (Essentia engine). Featured-artist credits are stripped from titles before search to improve match rates.

2. **Artist seed fallback** — if direct track lookup yields no ISRCs (common when Last.fm returns mainstream pop tracks not indexed in MeloData), the system retries using artist-name-only queries across all distinct Last.fm artists. One resolved ISRC is sufficient to proceed to Phase 3.

3. **Catalog discovery** — resolved ISRCs seed a call to `/v1/recommendations` with the profile's target features (`target_bpm`, `energy`, `danceability`, `valence`). MeloData returns up to 20 tracks from its own catalog that match the target audio signature — these arrive with BPM and full audio features already attached. They enter the catalog as `source="melodata"` and are scored and explained alongside Last.fm and Radio Browser candidates.

This means MasterMix can surface tracks from a third source that are specifically aligned to the listener's target tempo, even when the primary catalog sources have no BPM overlap with MeloData's index.

Radio Browser stations carry no ISRC and are always treated as neutral — never excluded by the BPM filter, never submitted to MeloData.

If `MELODATA_API_KEY` is absent, all three phases are silently skipped. MasterMix degrades gracefully to four-dimensional cosine scoring with no filter applied.

---

## 3. System Architecture

### Component Diagram

![Component Diagram](assets/architecture_component.png)

The component diagram reflects the actual implementation. Seven named agent components are shown inside the system boundary, each corresponding to a specific file. The Gatekeeper sits outside the LangGraph subgraph — it runs before the graph initializes and gates access to it. AgentState is the shared state cylinder that all nodes read from and write to. Tempo is the only internal component with no external API dependency; it uses numpy cosine similarity exclusively. The critique loop edge from Hertz back to Misty is labeled with its condition: `loop_back = True` and `loop_count < 3`.

### Sequence Diagram

![Sequence Diagram](assets/architecture_sequence.png)

The sequence diagram reflects the actual runtime flow across a single session. All 14 participants appear in order of first activation. The `par` block shows Last.fm and Radio Browser being called simultaneously inside a `ThreadPoolExecutor`. The `alt` block at the Gatekeeper shows all three outcomes: flagged input, API failure (fail closed), and cleared input. The `loop` block shows Prestige making one API call per song. The `alt` block at Hertz shows the conditional loop-back branch and the ceiling-reached branch. Numbered steps correspond to the 24-step sequence in the HANDOFF specification.

---

## 4. Agent Cast

| Name | Role | File |
| --- | --- | --- |
| Cass | Input + Output | `nodes/` (main.py) |
| Misty | Retrieve | `nodes/retrieve.py` |
| Tempo | Score | `nodes/score.py` |
| Prestige | Explain + RAG | `nodes/explain.py` |
| Hertz | Critique | `nodes/critique.py` |
| Maestro | Orchestrate + Rank | `nodes/rerank.py` |
| Base | Narrator | `display/agents.py` |

<table>
  <tr>
    <td colspan="3" align="center" valign="bottom"><img src="assets/base.png" alt="Base" width="180"/><br/><b>Base</b></td>
  </tr>
  <tr>
    <td align="center" valign="bottom"><img src="assets/cass.png" alt="Cass" width="120"/><br/><b>Cass</b></td>
    <td align="center" valign="bottom"><img src="assets/misty.png" alt="Misty" width="120"/><br/><b>Misty</b></td>
    <td align="center" valign="bottom"><img src="assets/tempo.png" alt="Tempo" width="120"/><br/><b>Tempo</b></td>
  </tr>
  <tr>
    <td align="center" valign="bottom"><img src="assets/prestige.png" alt="Prestige" width="120"/><br/><b>Prestige</b></td>
    <td align="center" valign="bottom"><img src="assets/hertz.png" alt="Hertz" width="120"/><br/><b>Hertz</b></td>
    <td align="center" valign="bottom"><img src="assets/maestro.png" alt="Maestro" width="120"/><br/><b>Maestro</b></td>
  </tr>
</table>

**Cass** opens and closes every session — the Sony Walkman is always the first thing and the last thing the listener sees.

**Misty** listens to two sources at once. The Neumann U87 is a studio microphone built for capturing everything in the room simultaneously — Misty calls Last.fm and Radio Browser in parallel, never one at a time.

**Tempo** counts without guessing. The metronome does not interpret; it measures. Tempo runs cosine similarity — deterministic math with a correct answer — and produces a ranked list with per-dimension evidence.

**Prestige** opens the glass box. The Technics SL-1200 is a precision instrument that reveals the mechanics of the music. Prestige writes explanations that name the numbers, not just the feelings.

**Hertz** reads the signal, not the story. The VU meter does not care what the music sounds like — it measures whether the level is right. Hertz evaluates whether explanations meet the Glass Box standard and sends the system back if they do not.

**Maestro** arranges the set list. From the approved candidates, Maestro selects five songs and sequences them as a trajectory — a listening journey with movement, not just a sorted score table.

**Base** narrates. The upright bass sets the foundation that everything else rests on. Base opens the session, closes it, and provides the numbers that summarize what the system produced.

---

## 5. Setup Instructions

### Step 0: Install Kitty terminal

Image rendering (character portraits during node execution) requires the Kitty terminal emulator. Kitty supports inline image display via the icat protocol.

Download: <https://sw.kovidgoyal.net/kitty/>

**Non-Kitty terminals will display text output only — images will not render. All recommendation output and Glass Box explanations are fully available without Kitty.**

### Step 1: Clone the repository

```bash
git clone <repository-url>
cd applied-ai-system-project
```

### Step 2: Activate the environment and install dependencies

```bash
source ~/.venvs/ai-engineering/bin/activate
pip install -r requirements.txt
```

### Step 3: Configure .env

```bash
cp .env.example .env
```

Open `.env` and add your API keys. Two keys are required: `ANTHROPIC_API_KEY` and `LASTFM_API_KEY`. A third key, `MELODATA_API_KEY`, is optional — it enables MasterMix BPM enrichment when present. Without it, `--mastermix` activates but the BPM filter has no effect.

### Step 4: Run the recommender

```bash
python main.py                                    # default: afrobeats profile
python main.py --profile ambient                  # quiet, acoustic session
python main.py --profile jazz                     # Sunday morning profile
python main.py --profile afrobeats --mastermix    # BPM-matched at profile's target (100 BPM)
python main.py --profile afrobeats --mastermix --bpm 105   # override target BPM
python main.py --profile jazz --mastermix --bpm 80         # MasterMix on any profile
```

### Step 5: Run the eval harness

```bash
python eval/harness.py
```

Runs 10 predefined profiles through the full system and prints a pass/fail summary table.

---

## 6. Demo Walkthrough

[Video Walkthrough](https://drive.google.com/file/d/1xc2dvkt3l3kcnHrHUrRv7B1UrRVKy31k/view?usp=sharing)

---

## 8. Sample Interactions

### Example 1: Afrobeats Session (Standard)

```bash
python main.py --profile afrobeats
```

**Input TasteProfile:**

```text
Name:         Afrobeats Session
Energy:       0.85  Valence: 0.80  Danceability: 0.90  Acousticness: 0.15
Tags:         afrobeats, dance, african, pop
Context:      High-energy party playlist for a Friday night gathering.
```

**Misty:** 32 tracks from Last.fm · 38 stations from Radio Browser · Catalog: 70  
**Tempo:** Scored 70 songs · Top score: 0.9968 · 4 dimensions  
**Prestige:** 10 Glass Box explanations · Avg confidence: 0.97  
**Hertz:** Confidence 0.92 — approved  
**Maestro:** 5-song trajectory selected · Source diversity: lastfm + radio

**Sample track card:**

```text
Track 3: WDR COSMO - Afrobeats  by Germany  [radio]
  Similarity: 0.9485
  Energy: 0.3175  Valence: 0.1992  Dance: 0.3944  Acoustic: 0.0374
  Tag overlap: afrobeats, dance
  Explanation: This track scores 0.9485 similarity. Danceability (0.3944)
  and energy (0.3175) are its strongest contributors — both align with your
  high-energy party profile. The tags 'afrobeats' and 'dance' overlap (2
  matches). Afrobeats draws on West African percussion and groove traditions,
  which directly drives the exceptionally high danceability signal.
  Stream: https://wdr-cosmo-afrobeat.icecastssl.wdr.de/wdr/cosmo/afrobeat/mp3/128/stream.mp3
```

---

### Example 2: Afrobeats Session — MasterMix at 105 BPM

```bash
python main.py --profile afrobeats --mastermix --bpm 105
```

**Input TasteProfile:**

```text
Name:         Afrobeats Session
Energy:       0.85  Valence: 0.80  Danceability: 0.90  Acousticness: 0.15
Target BPM:   105.0  (set via --bpm)
MasterMix:    ON — ±5 BPM filter active
```

**BPM Enrichment (3 phases):**

```text
Last.fm tracks searched:        32
BPM values resolved:             4   ← direct ISRC match
MeloData recommendations added: 18   ← Phase 3 catalog discovery
Radio Browser stations:         38   (neutral — no ISRC)
```

**Tempo:** Scored 88 songs · 5 dimensions (energy · valence · danceability · acousticness · bpm)

**Final Trajectory table (with BPM column):**

```text
 #  Title                      Artist         Source    Score   Confidence  BPM
 1  Ye                         Burna Boy      melodata  0.9841  0.97        104
 2  Just Dance                 Lady Gaga      lastfm    0.9968  0.99        —
 3  WDR COSMO - Afrobeats      Germany        radio     0.9485  0.97        —
 4  Essence                    Wizkid         melodata  0.9812  0.98        107
 5  Don't Stop the Music       Rihanna        lastfm    0.9968  0.97        —
```

MeloData tracks show BPM; Last.fm and radio tracks without resolved BPM show `—` and remain eligible as neutral candidates.

---

### Example 2: Late Night Ambient

**Input TasteProfile:**

```text
Name:         Late Night Ambient
Energy:       0.20
Valence:      0.45
Danceability: 0.15
Acousticness: 0.85
Tags:         ambient, chill, acoustic, meditation
Context:      Wind-down session after a long day. Need something quiet.
```

**Gatekeeper:** All fields cleared moderation.

**Misty:** 22 songs from Last.fm, 14 stations from Radio Browser. Merged: 36 items.

**Tempo:** Top score: 0.9718 (high acousticness alignment dominates).

**Prestige:** 10 explanations generated. Avg confidence: 0.81.

**Hertz:** Confidence 0.81 — approved.

**Final Trajectory (sample):**

```text
Track 1: Stars of the Lid — Requiem for Dying Mothers  [lastfm]
  Similarity: 0.9718
  Energy: 0.0281  Valence: 0.1421  Dance: 0.0198  Acoustic: 0.3818
  Tag overlap: ambient, chill
  Explanation: This track scores 0.9718 similarity. Acousticness (0.3818) is
  the strongest contributor — the song is almost entirely acoustic texture
  with no rhythmic drive, matching your 0.85 acousticness target precisely.
  Energy (0.0281) is negligible, consistent with your 0.20 energy floor.
  The tags 'ambient' and 'chill' overlap with your preferred tags. Stars of
  the Lid is a Texas-based orchestral ambient duo whose work is used
  clinically for relaxation and focus protocols.
```

---

### Example 3: Sunday Jazz

**Input TasteProfile:**

```text
Name:         Sunday Jazz
Energy:       0.50
Valence:      0.75
Danceability: 0.40
Acousticness: 0.65
Tags:         jazz, soul, blues, classic
Context:      Sunday morning coffee. Relaxed but engaged.
```

**Gatekeeper:** All fields cleared moderation.

**Misty:** 29 songs from Last.fm, 11 stations from Radio Browser. Merged: 40 items.

**Tempo:** Top score: 0.9312. Valence and acousticness are the dominant axes.

**Prestige:** 10 explanations generated. Avg confidence: 0.79.

**Hertz:** Confidence 0.79 — approved.

**Final Trajectory (sample):**

```text
Track 1: Miles Davis — Kind of Blue  [lastfm]
  Similarity: 0.9312
  Energy: 0.1821  Valence: 0.2914  Dance: 0.1201  Acoustic: 0.2376
  Tag overlap: jazz, blues, classic
  Explanation: This track scores 0.9312 similarity. Valence (0.2914) and
  acousticness (0.2376) are the leading contributors — both reflect the
  warm, unhurried tone of the album. Three tags overlap: 'jazz', 'blues',
  and 'classic'. Kind of Blue (1959) is the best-selling jazz album of all
  time and is widely cited as the defining modal jazz recording. Its
  moderate energy aligns with your 0.50 energy target — present but not
  demanding.
```

---

## 9. Design Decisions

### Tempo uses no LLM

Cosine similarity is deterministic math. Given a fixed input pair, it produces a single correct answer. Delegating this calculation to a language model would introduce variance, cost, and the possibility of a numerically wrong result. Tempo uses numpy directly. This decision is documented explicitly because it is a deliberate rejection of the default assumption that more AI is always better.

### Moderation via Claude Haiku, not a dedicated moderation API

The Gatekeeper uses Claude Haiku (via the Anthropic API) for content moderation rather than a dedicated moderation service such as the OpenAI Moderation API. This choice eliminates a second API key and a second external dependency — the system uses a single provider (Anthropic) for all LLM calls. Claude Haiku supports 100+ languages, which covers the multilingual threat in the threat model. The trade-off is that a general-purpose model is less specialized for moderation than a purpose-built classifier, but structured tool use constrains output to a boolean verdict that limits model drift.

### Fail closed on moderation timeout

When the Claude Haiku moderation call is unavailable, the system rejects the input rather than passing it to the graph unmoderated. The cost of letting harmful input reach the LLM nodes is higher than the cost of one failed session. The user receives a plain-language message and is asked to retry.

### Multi-source retrieval design

Last.fm and Radio Browser are called simultaneously rather than sequentially. This halves the effective wait time and ensures both sources contribute to the catalog regardless of which one responds first. The source field on every SongFeature makes the origin traceable through the entire pipeline.

### Loop ceiling at 3 iterations

Without a hard ceiling, a persistently low-quality catalog could cause the critique loop to run indefinitely. Three iterations give the system two additional chances to improve on the first pass while bounding the session time. On reaching the ceiling, the best available result is delivered and the ceiling is flagged in the terminal output and log.

### MasterMix is opt-in, not always-on

MasterMix adds BPM-based beat-matching and a third live catalog source (MeloData), but activates only when the listener explicitly passes `--mastermix`. An always-on filter would silently exclude high-similarity candidates that fall outside the ±5 BPM window. Listeners who do not need rhythmic consistency should not have their results constrained by a rule they never set.

When active, BPM enters the pipeline at two deterministic points. Tempo adds it as a fifth cosine dimension — `target_bpm` is Min-Max normalized against the catalog range and scored alongside the four base dimensions. Maestro applies a hard ±5 BPM proximity filter before its LLM selection; tracks with no BPM metadata pass unconditionally. If fewer than two candidates carry BPM data, the filter disables itself rather than returning a near-empty pool.

The `--bpm` flag makes target BPM a runtime argument rather than a compile-time constant. Any profile can be used with MasterMix by supplying `--bpm <value>` at the command line — the value is validated (0–300) and applied to the profile before the session starts. Passing `--mastermix` with no BPM (no flag, no profile default) is rejected at the CLI with a plain-language error. The constraint is explicit or it is not active.

The MeloData catalog discovery step (Phase 3) means MasterMix can introduce tracks the listener would not have seen in a standard session — tracks sourced directly from MeloData's audio-analyzed index, matched to the listener's target tempo and feature vector.

### Prestige uses Sonnet; other nodes use Haiku

Glass Box explanation requires reasoning depth — the model must hold the scoring evidence, the tag overlap, and relevant cultural context simultaneously and integrate them into a coherent paragraph. Sonnet is the appropriate tool for this task. Misty, Hertz, and Maestro perform structured classification and extraction tasks where Haiku is sufficient and meaningfully faster.

---

## 10. Testing Summary

Results from `python eval/harness.py` — 10 profiles across 6 edge cases and 4 normal profiles.

The harness runs every profile through the full pipeline and prints a pass/fail summary table. Edge case profiles validate boundary conditions and guardrail behavior. Normal profiles confirm that the system returns 5 songs with confidence ≥ 0.7 across diverse listener types.

**Pass criteria per profile:**

- Profile 1 (all zeros): passes if 5 songs returned and cosine handles zero-vector gracefully
- Profile 2 (all ones): passes if 5 songs returned
- Profile 3 (empty tags): passes if Pydantic validator rejects before graph
- Profile 4 (non-English context): passes if moderation handles Spanish input and graph proceeds normally
- Profile 5 (200-char context): passes if Pydantic accepts the boundary value
- Profile 6 (minimal valid): passes if optional fields default correctly
- Profiles 7–10 (diverse normal): pass if 5 songs returned, confidence ≥ 0.7, source mix present

A dedicated MasterMix unit test suite is in `tests/test_mastermix.py` (34 tests). It covers BPM range computation, Min-Max normalization boundary cases, 4D/5D vector dimension switching, per-dimension breakdown key correctness, filter window inclusion/exclusion, neutral-track passthrough, filter fallback when fewer than 2 candidates pass, and all new model field constraints. These tests run without API calls: `pytest tests/test_mastermix.py -v`.

---

## 11. Reflection

Music Theory demonstrates that agentic engineering is a discipline of constraints, not capabilities.

The interesting decisions in this system are not the language model calls — those are commodity at this point. The interesting decisions are the ones that restrict what the language model is allowed to do: Tempo does not use an LLM because math has a correct answer; the Gatekeeper fails closed because unmoderated input is a harder problem than a failed session; the loop ceiling exists because unbounded recursion is not a feature.

Each of those decisions required understanding not just what the system can do, but what it should not do and what happens when it cannot. That is what separates an agentic system from a demo: the edge cases are designed, not discovered after the fact.

The Glass Box principle — making every intermediate reasoning step visible and auditable — is both a product decision and an engineering constraint. It forces the system to produce evidence, not just outputs. A recommendation that cannot cite its own reasoning is not a recommendation; it is a guess with formatting.

Building this system makes the limits of current recommendation architectures more legible: the tag quality problem (Last.fm tags are user-generated and inconsistent), the cold start problem (no user history, only the single submitted profile), the cultural coverage problem (the data sources skew toward Western English-language music), and the confidence measurement problem (the scores measure internal consistency, not ground truth). These are documented in the model card, not because they are failures, but because honest documentation of limitations is part of what makes a system trustworthy.

---

## What This Project Says About Me as an AI Engineer

This project shows that I build systems that are honest about what they do. The interesting decisions weren't about adding more AI — they were about knowing what each agent should and should not do.
FYI, the app is not finished. As it stands, a user is placing their trust in a wall of text. That will not do. The app needs to implement one of the best sensors we humans are born with. To be continued...

| Agent | Role | Design Decision |
|---|---|---|
| Misty | Retrieve | Calls Last.fm and Radio Browser simultaneously — never one at a time |
| Tempo | Score | Uses cosine similarity, not a language model — math has a correct answer |
| Prestige | Explain | Every explanation names the score, the dimensions, and the evidence |
| Hertz | Critique | Nothing passes that it isn't confident in — loops back if quality is low |
| Maestro | Rank | Arranges a trajectory, not just a sorted list |
| Base | Narrator | Opens and closes every session — the system has a beginning and an end |
| Cass | Input + Output | First thing the listener sees, last thing they see |

A system that hides what it can't do isn't trustworthy. I documented the limitations, the threat model, and the failure modes not because I had to — because that's the kind of engineer I'm trying to be.
