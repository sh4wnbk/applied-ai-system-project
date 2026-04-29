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

The system retrieves songs from two live data sources simultaneously: Last.fm (track-level data) and Radio Browser (live station directory). A cosine similarity engine scores every retrieved song against the listener's profile vector. A language model generates a Glass Box explanation for each top candidate — an explanation that names the specific dimensions that drove the match, states the numerical score, and lists which tags overlapped. A critique agent evaluates the explanation set and requests a second retrieval pass if the quality is insufficient. A final ranking agent selects five songs and arranges them as a listening trajectory.

Every intermediate step is visible in the terminal. Every recommendation earns its place with evidence.

### MasterMix — Precision Tempo Alignment

MasterMix is an optional beat-matching mode that adds BPM awareness to the scoring and selection pipeline. It is activated by passing `--mastermix` at the command line together with a `target_bpm` value declared in the listener profile.

When active, two things change. Tempo introduces BPM as a fifth cosine dimension alongside energy, valence, danceability, and acousticness — the listener's target tempo is scored against every candidate after Min-Max normalization across the catalog range. Maestro then applies a hard ±5 BPM proximity filter to the candidate pool before its final selection. Tracks within five beats per minute of the target pass; tracks outside are excluded. Tracks with no BPM data are treated as neutral and always remain eligible — the system does not penalize a candidate for missing metadata.

MasterMix is designed for contexts where rhythmic consistency matters as much as genre: a workout session that must hold a training cadence, a DJ-style set where tracks need to transition without jarring tempo shifts, or any listening context where pace defines the experience as much as sound.

BPM metadata is populated via the MeloData API enrichment step. Until that integration is in place, all retrieved tracks carry no BPM data and MasterMix degrades gracefully — the flag is accepted, Tempo scores on four dimensions as normal, and the candidate pool is unfiltered. When BPM data is present, the Glass Box explanation for each recommended track states the track's tempo and whether it falls within the requested window. The constraint is visible, not hidden.

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
    <td align="center"><img src="assets/cass.png" alt="Cass" width="120"/><br/><b>Cass</b></td>
    <td align="center"><img src="assets/misty.png" alt="Misty" width="120"/><br/><b>Misty</b></td>
    <td align="center"><img src="assets/tempo.png" alt="Tempo" width="120"/><br/><b>Tempo</b></td>
    <td align="center"><img src="assets/prestige.png" alt="Prestige" width="120"/><br/><b>Prestige</b></td>
  </tr>
  <tr>
    <td align="center"><img src="assets/hertz.png" alt="Hertz" width="120"/><br/><b>Hertz</b></td>
    <td align="center"><img src="assets/maestro.png" alt="Maestro" width="120"/><br/><b>Maestro</b></td>
    <td align="center"><img src="assets/base.png" alt="Base" width="120"/><br/><b>Base</b></td>
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
python main.py                              # default: afrobeats profile
python main.py --profile ambient            # quiet, acoustic session
python main.py --profile jazz               # Sunday morning profile
python main.py --profile afrobeats --mastermix  # BPM-matched trajectory (requires target_bpm in profile and MELODATA_API_KEY)
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

### Example 1: Afrobeats Session

**Input TasteProfile:**

```text
Name:         Afrobeats Session
Energy:       0.85
Valence:      0.80
Danceability: 0.90
Acousticness: 0.15
Tags:         afrobeats, dance, african, pop
Context:      High-energy party playlist for a Friday night gathering.
```

**Gatekeeper:** All fields cleared moderation. Profile passed to graph.

**Misty:** Called Last.fm and Radio Browser simultaneously. Retrieved 34 songs from Last.fm, 18 stations from Radio Browser. Merged catalog: 52 items.

**Tempo:** Scored 52 songs via cosine similarity. Top score: 0.9421. Songs sorted by similarity descending.

**Prestige:** Generated Glass Box explanations for top 10 candidates.

**Hertz:** Confidence 0.84 — approved. No loop-back required.

**Maestro:** Selected 5-song trajectory prioritizing source diversity and tag overlap.

**Final Trajectory (sample):**

```text
Track 1: Burna Boy — Last Last  [lastfm]
  Similarity: 0.9421
  Energy: 0.3312  Valence: 0.2891  Dance: 0.3118  Acoustic: 0.0100
  Tag overlap: afrobeats, dance
  Explanation: This track scores 0.9421 similarity. Danceability (0.3118) and
  energy (0.3312) are the dominant contributors — both align with your
  high-danceability, high-energy profile. The tags 'afrobeats' and 'dance'
  overlap directly with your preferred tags. Afrobeats as a genre carries a
  long tradition of rhythmic complexity rooted in West African percussion,
  which drives the strong danceability signal. Valence (0.2891) reflects the
  track's celebratory tone, consistent with your 0.80 valence target.
```

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

MasterMix adds BPM-based beat-matching to the recommendation pipeline, but it activates only when the listener explicitly requests it with `--mastermix`. An always-on filter would silently exclude high-similarity candidates that fall just outside the ±5 BPM window. Listeners who do not need rhythmic consistency should not have their results constrained by a rule they never set.

When active, BPM enters the pipeline at two points. Tempo introduces it as a fifth cosine dimension — the listener's `target_bpm` is normalized against the catalog range and scored alongside energy, valence, danceability, and acousticness. Maestro then applies a hard ±5 BPM proximity filter to the explained candidate pool before its LLM selection step. That filter is deterministic, not a language model judgment. Tracks with no BPM metadata pass unconditionally. If fewer than two candidates carry BPM data, the filter disables itself rather than returning a near-empty pool.

The flag requires `target_bpm` to be declared in the listener profile. Passing `--mastermix` without a declared tempo is rejected at the CLI with a plain-language error. The system does not substitute a default tempo and silently activate BPM scoring — the constraint is explicit or it is not active.

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
