# Step 3: Weight Shift Experiment Log

## Experiment Design

**Baseline Weights (Current):**
- Genre match: +1.0
- Mood match: +1.0
- Energy proximity: ×3.0

**Experimental Weights (Modified):**
- Genre match: +0.5 (halved)
- Mood match: +1.0 (unchanged)
- Energy proximity: ×6.0 (doubled)

**Hypothesis:** Doubling energy weight and halving genre weight will increase energy-driven recommendations and allow more non-genre matches when energy is very close.

---

## Test Results

### Profile 1: High-Energy Pop

| Position | Baseline | Experimental | Change? |
|----------|----------|--------------|---------|
| #1 | Sunrise City (4.76) | Sunrise City (7.02) | ✅ Same |
| #2 | Gym Hero (3.91) | Gym Hero (6.32) | ✅ Same |
| #3 | Sunset Groove (3.64) | Sunset Groove (6.28) | ✅ Same |
| #4 | Rooftop Lights (3.58) | Rooftop Lights (6.16) | ✅ Same |
| #5 | Retro Future (3.46) | **Storm Runner (5.94)** | ⚠️ Changed |

**Overlap: 4/5**  
**Interpretation:** Stable. Despite higher energy weight, the baseline top-4 remain because they are both genre-matched AND high-energy. Storm Runner enters at #5 because it has extremely high energy (0.91), but doesn't have the genre match boost.

---

### Profile 2: Deep Intense Rock

| Position | Baseline | Experimental | Change? |
|----------|----------|--------------|---------|
| #1 | Storm Runner (4.82) | Storm Runner (7.14) | ✅ Same |
| #2 | Gritty Streets (3.91) | Gritty Streets (6.82) | ✅ Same |
| #3 | Gym Hero (3.76) | Gym Hero (6.52) | ✅ Same |
| #4 | Viking Thunder (3.67) | Viking Thunder (6.34) | ✅ Same |
| #5 | Funkytown Express (3.00) | Funkytown Express (6.00) | ✅ Same |

**Overlap: 5/5**  
**Interpretation:** Perfect stability. This profile has consistent matches across all dimensions (rock genre + intense mood + high energy), so the weight shift doesn't disrupt rankings. All songs are genre-matched, so reducing genre weight doesn't flip the order.

---

### Profile 3: Chill Lofi (The Critical Test)

| Position | Baseline | Experimental | Change? |
|----------|----------|--------------|---------|
| #1 | **Library Rain (4.55)** | **Rainy Day Piano (6.70)** | ❌ CHANGED |
| #2 | Midnight Coding (4.34) | Library Rain (6.60) | ⚠️ Dropped |
| #3 | Rainy Day Piano (3.85) | Spacewalk Thoughts (6.52) | ⚠️ Reordered |
| #4 | Spacewalk Thoughts (3.76) | Midnight Coding (6.18) | ⚠️ Reordered |
| #5 | Focus Flow (3.40) | Focus Flow (5.30) | ✅ Stayed |

**Overlap: 1/5**  
**Interpretation:** Significant disruption. 

- **Baseline:** Library Rain wins because it has BOTH lofi genre match (+1.0) AND mood match (+1.0), energy proximity is only +2.55
- **Experimental:** Rainy Day Piano wins because it has ONLY mood match (+1.0) but exceptional energy proximity (+2.85 → now ×6.0 = +5.70), totaling 6.70

Library Rain is now #2 (6.60) because while it has genre + mood, the reduced genre weight (0.5 instead of 1.0) and increased energy math makes it lose to the pure energy match.

---

## Math Validation

### Library Rain (Baseline vs Experimental)

**Baseline:**
- Genre match (lofi): +1.0
- Mood match (chill): +1.0
- Energy proximity (0.35 energy vs 0.2 target = 0.85 proximity): +2.55
- **Total: 4.55** ✓ Correct

**Experimental:**
- Genre match (lofi): +0.5
- Mood match (chill): +1.0
- Energy proximity (0.85 × 6.0): +5.10
- **Total: 6.60** ✓ Correct

### Rainy Day Piano (Baseline vs Experimental)

**Baseline:**
- Genre match (ambient, not lofi): +0.0
- Mood match (chill): +1.0
- Energy proximity (0.15 energy vs 0.2 target = 0.95 proximity): +2.85
- **Total: 3.85** ✓ Correct

**Experimental:**
- Genre match (ambient, not lofi): +0.0
- Mood match (chill): +1.0
- Energy proximity (0.95 × 6.0): +5.70
- **Total: 6.70** ✓ Correct

---

## Conclusions

### What Changed?

1. **Stable Profiles (High-Energy Pop, Deep Intense Rock):** Top recommendations remain unchanged when users have strong multi-dimensional matches (genre + mood + energy alignment).

2. **Conflicting Profiles (Chill Lofi):** When a user requests low energy, energy weighting directly competes with genre weighting. Doubling energy makes the system prefer "energy-perfect but genre-mismatched" songs over "genre-perfect but energy-slightly-off" songs.

3. **Accuracy Impact:** For Chill Lofi, the experimental result is arguably WORSE in human terms:
   - Baseline ranks lofi songs first (satisfies genre preference)
   - Experimental ranks ambient songs first (satisfies energy but violates genre)
   - User likely cares more about "chill lofi" as a package than pure low energy

### Is It More Accurate or Just Different?

**Answer: Just Different, and arguably worse.**

- Baseline weights (Genre ×1.0, Energy ×3.0) respect user intent better because they balance all three signals.
- Experimental weights (Genre ×0.5, Energy ×6.0) over-optimize for energy and under-value genre, creating "energy-perfect but tonally wrong" recommendations.
- The aggressive energy weighting creates a "workout playlist" bias—everything becomes high-energy-seeking.

### Recommendation

Reduce energy multiplier from 3.0 to 2.0 OR increase genre weight back to 1.5 to restore balance. The baseline weights are well-tuned.

---

## Screenshots

All terminal outputs for the 6 main profiles are in `assets/`:
- `profile-high-energy-pop.png`
- `profile-chill-lofi.png`
- `profile-deep-intense-rock.png`
- `profile-conflicting-happy-low-energy.png`
- `profile-sad-high-energy.png`
- `profile-noisy-mismatch.png`
