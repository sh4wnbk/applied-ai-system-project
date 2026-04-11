# Music Recommender Model Card

## 1. Model Name

VibeFinder Classroom Edition

---

## 2. Intended Use

This recommender suggests 5 songs from a small catalog based on what a user says they like.
It is built for classroom learning, not for production use.

The model assumes a user can be described with only three inputs:
- favorite genre
- favorite mood
- target energy (from 0.0 to 1.0)

---

## 3. How the Model Works

The model gives each song a score.

It adds points for:
- genre match
- mood match
- energy closeness

In this version, energy closeness is multiplied by 3, so energy has a strong effect.
This means high-energy songs can still rank near the top even when genre or mood does not match.

Plain-language example:
If someone asks for "Happy Pop" but also gives very high energy, songs like "Gym Hero" can keep showing up because they are very close on energy and still partly match the request.

---

## 4. Data

The dataset has 20 songs in [data/songs.csv](data/songs.csv).

Genres include pop, lofi, rock, ambient, jazz, synthwave, indie pop, funk, metal, house, folk, and hiphop.
Moods include happy, chill, intense, focused, and energetic.

This is a tiny catalog, so it does not represent the full range of music taste.

---

## 5. Strengths

The system works well when the user profile is clear and consistent.

Examples from testing:
- "Chill Lofi" returned lofi/chill songs near the top.
- "Deep Intense Rock" returned intense high-energy songs at the top.

Because the logic is simple, results are easy to explain to non-programmers.

---

## 6. Limitations and Bias

One weakness is energy dominance.
When the energy target is high, intense songs rise even if genre or mood does not match.

That can create a filter-bubble effect around "workout" tracks.
In our tests, "Viking Thunder" became top-1 for 2 out of 6 profiles, including mismatch cases.
"Gym Hero" appeared in multiple top-5 lists because its energy is very close to high-energy requests.

Another limitation is small-data bias.
With only 20 songs, some profile types have too few true matches, so the model falls back to songs that are "numerically close" rather than "taste-correct."

---

## 7. Evaluation

We ran 6 profiles (3 standard + 3 adversarial/conflicting):
- High-Energy Pop
- Chill Lofi
- Deep Intense Rock
- Conflicting Happy but Low Energy
- Sad but High Energy
- Noisy Mismatch

What we measured:
- top-5 recommendations per profile
- top-1 repetition rate
- unique songs across all recommendations
- genre distribution in catalog vs genre distribution in recommendations

Results:
- Top-1 repetition rate: 0.33 (same song can dominate across different users)
- Unique songs in all 30 recommended slots: 15
- Catalog pop count: 2 songs, but pop appeared 6 times in recommendations
- Lofi had 3 songs in catalog and appeared 4 times in recommendations

Interpretation:
The recommender is responsive to user inputs, but strong energy weighting can override mood and genre intent in edge cases.

---

## 8. Future Work

- Rebalance weights so genre and mood are not overwhelmed by energy
- Add diversity logic so the same top songs do not repeat as often
- Add a "must match mood" option for users who care more about feeling than intensity
- Expand dataset coverage so more profile types have fair representation

---

## 9. Personal Reflection

This project showed how easy it is for a simple scoring rule to look "smart" while still missing user intent.
Even with clear inputs, a weight choice can push the model toward songs that feel wrong to a person.

It also changed how I think about music apps.
If a song like "Gym Hero" keeps appearing, it may not mean the app "understands" me; it may just be over-prioritizing one signal.
Human judgment is still important for deciding whether recommendations feel useful, not just mathematically consistent.
