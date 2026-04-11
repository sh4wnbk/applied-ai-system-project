# Music Recommender Model Card

## Model Name

🎧 THE SOUND-BENDER

## Goal / Task

This recommender tries to suggest songs a user might like.
 - Looks at genre, mood, and energy.
 - Gives each song a score and returns the top matches.

Status: ACTIVE 🟢

## 📂 Data Used

The dataset has 20 songs in [data/songs.csv](data/songs.csv).
Each song has genre, mood, energy, tempo, valence, danceability, and acousticness.
The catalog is small, so it does not cover every kind of music taste.
Some genres only have one or two songs.

## ⚙️ Algorithm Summary

The model gives points for a genre 🏷️ match, a mood 🎭 match, and energy 🔥 closeness.
Genre and mood each add 1 point.
Energy closeness is multiplied by 3, so it has the strongest effect.
That means a song with close energy can rank high even if one other label does not match.

## ⚠️ Observed Behavior / Biases

One pattern is energy ⚡ dominance.
High-energy songs often rise to the top, even when the user wanted a calmer feel.
This can make songs like "Gym Hero" show up for many different profiles.

Another bias is small-data 📦 bias.
With only 20 songs, the system repeats the same songs for some users because there are not many options.
That is especially true for underrepresented genres like rock, metal, and jazz.

The model also has an energy-gap 📉 trap.
Users with very low energy or very high energy can get weaker matches than mid-range users.
That happens because the energy score is linear and the catalog does not have many songs at the extremes.

## 🧪 Evaluation Process

Tested 6 profiles.
Three were normal profiles and three were adversarial or conflicting profiles.
The profiles were High-Energy Pop, Chill Lofi, Deep Intense Rock, Conflicting Happy but Low Energy, Sad but High Energy, and Noisy Mismatch.

Checked the top 5 results for each profile.
Compared the outputs before and after a weight-shift experiment.
That experiment showed that stronger energy weighting changed the ranking more than the genre weight did.
Compared the catalog genre counts with the recommendation counts to look for repetition and bias.

## Optional Extensions

### Advanced Song Features

The song data was expanded with additional features that were not in the baseline:

- Popularity (0-100)
- Release Year
- Detailed Mood Tag
- Instrumentalness
- Speechiness
- Liveness

In effect, the recommender now looks beyond just genre/mood/energy and can reward songs that better match a preferred era, detailed vibe, and audio texture profile.

### Multiple Scoring Modes

Multiple scoring strategies were built so the same user can be ranked in different ways:

- Conservative mode: balanced and stable with trend/discovery bonuses
- Discovery mode: leans more toward hidden gems and exploratory matching
- Hybrid mode: blends Conservative and Discovery using alpha

In effect, the recommender can behave like a "safe" mode, an "explore" mode, or a blend between the two.

### Diversity and Fairness Logic

A Diversity Penalty of -0.5 was added when an artist is already in the selected top-k list.

In effect, this reduces artist repetition so recommendations feel less repetitive and more varied.

### Visual Top Recommendations Summary Table: 

![Dashboard Side-by-Side](assets/side_by_side_table.png)

A visual terminal summary was built using Rich, including scores and per-mode reasons.

In effect, songs can be quickly compared to see why they rank differently across Conservative, Discovery, and Hybrid modes without reading raw logs.

## ✅ Intended Use and ❌ Non-Intended Use

This system is for classroom learning and simple experiments.
It is good for showing how a scoring rule works.
It is also good for explaining why a result ranked first.

It should not be used as a real music app.
It does not know lyrics, context, or personal history.
It should not be treated as a full picture of a user's taste.

## Ideas 💡 for Improvement

- Reduce the energy multiplier so genre and mood matter more.
- Add more songs, especially for underrepresented genres.
- Add diversity logic so the same songs do not keep repeating.

## Personal 💭 Reflection

The project demonstrated that a recommender system can appear balanced in its design while still producing uneven outcomes in practice. Small adjustments to feature weights altered which songs ranked highest, highlighting the distinction between a mathematically valid score and an output that aligns with expected behavior.

AI tools such as Gemini and Copilot accelerated tasks involving idea testing, output comparison, and summary generation. However, their outputs required verification when rankings depended on precise numerical values, since minor score changes could shift the ordering of recommendations. This was most evident when examining the repeated appearance of “Gym Hero” and when evaluating the effects of weight modifications.

The results also showed that simple, rule‑based systems can generate recommendations that seem appropriate for some user profiles and misaligned for others, even without user‑level learning. If the project were extended, the next steps would include expanding the song dataset, adjusting the weighting scheme, and introducing a diversity constraint to reduce repetition in the top recommendations.
