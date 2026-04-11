# Reflection Comparisons

Profiles compared:
- High-Energy Pop
- Chill Lofi
- Deep Intense Rock
- Conflicting Happy but Low Energy
- Sad but High Energy
- Noisy Mismatch

## Pairwise Comparison Notes

1. High-Energy Pop vs Chill Lofi:
High-Energy Pop returns brighter and more active songs, while Chill Lofi shifts to calmer tracks with lower energy. This makes sense because the energy target is very different.

2. High-Energy Pop vs Deep Intense Rock:
Both profiles like high energy, but Rock gets more intense tracks while Pop keeps more upbeat tracks. The overlap happens because both ask for high energy.

3. High-Energy Pop vs Conflicting Happy but Low Energy:
Both ask for happy songs, but low energy pulls the list toward softer tracks. This shows energy can change the recommendation direction even when mood is the same.

4. High-Energy Pop vs Sad but High Energy:
Both ask for high energy, so energetic songs can overlap even though mood and genre differ. This is one reason songs like Gym Hero can appear in multiple lists.

5. High-Energy Pop vs Noisy Mismatch:
Noisy Mismatch has no clean genre/mood match, so it mostly gets high-energy fallback songs. High-Energy Pop gets more relevant matches because at least one preference aligns.

6. Chill Lofi vs Deep Intense Rock:
These two profiles are opposite in both mood and energy, so their outputs are very different. Lofi gets chill tracks, while Rock gets intense tracks.

7. Chill Lofi vs Conflicting Happy but Low Energy:
Both ask for lower energy, so some calmer songs overlap. The mood difference explains why not all top songs are the same.

8. Chill Lofi vs Sad but High Energy:
These profiles differ strongly on energy, and the outputs reflect that. Sad but High Energy still receives intense tracks because energy is weighted heavily.

9. Chill Lofi vs Noisy Mismatch:
Chill Lofi has clear matches and gets focused recommendations. Noisy Mismatch has weak categorical matches, so it relies more on numeric energy closeness.

10. Deep Intense Rock vs Conflicting Happy but Low Energy:
Rock gets heavy and intense songs, while Conflicting Happy but Low Energy gets softer songs. This split shows the model is sensitive to energy direction.

11. Deep Intense Rock vs Sad but High Energy:
Both can receive intense high-energy songs, even with different genre requests. This shows energy can dominate when it is close to the target.

12. Deep Intense Rock vs Noisy Mismatch:
Rock profile has meaningful genre/mood matches, but Noisy Mismatch does not. Noisy Mismatch behaves more like a fallback search for high-energy songs.

13. Conflicting Happy but Low Energy vs Sad but High Energy:
These two profiles are opposites on energy, and their outputs diverge sharply. Low-energy profiles avoid most intense tracks unless no better match exists.

14. Conflicting Happy but Low Energy vs Noisy Mismatch:
Both can produce some imperfect matches, but for different reasons. Conflicting profile has partial mood/genre alignment, while Noisy Mismatch has almost none.

15. Sad but High Energy vs Noisy Mismatch:
These two can look similar because both push for high energy and often miss genre/mood matches. That is another sign that energy weighting may be too strong.
