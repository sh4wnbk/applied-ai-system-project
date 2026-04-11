from src.recommender import (
    Song,
    UserProfile,
    Recommender,
    ConservativeStrategy,
    DiscoveryStrategy,
    HybridStrategy,
)

def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
            popularity=85,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
            popularity=25,
        ),
    ]
    return Recommender(songs)


def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    # Starter expectation: the pop, happy, high energy song should score higher
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


def test_popularity_bonuses_apply_for_mainstream_and_hidden_gems():
    user = UserProfile(
        favorite_genre="none",
        favorite_mood="none",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()

    mainstream_score, mainstream_reasons = rec.score_song(user, rec.songs[0], popularity_weight=0.0)
    hidden_gem_score, hidden_gem_reasons = rec.score_song(user, rec.songs[1], popularity_weight=0.0)

    assert any("Mainstream bonus" in reason for reason in mainstream_reasons)
    assert any("Hidden Gem bonus" in reason for reason in hidden_gem_reasons)
    # With zero popularity weight and matched target for first song, mainstream should still lead.
    assert mainstream_score > hidden_gem_score


def test_popularity_weight_changes_score():
    user = UserProfile(
        favorite_genre="none",
        favorite_mood="none",
        target_energy=0.8,
        likes_acoustic=False,
    )
    song = Song(
        id=99,
        title="Weighted Track",
        artist="Test Artist",
        genre="none",
        mood="none",
        energy=0.8,
        tempo_bpm=120,
        valence=0.5,
        danceability=0.5,
        acousticness=0.5,
        popularity=60,
    )

    low_weight_score, _ = Recommender.score_song(user, song, popularity_weight=0.5)
    high_weight_score, _ = Recommender.score_song(user, song, popularity_weight=2.0)

    assert high_weight_score > low_weight_score


def test_can_toggle_between_strategies():
    user = UserProfile(
        favorite_genre="none",
        favorite_mood="none",
        target_energy=0.4,
        likes_acoustic=False,
    )
    songs = [
        Song(
            id=1,
            title="Mainstream Candidate",
            artist="A",
            genre="none",
            mood="none",
            energy=0.4,
            tempo_bpm=120,
            valence=0.5,
            danceability=0.5,
            acousticness=0.2,
            popularity=90,
        ),
        Song(
            id=2,
            title="Hidden Gem Candidate",
            artist="B",
            genre="none",
            mood="none",
            energy=0.4,
            tempo_bpm=95,
            valence=0.5,
            danceability=0.5,
            acousticness=0.8,
            popularity=20,
        ),
    ]

    conservative = Recommender(songs, strategy=ConservativeStrategy(popularity_weight=2.0))
    discovery = Recommender(songs, strategy=DiscoveryStrategy(popularity_weight=2.0))

    conservative_hidden_score, _ = conservative.strategy.calculate_score(user, songs[1])
    discovery_hidden_score, discovery_reasons = discovery.strategy.calculate_score(user, songs[1])

    assert discovery_hidden_score > conservative_hidden_score
    assert any("Hidden Gem bonus (+1.0)" in reason for reason in discovery_reasons)


def test_hybrid_alpha_endpoints_match_base_strategies():
    user = UserProfile(
        favorite_genre="jazz",
        favorite_mood="focused",
        target_energy=0.4,
        likes_acoustic=False,
    )
    song = Song(
        id=7,
        title="Hybrid Probe",
        artist="A",
        genre="jazz",
        mood="focused",
        energy=0.4,
        tempo_bpm=100,
        valence=0.5,
        danceability=0.5,
        acousticness=0.5,
        popularity=20,
    )

    conservative_score, _ = ConservativeStrategy(popularity_weight=2.0).calculate_score(user, song)
    discovery_score, _ = DiscoveryStrategy(popularity_weight=2.0).calculate_score(user, song)
    hybrid_conservative_score, _ = HybridStrategy(popularity_weight=2.0, alpha=1.0).calculate_score(user, song)
    hybrid_discovery_score, _ = HybridStrategy(popularity_weight=2.0, alpha=0.0).calculate_score(user, song)

    assert hybrid_conservative_score == conservative_score
    assert hybrid_discovery_score == discovery_score


def test_hybrid_alpha_can_change_ranking_vs_conservative():
    user = UserProfile(
        favorite_genre="none",
        favorite_mood="none",
        target_energy=0.4,
        likes_acoustic=False,
    )
    songs = [
        Song(
            id=1,
            title="Mainstream Fit",
            artist="A",
            genre="none",
            mood="none",
            energy=0.1,
            tempo_bpm=120,
            valence=0.5,
            danceability=0.5,
            acousticness=0.2,
            popularity=90,
        ),
        Song(
            id=2,
            title="Hidden Gem Fit",
            artist="B",
            genre="none",
            mood="none",
            energy=0.4,
            tempo_bpm=95,
            valence=0.5,
            danceability=0.5,
            acousticness=0.8,
            popularity=20,
        ),
    ]

    conservative = Recommender(songs, strategy=ConservativeStrategy(popularity_weight=2.0))
    hybrid = Recommender(songs, strategy=HybridStrategy(popularity_weight=2.0, alpha=0.2))

    conservative_top = conservative.recommend(user, k=1)[0].title
    hybrid_top = hybrid.recommend(user, k=1)[0].title

    assert conservative_top == "Mainstream Fit"
    assert hybrid_top == "Hidden Gem Fit"


def test_diversity_penalty_can_promote_different_artist_in_top_k():
    user = UserProfile(
        favorite_genre="none",
        favorite_mood="none",
        target_energy=0.8,
        likes_acoustic=False,
    )
    songs = [
        Song(
            id=1,
            title="Artist X Prime",
            artist="Artist X",
            genre="none",
            mood="none",
            energy=0.8,
            tempo_bpm=120,
            valence=0.5,
            danceability=0.5,
            acousticness=0.2,
            popularity=80,
        ),
        Song(
            id=2,
            title="Artist X Followup",
            artist="Artist X",
            genre="none",
            mood="none",
            energy=0.79,
            tempo_bpm=118,
            valence=0.5,
            danceability=0.5,
            acousticness=0.2,
            popularity=80,
        ),
        Song(
            id=3,
            title="Artist Y Alternative",
            artist="Artist Y",
            genre="none",
            mood="none",
            energy=0.7,
            tempo_bpm=110,
            valence=0.5,
            danceability=0.5,
            acousticness=0.2,
            popularity=80,
        ),
    ]

    rec = Recommender(songs, strategy=ConservativeStrategy(popularity_weight=1.0))
    results = rec.recommend(user, k=2)

    assert results[0].artist == "Artist X"
    assert results[1].artist == "Artist Y"


def test_diversity_penalty_appears_in_reasons_when_repeat_artist_is_selected():
    user = UserProfile(
        favorite_genre="none",
        favorite_mood="none",
        target_energy=0.8,
        likes_acoustic=False,
    )
    songs = [
        Song(
            id=1,
            title="Artist X Lead",
            artist="Artist X",
            genre="none",
            mood="none",
            energy=0.8,
            tempo_bpm=120,
            valence=0.5,
            danceability=0.5,
            acousticness=0.2,
            popularity=90,
        ),
        Song(
            id=2,
            title="Artist X Strong Followup",
            artist="Artist X",
            genre="none",
            mood="none",
            energy=0.8,
            tempo_bpm=121,
            valence=0.5,
            danceability=0.5,
            acousticness=0.2,
            popularity=88,
        ),
        Song(
            id=3,
            title="Artist Y Weaker Alternative",
            artist="Artist Y",
            genre="none",
            mood="none",
            energy=0.2,
            tempo_bpm=95,
            valence=0.5,
            danceability=0.5,
            acousticness=0.2,
            popularity=60,
        ),
    ]

    rec = Recommender(songs, strategy=ConservativeStrategy(popularity_weight=1.0))
    ranked = rec.recommend_with_details(user, k=3)

    second_song_reasons = ranked[1][2]
    assert ranked[1][0].artist == "Artist X"
    assert any("Diversity Penalty (-0.5)" in reason for reason in second_song_reasons)
