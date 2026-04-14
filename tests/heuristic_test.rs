//! Integration tests for the heuristic library (Phase 1 Step 1a).
//!
//! These tests play full games by driving both players through
//! `apply_heuristic` and assert behavioural properties:
//!
//! - A game completes (termination, winner set).
//! - Configuration differences manifest as play-pattern differences (e.g.,
//!   `SupporterGreed::Never` → no Supporter plays; `RetreatPolicy::BelowHp`
//!   → at least one retreat when the active is wounded).
//!
//! Running the full-game tests is cheap (< 1s each on debug builds) and
//! exercises `apply_heuristic` against a real board with stack-forced
//! sequences, setup phase, and end-of-turn mechanics — the same surface the
//! wasm bindings hit.

use deckgym::actions::{Action, SimpleAction};
use deckgym::heuristics::{
    apply_heuristic, AttackSelector, BenchPriority, HeuristicConfig, RetreatPolicy,
    SupporterGreed, TargetPriority,
};
use deckgym::models::TrainerType;
use deckgym::state::State;
use deckgym::test_support::{DECK_A, DECK_B};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Drive both players through `apply_heuristic` until the game ends or the
/// action cap is hit. Returns the final state and every action taken.
fn play_full_game(
    config_a: &HeuristicConfig,
    config_b: &HeuristicConfig,
    seed: u64,
    max_steps: usize,
) -> (State, Vec<Action>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut state = State::initialize(&DECK_A, &DECK_B, &mut rng);
    let mut log: Vec<Action> = Vec::with_capacity(200);

    for _ in 0..max_steps {
        if state.winner.is_some() {
            break;
        }
        let (actor, possible) = state.generate_possible_actions();
        if possible.is_empty() {
            panic!(
                "no legal actions at turn {} but game is not over",
                state.turn_count
            );
        }
        let config = if actor == 0 { config_a } else { config_b };
        let action = apply_heuristic(config, &state, &possible);
        deckgym::actions::apply_action_public(&mut rng, &mut state, &action);
        log.push(action);
    }
    (state, log)
}

#[test]
fn default_config_heuristic_games_terminate_across_seeds() {
    let config = HeuristicConfig::default();
    for seed in [1u64, 42, 1234, 2026] {
        let (final_state, log) = play_full_game(&config, &config, seed, 4_000);
        assert!(
            final_state.winner.is_some(),
            "seed {seed}: game did not terminate (turn {}, {} actions)",
            final_state.turn_count,
            log.len()
        );
    }
}

#[test]
fn supporter_greed_never_suppresses_supporter_plays() {
    let mut cfg = HeuristicConfig::default();
    cfg.supporter_greed = SupporterGreed::Never;

    let (_, log) = play_full_game(&cfg, &cfg, 7, 4_000);
    let supporter_plays = log
        .iter()
        .filter(|a| {
            matches!(
                &a.action,
                SimpleAction::Play { trainer_card }
                    if trainer_card.trainer_card_type == TrainerType::Supporter
            )
        })
        .count();
    assert_eq!(
        supporter_plays, 0,
        "SupporterGreed::Never still produced {supporter_plays} supporter plays"
    );
}

#[test]
fn supporter_greed_always_actually_plays_supporters() {
    // Reference baseline: the default (Always) config should play at least
    // one supporter over the course of a full game against itself, given the
    // test decks have multiple supporters.
    let cfg = HeuristicConfig::default();
    let (_, log) = play_full_game(&cfg, &cfg, 7, 4_000);
    let supporter_plays = log
        .iter()
        .filter(|a| {
            matches!(
                &a.action,
                SimpleAction::Play { trainer_card }
                    if trainer_card.trainer_card_type == TrainerType::Supporter
            )
        })
        .count();
    assert!(
        supporter_plays > 0,
        "SupporterGreed::Always never played a supporter across a full game — the \
         baseline is broken",
    );
}

#[test]
fn attack_selector_highest_damage_picks_largest_attack_when_two_offered() {
    // Narrow property test: if a single turn is synthesized where two
    // Attack actions are legal, HighestDamage picks the one with the
    // higher fixed_damage value. This avoids the noisy full-game path.
    use deckgym::card_ids::CardId;
    use deckgym::state::PlayedCard;
    let mut state = State::default();
    state.in_play_pokemon[0][0] = Some(PlayedCard::from_id(CardId::A1004VenusaurEx));

    let attack0 = Action {
        actor: 0,
        action: SimpleAction::Attack(0),
        is_stack: false,
    };
    let attack1 = Action {
        actor: 0,
        action: SimpleAction::Attack(1),
        is_stack: false,
    };
    let end_turn = Action {
        actor: 0,
        action: SimpleAction::EndTurn,
        is_stack: false,
    };
    let legal = vec![end_turn, attack0, attack1.clone()];

    let mut cfg = HeuristicConfig::default();
    cfg.attack_selector = AttackSelector::HighestDamage;
    let picked = apply_heuristic(&cfg, &state, &legal);
    // Venusaur ex attack indices: 0 = Razor Leaf (60), 1 = Giant Bloom (100).
    assert_eq!(picked, attack1, "HighestDamage should select Giant Bloom");
}

#[test]
fn heuristic_config_round_trips_through_json() {
    // Belt-and-suspenders check that HeuristicConfig serialises cleanly
    // through the same serde_json path the wasm binding uses — protects
    // against accidental tag-rename regressions.
    let cfg = HeuristicConfig {
        target_priority: TargetPriority::HighestThreat,
        retreat_policy: RetreatPolicy::BelowHpRatio(0.25),
        attack_selector: AttackSelector::HighestDamage,
        supporter_greed: SupporterGreed::Always,
        bench_priority: BenchPriority::HighestHp,
    };
    let json = serde_json::to_string(&cfg).expect("serialise");
    let parsed: HeuristicConfig = serde_json::from_str(&json).expect("deserialise");
    assert_eq!(cfg, parsed);
    // And a human-readable sanity check: BelowHpRatio should round-trip
    // its float.
    assert!(
        json.contains("BelowHpRatio")
            && json.contains("0.25"),
        "unexpected serialisation: {json}"
    );
}
