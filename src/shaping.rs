//! Reward-shaping signal helpers.
//!
//! Extracted from `wasm_bindings.rs` so the math is testable without the
//! `wasm` feature flag. `board_value_for_player` is the V function used by
//! the Phase 2 Step 0j HP-remaining-potential shape.

use crate::models::Card;
use crate::state::State;

/// Phase 2 Step 0j — prize-weighted "KO-progress" potential.
///
/// `V(state, player) = Σ_in_play prize × (1 − (hp/max_hp)²) +
/// Σ_discard_pokemon prize × 1.0`.
///
/// The discard-pile term is the critical one: it makes V monotonically
/// non-decreasing under damage and KO, so the delta `V(post) − V(pre)` is
///
/// - `+positive` for damage dealt to an in-play Pokémon,
/// - `+positive` for a KO (the slot empties; the discard term picks up the
///   remaining `prize × ratio²` that the in-play term hadn't yet booked),
/// - `0` for a retreat or forced bench-promote (the Pokémon just moved
///   slots), and
/// - `negative` for a heal (rare, penalised under the opposite sign).
///
/// Before this fix, KO'd Pokémon contributed zero to V (slot goes to
/// `None` and the discard pile wasn't consulted), so `V(post) − V(pre)`
/// was large and *negative* on every KO event — the agent was punished
/// for the outcome it was supposed to be rewarded for. See
/// `devlogs/2026-04-18-phase-2-step-0j-validation.md` for the measurement
/// that surfaced the bug.
///
/// The discard pile is shared between KO'd Pokémon and cards discarded
/// from hand via trainer effects; the `Card::Pokemon(_)` filter keeps
/// only the former from contributing.
pub(crate) fn board_value_for_player(state: &State, player: usize) -> f32 {
    let mut v: f32 = 0.0;
    for slot in 0..4 {
        if let Some(pc) = &state.in_play_pokemon[player][slot] {
            let max_hp = pc.get_effective_total_hp().max(1) as f32;
            let remaining = pc.get_remaining_hp() as f32;
            let ratio = (remaining / max_hp).clamp(0.0, 1.0);
            let ko_proximity = 1.0 - ratio * ratio;
            let prize = pc.card.get_knockout_points() as f32;
            v += prize * ko_proximity;
        }
    }
    for card in &state.discard_piles[player] {
        if matches!(card, Card::Pokemon(_)) {
            v += card.get_knockout_points() as f32;
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card_ids::CardId;
    use crate::database::get_card_by_enum;
    use crate::state::PlayedCard;

    // Bulbasaur: 70 HP regular basic (1 prize)
    // Charmander: 60 HP regular basic (1 prize)
    // Venusaur ex: 190 HP EX (2 prizes)

    fn in_play(card_id: CardId, remaining_hp: u32) -> PlayedCard {
        let card = get_card_by_enum(card_id);
        let base_hp = match &card {
            Card::Pokemon(p) => p.hp,
            _ => panic!("expected Pokemon card"),
        };
        assert!(remaining_hp <= base_hp);
        PlayedCard::new(card, base_hp - remaining_hp, base_hp, vec![], false, vec![])
    }

    #[test]
    fn full_hp_contributes_zero() {
        let mut s = State::default();
        s.in_play_pokemon[0][0] = Some(in_play(CardId::A1001Bulbasaur, 70));
        assert!(board_value_for_player(&s, 0).abs() < 1e-6);
    }

    #[test]
    fn half_hp_basic_contributes_prize_times_three_quarters() {
        // 35/70 → ratio 0.5 → 1 − 0.25 = 0.75 × 1 prize
        let mut s = State::default();
        s.in_play_pokemon[0][0] = Some(in_play(CardId::A1001Bulbasaur, 35));
        assert!((board_value_for_player(&s, 0) - 0.75).abs() < 1e-6);
    }

    #[test]
    fn ex_contributes_prize_weighted() {
        // Venusaur ex at 95/190 (half): 2 × 0.75 = 1.5
        let mut s = State::default();
        s.in_play_pokemon[0][0] = Some(in_play(CardId::A1004VenusaurEx, 95));
        assert!((board_value_for_player(&s, 0) - 1.5).abs() < 1e-6);
    }

    #[test]
    fn discarded_pokemon_contributes_full_prize() {
        let mut s = State::default();
        s.discard_piles[0].push(get_card_by_enum(CardId::A1001Bulbasaur));
        s.discard_piles[0].push(get_card_by_enum(CardId::A1004VenusaurEx));
        // 1 (Bulbasaur) + 2 (Venusaur ex) = 3.0
        assert!((board_value_for_player(&s, 0) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn trainer_cards_in_discard_do_not_contribute() {
        let mut s = State::default();
        s.discard_piles[0].push(get_card_by_enum(CardId::PA001Potion));
        assert!(board_value_for_player(&s, 0).abs() < 1e-6);
    }

    /// The bug scenario: finishing-KO on a near-dead Pokémon.
    /// Pre: Bulbasaur at 10/70 HP in play → V = 1 × (1 − (1/7)²) ≈ 0.9796
    /// Post: Bulbasaur in discard, slot empty → V = 0 + 1.0 = 1.0
    /// Delta must be positive (the residual `ratio² × prize`).
    #[test]
    fn ko_finisher_gives_positive_delta() {
        let mut pre = State::default();
        pre.in_play_pokemon[0][0] = Some(in_play(CardId::A1001Bulbasaur, 10));

        let mut post = State::default();
        post.discard_piles[0].push(get_card_by_enum(CardId::A1001Bulbasaur));

        let delta = board_value_for_player(&post, 0) - board_value_for_player(&pre, 0);
        let expected = 1.0 - (1.0 - (10.0_f32 / 70.0).powi(2));
        assert!(delta > 0.0, "expected positive delta, got {delta}");
        assert!((delta - expected).abs() < 1e-6);
    }

    /// One-shot KO: full HP → discarded. Delta must be `+prize`.
    #[test]
    fn one_shot_ko_delta_equals_full_prize() {
        let mut pre = State::default();
        pre.in_play_pokemon[0][0] = Some(in_play(CardId::A1001Bulbasaur, 70));

        let mut post = State::default();
        post.discard_piles[0].push(get_card_by_enum(CardId::A1001Bulbasaur));

        let delta = board_value_for_player(&post, 0) - board_value_for_player(&pre, 0);
        assert!((delta - 1.0).abs() < 1e-6);
    }

    /// One-shot KO of a 2-prize EX must contribute +2.0.
    #[test]
    fn one_shot_ex_ko_delta_equals_two() {
        let mut pre = State::default();
        pre.in_play_pokemon[0][0] = Some(in_play(CardId::A1004VenusaurEx, 190));

        let mut post = State::default();
        post.discard_piles[0].push(get_card_by_enum(CardId::A1004VenusaurEx));

        let delta = board_value_for_player(&post, 0) - board_value_for_player(&pre, 0);
        assert!((delta - 2.0).abs() < 1e-6);
    }

    /// Retreat swaps active with a benched Pokémon. V is invariant to slot.
    #[test]
    fn retreat_preserves_v() {
        let mut pre = State::default();
        pre.in_play_pokemon[0][0] = Some(in_play(CardId::A1001Bulbasaur, 10));
        pre.in_play_pokemon[0][1] = Some(in_play(CardId::A1033Charmander, 60));

        let mut post = State::default();
        post.in_play_pokemon[0][0] = Some(in_play(CardId::A1033Charmander, 60));
        post.in_play_pokemon[0][1] = Some(in_play(CardId::A1001Bulbasaur, 10));

        let delta = board_value_for_player(&post, 0) - board_value_for_player(&pre, 0);
        assert!(delta.abs() < 1e-6);
    }

    /// KO + bench-promote: the normal Pocket KO outcome. Active Pokémon dies,
    /// bench promotes to active. Delta must be positive (the KO finisher),
    /// equal to the residual `prize × ratio²` of the KO'd Pokémon — *not*
    /// the healthy promoted Pokémon's full prize value.
    #[test]
    fn ko_with_bench_promote_gives_finisher_sized_positive_delta() {
        let mut pre = State::default();
        pre.in_play_pokemon[0][0] = Some(in_play(CardId::A1001Bulbasaur, 10));
        pre.in_play_pokemon[0][1] = Some(in_play(CardId::A1033Charmander, 60));

        let mut post = State::default();
        post.in_play_pokemon[0][0] = Some(in_play(CardId::A1033Charmander, 60));
        post.discard_piles[0].push(get_card_by_enum(CardId::A1001Bulbasaur));

        let delta = board_value_for_player(&post, 0) - board_value_for_player(&pre, 0);
        let expected = (10.0_f32 / 70.0).powi(2);
        assert!(delta > 0.0, "KO-with-promote delta must be positive, got {delta}");
        assert!(
            (delta - expected).abs() < 1e-6,
            "expected {expected}, got {delta}"
        );
    }
}
