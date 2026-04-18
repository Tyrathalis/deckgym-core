//! Configurable heuristic action picker for Ditto Trainer's Phase 1 engine.
//!
//! The heuristic is a *configuration* — each decision type (promotion target,
//! retreat policy, attack selection, supporter greed, bench priority) is
//! exposed as a small enum. `apply_heuristic` routes the current legal-action
//! list to the matching sub-picker and falls back to a fixed action-type
//! priority when several types are legal at once.
//!
//! One source of truth for campaign opponents, gym sparring partners, and
//! (Phase 2+) Ditto's heuristic scaffolding. Keeping the module here rather
//! than under `wasm_bindings/` means non-wasm targets (Rust unit +
//! integration tests) can use it without pulling in `wasm-bindgen`.
//!
//! Thresholds live in enum fields (`BelowHp(u32)`), not constants —
//! parameter tuning is a TS config change, not a rebuild.

use serde::{Deserialize, Serialize};

use crate::actions::{Action, SimpleAction};
use crate::models::{Card, TrainerType};
use crate::state::State;

// =============================================================================
// Sub-module configuration enums
// =============================================================================

/// How the heuristic picks which of its own benched Pokémon to promote after
/// the active is knocked out. (TCG Pocket attacks have fixed targets, so the
/// only "pick a Pokémon" decision opponents routinely face is promotion.)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TargetPriority {
    /// Promote the benched Pokémon with the **lowest** remaining HP. Useful
    /// for "sacrifice" personalities that feed the front line and keep their
    /// healthy Pokémon in reserve.
    LowestHp,
    /// Promote the benched Pokémon with the **highest** remaining HP.
    /// Defensive default — keeps the board alive.
    HighestHp,
    /// Promote the benched Pokémon with the **highest max attack damage**.
    /// Threat-minded: puts the biggest hitter up front.
    HighestThreat,
    /// Pick the first legal `Activate` (stable tie-breaker, matches the
    /// engine's enumeration order).
    First,
}

/// Whether and when the heuristic should retreat the active Pokémon.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RetreatPolicy {
    /// Never retreat.
    Never,
    /// Retreat when the active's **remaining HP** is at or below the
    /// threshold (absolute HP, not damage counters).
    BelowHp(u32),
    /// Retreat when the active's **remaining HP ratio** is at or below the
    /// threshold (0.0..=1.0). `0.25` ≈ "retreat at 25% HP".
    BelowHpRatio(f32),
}

/// Which attack to pick when multiple are legal (same active, several
/// attacks with enough energy).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AttackSelector {
    /// Attack with the highest `fixed_damage` value. Ties broken by
    /// enumeration order.
    HighestDamage,
    /// Pick the first legal attack (matches the engine's enumeration order).
    First,
}

/// How aggressively to play Supporter cards when they're legal.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SupporterGreed {
    /// Play the first legal Supporter whenever one is available (once per
    /// turn — the engine enforces the limit).
    Always,
    /// Never play Supporters.
    Never,
}

/// How the heuristic picks which basic Pokémon to place onto the bench (or
/// into the active slot during setup).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BenchPriority {
    /// Prefer the Pokémon with the highest printed HP.
    HighestHp,
    /// Pick the first legal `Place` action (matches the engine's
    /// enumeration order).
    First,
}

/// Full heuristic configuration. One per opponent/sparring partner; also used
/// (Phase 2+) as the surface Ditto's behaviour-scaffold writes into.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeuristicConfig {
    pub target_priority: TargetPriority,
    pub retreat_policy: RetreatPolicy,
    pub attack_selector: AttackSelector,
    pub supporter_greed: SupporterGreed,
    pub bench_priority: BenchPriority,
}

impl Default for HeuristicConfig {
    /// Sensible "average competent play" defaults: never retreat, always play
    /// supporters, swing for highest damage, place highest-HP pokemon first,
    /// promote the sturdier benched Pokémon after a KO.
    fn default() -> Self {
        Self {
            target_priority: TargetPriority::HighestHp,
            retreat_policy: RetreatPolicy::Never,
            attack_selector: AttackSelector::HighestDamage,
            supporter_greed: SupporterGreed::Always,
            bench_priority: BenchPriority::HighestHp,
        }
    }
}

// =============================================================================
// Sub-module pickers
// =============================================================================

/// Pick an `Activate` action (post-KO promotion) per `TargetPriority`.
/// Returns `None` if no `Activate` action is legal.
pub fn pick_activate(priority: &TargetPriority, state: &State, legal: &[Action]) -> Option<Action> {
    let activates: Vec<&Action> = legal
        .iter()
        .filter(|a| matches!(a.action, SimpleAction::Activate { .. }))
        .collect();
    if activates.is_empty() {
        return None;
    }
    let chosen = match priority {
        TargetPriority::First => activates[0],
        TargetPriority::LowestHp => activates
            .iter()
            .min_by_key(|a| promotion_hp(state, a))
            .copied()
            .unwrap_or(activates[0]),
        TargetPriority::HighestHp => activates
            .iter()
            .max_by_key(|a| promotion_hp(state, a))
            .copied()
            .unwrap_or(activates[0]),
        TargetPriority::HighestThreat => activates
            .iter()
            .max_by_key(|a| promotion_threat(state, a))
            .copied()
            .unwrap_or(activates[0]),
    };
    Some(chosen.clone())
}

/// Pick a `Retreat` action per `RetreatPolicy`. Returns `None` if no
/// `Retreat` action is legal or the policy doesn't want to retreat now.
pub fn pick_retreat(policy: &RetreatPolicy, state: &State, legal: &[Action]) -> Option<Action> {
    let retreats: Vec<&Action> = legal
        .iter()
        .filter(|a| matches!(a.action, SimpleAction::Retreat(_)))
        .collect();
    if retreats.is_empty() {
        return None;
    }
    let actor = retreats[0].actor;
    let active = state.in_play_pokemon[actor][0].as_ref()?;
    let remaining = active.get_remaining_hp();
    match policy {
        RetreatPolicy::Never => None,
        RetreatPolicy::BelowHp(threshold) => {
            if remaining <= *threshold {
                Some(retreats[0].clone())
            } else {
                None
            }
        }
        RetreatPolicy::BelowHpRatio(ratio) => {
            let total = active.get_effective_total_hp();
            if total == 0 {
                return None;
            }
            let actual_ratio = remaining as f32 / total as f32;
            if actual_ratio <= *ratio {
                Some(retreats[0].clone())
            } else {
                None
            }
        }
    }
}

/// Pick an `Attack` (or `UseCopiedAttack`) per `AttackSelector`. Returns
/// `None` if no attack is legal.
pub fn pick_attack(selector: &AttackSelector, state: &State, legal: &[Action]) -> Option<Action> {
    let attacks: Vec<&Action> = legal
        .iter()
        .filter(|a| {
            matches!(
                a.action,
                SimpleAction::Attack(_) | SimpleAction::UseCopiedAttack { .. }
            )
        })
        .collect();
    if attacks.is_empty() {
        return None;
    }
    let chosen = match selector {
        AttackSelector::First => attacks[0],
        AttackSelector::HighestDamage => attacks
            .iter()
            .max_by_key(|a| attack_damage(state, a))
            .copied()
            .unwrap_or(attacks[0]),
    };
    Some(chosen.clone())
}

/// Pick a Supporter `Play` action per `SupporterGreed`. Returns `None` if no
/// Supporter is legal or the policy says not to play one.
pub fn pick_supporter(policy: &SupporterGreed, legal: &[Action]) -> Option<Action> {
    if matches!(policy, SupporterGreed::Never) {
        return None;
    }
    legal
        .iter()
        .find(|a| {
            matches!(
                &a.action,
                SimpleAction::Play { trainer_card }
                    if trainer_card.trainer_card_type == TrainerType::Supporter
            )
        })
        .cloned()
}

/// Pick a `Place` action per `BenchPriority`. Returns `None` if no `Place`
/// is legal.
pub fn pick_place(priority: &BenchPriority, legal: &[Action]) -> Option<Action> {
    let places: Vec<&Action> = legal
        .iter()
        .filter(|a| matches!(a.action, SimpleAction::Place(_, _)))
        .collect();
    if places.is_empty() {
        return None;
    }
    let chosen = match priority {
        BenchPriority::First => places[0],
        BenchPriority::HighestHp => places
            .iter()
            .max_by_key(|a| place_hp(a))
            .copied()
            .unwrap_or(places[0]),
    };
    Some(chosen.clone())
}

// =============================================================================
// Top-level router
// =============================================================================

/// Route the legal-action list to the matching sub-module per `config`.
///
/// Precedence is a fixed play-order (stack-forced > ability > supporter >
/// place > evolve > non-supporter trainer > turn-energy attach > retreat >
/// attack > miscellaneous > EndTurn). This is how a typical "setup → attack"
/// turn unfolds, one action per call.
///
/// Panics if `legal` is empty (every call site already requires at least one
/// legal action — propagating that invariant is the cleanest contract).
pub fn apply_heuristic(config: &HeuristicConfig, state: &State, legal: &[Action]) -> Action {
    assert!(
        !legal.is_empty(),
        "apply_heuristic called with empty legal list"
    );

    // Stack-forced choices (is_stack) go through the matching sub-module if
    // there is one, and otherwise fall back to the first option. The engine
    // only produces one stack frame at a time, so all legal actions share
    // the same is_stack flag.
    if legal[0].is_stack {
        if let Some(a) = pick_activate(&config.target_priority, state, legal) {
            return a;
        }
        return legal[0].clone();
    }

    // Free-play priority ordering (setup > finishing):
    //   1. Use Ability (free value; most are strict upgrades)
    //   2. Play Supporter (once per turn — SupporterGreed)
    //   3. Place basic on bench / active during setup (BenchPriority)
    //   4. Evolve (always — evolution makes decks stronger)
    //   5. Play non-supporter trainer (items/tools/stadium play & discard)
    //   6. Attach the turn's Energy Zone energy
    //   7. Retreat (RetreatPolicy)
    //   8. Attack (AttackSelector — typically ends the turn)
    //   9. Any remaining non-trivial sub-actions
    //  10. EndTurn (fallback)

    if let Some(a) = first_matching(legal, |a| {
        matches!(a.action, SimpleAction::UseAbility { .. })
    }) {
        return a;
    }
    if let Some(a) = pick_supporter(&config.supporter_greed, legal) {
        return a;
    }
    if let Some(a) = pick_place(&config.bench_priority, legal) {
        return a;
    }
    if let Some(a) = first_matching(legal, |a| matches!(a.action, SimpleAction::Evolve { .. })) {
        return a;
    }
    if let Some(a) = first_matching(legal, is_non_supporter_trainer) {
        return a;
    }
    if let Some(a) = first_matching(legal, |a| {
        matches!(
            a.action,
            SimpleAction::Attach {
                is_turn_energy: true,
                ..
            }
        )
    }) {
        return a;
    }
    if let Some(a) = pick_retreat(&config.retreat_policy, state, legal) {
        return a;
    }
    if let Some(a) = pick_attack(&config.attack_selector, state, legal) {
        return a;
    }
    // Anything else (AttachTool, MoveEnergy, Heal, ...) that isn't EndTurn —
    // and, when SupporterGreed is Never, isn't a Supporter Play either. The
    // latter guard matters because the fallback is a catch-all; without it,
    // a turn that has only `EndTurn` plus a Supporter `Play` left would hand
    // the Supporter to the heuristic even though the config forbids it.
    let forbid_supporters = matches!(config.supporter_greed, SupporterGreed::Never);
    if let Some(a) = first_matching(legal, |a| {
        if matches!(a.action, SimpleAction::EndTurn) {
            return false;
        }
        if forbid_supporters && is_supporter_play(a) {
            return false;
        }
        true
    }) {
        return a;
    }
    // EndTurn is the only remaining option.
    legal[0].clone()
}

// =============================================================================
// Helpers
// =============================================================================

fn first_matching<F: Fn(&Action) -> bool>(legal: &[Action], predicate: F) -> Option<Action> {
    legal.iter().find(|a| predicate(a)).cloned()
}

fn is_non_supporter_trainer(action: &Action) -> bool {
    matches!(
        &action.action,
        SimpleAction::Play { trainer_card }
            if trainer_card.trainer_card_type != TrainerType::Supporter
    )
}

fn is_supporter_play(action: &Action) -> bool {
    matches!(
        &action.action,
        SimpleAction::Play { trainer_card }
            if trainer_card.trainer_card_type == TrainerType::Supporter
    )
}

fn promotion_hp(state: &State, action: &Action) -> u32 {
    if let SimpleAction::Activate {
        player,
        in_play_idx,
    } = action.action
    {
        state.in_play_pokemon[player][in_play_idx]
            .as_ref()
            .map(|p| p.get_remaining_hp())
            .unwrap_or(0)
    } else {
        0
    }
}

fn promotion_threat(state: &State, action: &Action) -> u32 {
    if let SimpleAction::Activate {
        player,
        in_play_idx,
    } = action.action
    {
        state.in_play_pokemon[player][in_play_idx]
            .as_ref()
            .map(|p| max_attack_damage(&p.card))
            .unwrap_or(0)
    } else {
        0
    }
}

fn place_hp(action: &Action) -> u32 {
    if let SimpleAction::Place(card, _) = &action.action {
        match card {
            Card::Pokemon(p) => p.hp,
            _ => 0,
        }
    } else {
        0
    }
}

fn attack_damage(state: &State, action: &Action) -> u32 {
    match &action.action {
        SimpleAction::Attack(idx) => {
            let actor = action.actor;
            state.in_play_pokemon[actor][0]
                .as_ref()
                .and_then(|p| attack_damage_for_card(&p.card, *idx))
                .unwrap_or(0)
        }
        SimpleAction::UseCopiedAttack {
            source_player,
            source_in_play_idx,
            attack_index,
            ..
        } => state.in_play_pokemon[*source_player][*source_in_play_idx]
            .as_ref()
            .and_then(|p| attack_damage_for_card(&p.card, *attack_index))
            .unwrap_or(0),
        _ => 0,
    }
}

fn attack_damage_for_card(card: &Card, attack_index: usize) -> Option<u32> {
    if let Card::Pokemon(p) = card {
        p.attacks.get(attack_index).map(|a| a.fixed_damage)
    } else {
        None
    }
}

fn max_attack_damage(card: &Card) -> u32 {
    if let Card::Pokemon(p) = card {
        p.attacks.iter().map(|a| a.fixed_damage).max().unwrap_or(0)
    } else {
        0
    }
}

// =============================================================================
// Unit tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actions::{Action, SimpleAction};
    use crate::card_ids::CardId;
    use crate::database::get_card_by_enum;
    use crate::models::{Card, EnergyType};
    use crate::state::PlayedCard;

    fn mk_played(card_id: CardId, remaining_hp: u32) -> PlayedCard {
        PlayedCard::from_id(card_id).with_remaining_hp(remaining_hp)
    }

    fn mk_action(actor: usize, action: SimpleAction, is_stack: bool) -> Action {
        Action {
            actor,
            action,
            is_stack,
        }
    }

    fn activate(actor: usize, in_play_idx: usize) -> Action {
        mk_action(
            actor,
            SimpleAction::Activate {
                player: actor,
                in_play_idx,
            },
            true,
        )
    }

    fn retreat(actor: usize, bench_idx: usize) -> Action {
        mk_action(actor, SimpleAction::Retreat(bench_idx), false)
    }

    fn attack(actor: usize, index: usize) -> Action {
        mk_action(actor, SimpleAction::Attack(index), false)
    }

    fn place(actor: usize, card: Card, slot: usize) -> Action {
        mk_action(actor, SimpleAction::Place(card, slot), false)
    }

    fn end_turn(actor: usize) -> Action {
        mk_action(actor, SimpleAction::EndTurn, false)
    }

    fn attach_turn(actor: usize) -> Action {
        mk_action(
            actor,
            SimpleAction::Attach {
                attachments: vec![(1, EnergyType::Grass, 0)],
                is_turn_energy: true,
            },
            false,
        )
    }

    fn state_with_bench(actor: usize, slots: Vec<(usize, PlayedCard)>) -> State {
        let mut state = State::default();
        for (idx, p) in slots {
            state.in_play_pokemon[actor][idx] = Some(p);
        }
        state
    }

    // --- TargetPriority ---

    #[test]
    fn target_priority_first_picks_the_first_activate() {
        let state = state_with_bench(
            0,
            vec![
                (1, mk_played(CardId::A1001Bulbasaur, 50)),
                (2, mk_played(CardId::A1021Exeggcute, 30)),
            ],
        );
        let legal = vec![activate(0, 1), activate(0, 2)];
        let pick = pick_activate(&TargetPriority::First, &state, &legal).unwrap();
        assert_eq!(pick, legal[0]);
    }

    #[test]
    fn target_priority_lowest_hp_picks_lowest_hp() {
        let state = state_with_bench(
            0,
            vec![
                (1, mk_played(CardId::A1001Bulbasaur, 50)),
                (2, mk_played(CardId::A1021Exeggcute, 30)),
                (3, mk_played(CardId::A1002Ivysaur, 80)),
            ],
        );
        let legal = vec![activate(0, 1), activate(0, 2), activate(0, 3)];
        let pick = pick_activate(&TargetPriority::LowestHp, &state, &legal).unwrap();
        assert_eq!(pick, legal[1], "Exeggcute at 30hp should be picked");
    }

    #[test]
    fn target_priority_highest_hp_picks_highest_hp() {
        let state = state_with_bench(
            0,
            vec![
                (1, mk_played(CardId::A1001Bulbasaur, 50)),
                (2, mk_played(CardId::A1021Exeggcute, 30)),
                (3, mk_played(CardId::A1002Ivysaur, 80)),
            ],
        );
        let legal = vec![activate(0, 1), activate(0, 2), activate(0, 3)];
        let pick = pick_activate(&TargetPriority::HighestHp, &state, &legal).unwrap();
        assert_eq!(pick, legal[2], "Ivysaur at 80hp should be picked");
    }

    #[test]
    fn target_priority_highest_threat_picks_highest_max_attack_damage() {
        // Exeggcute max attack is 20, Exeggutor ex is 80 — the EX should win on threat
        let state = state_with_bench(
            0,
            vec![
                (1, mk_played(CardId::A1021Exeggcute, 50)),
                (2, mk_played(CardId::A1023ExeggutorEx, 50)),
            ],
        );
        let legal = vec![activate(0, 1), activate(0, 2)];
        let pick = pick_activate(&TargetPriority::HighestThreat, &state, &legal).unwrap();
        assert_eq!(pick, legal[1], "Exeggutor ex has higher max-damage attacks");
    }

    #[test]
    fn target_priority_returns_none_when_no_activate_in_legal_list() {
        let state = State::default();
        let legal = vec![end_turn(0)];
        assert!(pick_activate(&TargetPriority::First, &state, &legal).is_none());
    }

    // --- RetreatPolicy ---

    #[test]
    fn retreat_policy_never_returns_none() {
        let mut state = State::default();
        state.in_play_pokemon[0][0] = Some(mk_played(CardId::A1001Bulbasaur, 10));
        let legal = vec![retreat(0, 1)];
        assert!(pick_retreat(&RetreatPolicy::Never, &state, &legal).is_none());
    }

    #[test]
    fn retreat_policy_below_hp_triggers_when_at_or_below_threshold() {
        let mut state = State::default();
        state.in_play_pokemon[0][0] = Some(mk_played(CardId::A1001Bulbasaur, 20));
        let legal = vec![retreat(0, 1)];
        assert!(pick_retreat(&RetreatPolicy::BelowHp(20), &state, &legal).is_some());
        assert!(pick_retreat(&RetreatPolicy::BelowHp(10), &state, &legal).is_none());
    }

    #[test]
    fn retreat_policy_below_hp_ratio_triggers_at_or_below_threshold() {
        let mut state = State::default();
        // Bulbasaur base HP is 70 — remaining 20 is ~0.286
        state.in_play_pokemon[0][0] = Some(mk_played(CardId::A1001Bulbasaur, 20));
        let legal = vec![retreat(0, 1)];
        assert!(pick_retreat(&RetreatPolicy::BelowHpRatio(0.30), &state, &legal).is_some());
        assert!(pick_retreat(&RetreatPolicy::BelowHpRatio(0.20), &state, &legal).is_none());
    }

    #[test]
    fn retreat_policy_returns_none_when_no_retreat_in_legal_list() {
        let mut state = State::default();
        state.in_play_pokemon[0][0] = Some(mk_played(CardId::A1001Bulbasaur, 10));
        let legal = vec![end_turn(0)];
        assert!(pick_retreat(&RetreatPolicy::BelowHp(20), &state, &legal).is_none());
    }

    // --- AttackSelector ---

    #[test]
    fn attack_selector_first_picks_first_attack() {
        let mut state = State::default();
        state.in_play_pokemon[0][0] = Some(mk_played(CardId::A1004VenusaurEx, 100));
        let legal = vec![attack(0, 0), attack(0, 1)];
        let pick = pick_attack(&AttackSelector::First, &state, &legal).unwrap();
        assert_eq!(pick, legal[0]);
    }

    #[test]
    fn attack_selector_highest_damage_picks_by_damage() {
        let mut state = State::default();
        // Venusaur ex attacks: idx 0 = Razor Leaf (60), idx 1 = Giant Bloom (100)
        state.in_play_pokemon[0][0] = Some(mk_played(CardId::A1004VenusaurEx, 100));
        let legal = vec![attack(0, 0), attack(0, 1)];
        let pick = pick_attack(&AttackSelector::HighestDamage, &state, &legal).unwrap();
        assert_eq!(pick, legal[1]);
    }

    #[test]
    fn attack_selector_returns_none_when_no_attack_in_legal_list() {
        let state = State::default();
        let legal = vec![end_turn(0)];
        assert!(pick_attack(&AttackSelector::HighestDamage, &state, &legal).is_none());
    }

    // --- SupporterGreed ---

    #[test]
    fn supporter_greed_always_picks_first_supporter_when_legal() {
        let profs_research = get_card_by_enum(CardId::PA007ProfessorsResearch).as_trainer();
        let legal = vec![
            attach_turn(0),
            mk_action(
                0,
                SimpleAction::Play {
                    trainer_card: profs_research,
                },
                false,
            ),
        ];
        let pick = pick_supporter(&SupporterGreed::Always, &legal).unwrap();
        if let SimpleAction::Play { trainer_card } = &pick.action {
            assert_eq!(trainer_card.trainer_card_type, TrainerType::Supporter);
        } else {
            panic!("expected Play action");
        }
    }

    #[test]
    fn supporter_greed_never_returns_none() {
        let profs_research = get_card_by_enum(CardId::PA007ProfessorsResearch).as_trainer();
        let legal = vec![mk_action(
            0,
            SimpleAction::Play {
                trainer_card: profs_research,
            },
            false,
        )];
        assert!(pick_supporter(&SupporterGreed::Never, &legal).is_none());
    }

    #[test]
    fn supporter_greed_returns_none_when_no_supporter_in_legal_list() {
        let legal = vec![attach_turn(0)];
        assert!(pick_supporter(&SupporterGreed::Always, &legal).is_none());
    }

    // --- BenchPriority ---

    #[test]
    fn bench_priority_first_picks_first_place() {
        let bulbasaur = get_card_by_enum(CardId::A1001Bulbasaur);
        let exeggcute = get_card_by_enum(CardId::A1021Exeggcute);
        let legal = vec![place(0, bulbasaur, 1), place(0, exeggcute, 2)];
        let pick = pick_place(&BenchPriority::First, &legal).unwrap();
        assert_eq!(pick, legal[0]);
    }

    #[test]
    fn bench_priority_highest_hp_picks_by_printed_hp() {
        // Bulbasaur 70hp vs Exeggcute 60hp — Bulbasaur should win
        let bulbasaur = get_card_by_enum(CardId::A1001Bulbasaur);
        let exeggcute = get_card_by_enum(CardId::A1021Exeggcute);
        let legal = vec![
            place(0, exeggcute.clone(), 1),
            place(0, bulbasaur.clone(), 2),
        ];
        let pick = pick_place(&BenchPriority::HighestHp, &legal).unwrap();
        match &pick.action {
            SimpleAction::Place(card, _) => assert_eq!(card, &bulbasaur),
            _ => panic!("expected Place action"),
        }
    }

    #[test]
    fn bench_priority_returns_none_when_no_place_in_legal_list() {
        let legal = vec![end_turn(0)];
        assert!(pick_place(&BenchPriority::HighestHp, &legal).is_none());
    }

    // --- apply_heuristic routing ---

    #[test]
    #[should_panic(expected = "empty legal list")]
    fn apply_heuristic_panics_on_empty_legal() {
        let state = State::default();
        let config = HeuristicConfig::default();
        apply_heuristic(&config, &state, &[]);
    }

    #[test]
    fn apply_heuristic_always_returns_an_element_of_legal() {
        // Property: whatever the configuration, the returned action is one of
        // the provided legal actions.
        let mut state = State::default();
        state.in_play_pokemon[0][0] = Some(mk_played(CardId::A1001Bulbasaur, 70));
        let profs_research = get_card_by_enum(CardId::PA007ProfessorsResearch).as_trainer();
        let bulbasaur = get_card_by_enum(CardId::A1001Bulbasaur);
        let legal = vec![
            end_turn(0),
            attach_turn(0),
            attack(0, 0),
            place(0, bulbasaur, 1),
            mk_action(
                0,
                SimpleAction::Play {
                    trainer_card: profs_research,
                },
                false,
            ),
        ];
        let config = HeuristicConfig::default();
        let pick = apply_heuristic(&config, &state, &legal);
        assert!(legal.contains(&pick));
    }

    #[test]
    fn apply_heuristic_prefers_setup_over_attack() {
        // With Attach and Attack both available, the heuristic attaches first
        // (builds energy before swinging).
        let mut state = State::default();
        state.in_play_pokemon[0][0] = Some(mk_played(CardId::A1001Bulbasaur, 70));
        let legal = vec![attach_turn(0), attack(0, 0), end_turn(0)];
        let config = HeuristicConfig::default();
        let pick = apply_heuristic(&config, &state, &legal);
        assert!(matches!(
            pick.action,
            SimpleAction::Attach {
                is_turn_energy: true,
                ..
            }
        ));
    }

    #[test]
    fn apply_heuristic_routes_stack_activate_through_target_priority() {
        let state = state_with_bench(
            0,
            vec![
                (1, mk_played(CardId::A1001Bulbasaur, 10)),
                (2, mk_played(CardId::A1002Ivysaur, 90)),
            ],
        );
        let legal = vec![activate(0, 1), activate(0, 2)];
        let mut config = HeuristicConfig::default();
        config.target_priority = TargetPriority::LowestHp;
        let pick = apply_heuristic(&config, &state, &legal);
        assert_eq!(pick, legal[0]); // lowest HP — Bulbasaur
    }

    #[test]
    fn apply_heuristic_ends_turn_when_nothing_else_is_legal() {
        let state = State::default();
        let legal = vec![end_turn(0)];
        let pick = apply_heuristic(&HeuristicConfig::default(), &state, &legal);
        assert!(matches!(pick.action, SimpleAction::EndTurn));
    }
}
