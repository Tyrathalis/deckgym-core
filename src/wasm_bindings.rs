use rand::{rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use crate::{
    actions::{apply_action, Action},
    deck::Deck,
    heuristics::{apply_heuristic, HeuristicConfig},
    shaping::board_value_for_player,
    state::{GameOutcome, State},
};

/// Build the serde-wasm-bindgen serializer used by every binding. We enable
/// `serialize_missing_as_null` so `Option::None` crosses the boundary as
/// `null` rather than `undefined` — the TS layer's types declare
/// `GameOutcome | null`, and game-runner termination checks rely on `=== null`.
fn js_serializer() -> serde_wasm_bindgen::Serializer {
    serde_wasm_bindgen::Serializer::new().serialize_missing_as_null(true)
}

fn to_js<T: Serialize>(value: &T) -> Result<JsValue, JsError> {
    value
        .serialize(&js_serializer())
        .map_err(|e| JsError::new(&e.to_string()))
}

// =============================================================================
// Existing bindings (Phase 0)
// =============================================================================

/// Result wrapper for WASM API calls
#[derive(Serialize, Deserialize)]
struct CreateGameResult {
    state: State,
    /// Which player goes first (0 or 1)
    current_player: usize,
}

/// Result wrapper for legal actions query
#[derive(Serialize, Deserialize)]
struct LegalActionsResult {
    /// The player who must act
    actor: usize,
    actions: Vec<Action>,
}

/// Result wrapper for apply_action
#[derive(Serialize, Deserialize)]
struct ApplyActionResult {
    state: State,
    game_over: bool,
    winner: Option<GameOutcome>,
}

/// Create a new game from two deck definitions and a seed.
/// Returns the initial game state as a serialized JsValue.
#[wasm_bindgen]
pub fn create_game(deck_a_json: &str, deck_b_json: &str, seed: u64) -> Result<JsValue, JsError> {
    let deck_a: Deck =
        serde_json::from_str(deck_a_json).map_err(|e| JsError::new(&format!("deck_a: {e}")))?;
    let deck_b: Deck =
        serde_json::from_str(deck_b_json).map_err(|e| JsError::new(&format!("deck_b: {e}")))?;

    let mut rng = StdRng::seed_from_u64(seed);
    let state = State::initialize(&deck_a, &deck_b, &mut rng);

    let result = CreateGameResult {
        current_player: state.current_player,
        state,
    };
    to_js(&result)
}

/// Get the legal actions for the current game state.
/// Returns the acting player index and list of legal actions.
#[wasm_bindgen]
pub fn get_legal_actions(state_json: &str) -> Result<JsValue, JsError> {
    let state: State =
        serde_json::from_str(state_json).map_err(|e| JsError::new(&format!("state: {e}")))?;

    let (actor, actions) = state.generate_possible_actions();

    let result = LegalActionsResult { actor, actions };
    to_js(&result)
}

/// Apply an action to the game state.
/// Returns the new state, whether the game is over, and the winner (if any).
#[wasm_bindgen]
pub fn apply_action_wasm(
    state_json: &str,
    action_json: &str,
    seed: u64,
) -> Result<JsValue, JsError> {
    let mut state: State =
        serde_json::from_str(state_json).map_err(|e| JsError::new(&format!("state: {e}")))?;
    let action: Action =
        serde_json::from_str(action_json).map_err(|e| JsError::new(&format!("action: {e}")))?;

    let mut rng = StdRng::seed_from_u64(seed);
    apply_action(&mut rng, &mut state, &action);

    let result = ApplyActionResult {
        game_over: state.is_game_over(),
        winner: state.winner,
        state,
    };
    to_js(&result)
}

/// Get a human-readable text display of the game state.
#[wasm_bindgen]
pub fn get_state_display(state_json: &str) -> Result<String, JsError> {
    let state: State =
        serde_json::from_str(state_json).map_err(|e| JsError::new(&format!("state: {e}")))?;
    Ok(state.debug_string())
}

/// Parse a deck from the engine's text list format (`Energy: ...` + lines of
/// `<count> <set> <number>`). Returns a serialized `Deck` ready to hand back
/// to `create_game`.
#[wasm_bindgen]
pub fn parse_deck_from_string(contents: &str) -> Result<JsValue, JsError> {
    let deck = Deck::from_string(contents).map_err(|e| JsError::new(&e))?;
    to_js(&deck)
}

// =============================================================================
// Phase 1 bindings: semi-MDP step + 1-step counterfactual enumerator
// =============================================================================

/// Signals computed over the delta between the state passed into a step
/// primitive and the state returned. TS computes the scalar reward from
/// `ShapingSignals × RewardWeights`; the engine does not know about weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ShapingSignals {
    /// Prizes the agent gained during this step (`post - pre`).
    agent_prize_delta: i32,
    /// Prizes the opponent gained during this step.
    opponent_prize_delta: i32,
    /// Total damage dealt to opponent's Pokémon during this step, counting
    /// HP-from-KOed-pokemon as well as additional damage counters on
    /// still-alive opponent Pokémon.
    agent_damage_dealt: u32,
    /// Same, in the other direction.
    agent_damage_taken: u32,
    /// Change in the sum of the agent's Pokémon's remaining HP (post - pre).
    /// Negative when the agent took net damage; positive for net healing.
    agent_hp_preserved: i32,
    /// Count of turn-energy `Attach` actions the agent executed during this
    /// step. Non-turn-energy attaches (from card effects) are excluded.
    agent_energy_attached: u32,
    /// Phase 2 Step 0j — HP-remaining potential shaping. Change in the
    /// prize-weighted "KO-progress" sum across opponent Pokémon between
    /// `pre` and `post`. See `crate::shaping::board_value_for_player` for
    /// the V formula; the discard-pile term there ensures V is
    /// monotonically non-decreasing under damage and KOs, so
    /// `agent_v_opp_delta = V(post, opp) − V(pre, opp)` is positive when
    /// the agent pushed opponent Pokémon closer to KO *or* KO'd them
    /// (including bench-promote cases). Weights damage on a 20-HP basic
    /// more than the same damage on a 180-HP EX. Replaces the damage-
    /// counter shaping as the primary channel in Step 0j's balanced
    /// preset (rationale: `docs/reward-signal-catalog.md` §HP-remaining
    /// potential shaping).
    agent_v_opp_delta: f32,
    /// Symmetric self-damage channel: `V(post, agent) − V(pre, agent)`.
    /// Positive when the agent's own Pokémon took damage or were KO'd;
    /// the reward weight is typically negative so self-damage is a
    /// penalty. Captures EX-weighted damage taken without a separate
    /// multiplier.
    agent_v_self_delta: f32,
    /// Post-state prize totals. Exposed so the tie-reward curve
    /// (`tanh((agent_prize − opp_prize)/2) × tieBase`) can be computed at
    /// terminal without needing the full post-state in the reward path.
    /// Mirrors `post.points[agent]` / `post.points[opp]`.
    agent_prize_total: u32,
    opponent_prize_total: u32,
    /// Every action applied between input and return, in chronological
    /// order. Starts with the agent's action (for `step_until_agent_decision`)
    /// or with the opponent's setup actions (for `init_match`). For logging
    /// and compact-replay reconstruction.
    intermediate_actions: Vec<Action>,
}

/// Return shape of `step_until_agent_decision` / `init_match`.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StepResult {
    /// The state at the next agent decision, or the terminal state.
    state: State,
    shaping_signals: ShapingSignals,
    /// True iff the game ended during this step.
    done: bool,
    /// Set whenever `done` is true.
    winner: Option<GameOutcome>,
}

/// One branch in the 1-step counterfactual enumeration.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EnumerationBranch {
    /// The agent action this branch explored.
    action: Action,
    result: StepResult,
}

/// Return shape of `enumerate_one_step`.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EnumerationResult {
    branches: Vec<EnumerationBranch>,
}

/// Cap on follow-on actions applied inside a single step before we return
/// control to the agent. Generous — a setup sequence on the opponent's first
/// turn can chain dozens of forced stack actions. The cap exists to stop
/// runaway heuristic loops, not to bound expected play.
const MAX_FOLLOWUP_ACTIONS: usize = 500;

/// Initialize a match and advance the game to the agent's first decision.
///
/// If `agent_player == 0`, `deck_agent_text` becomes player 0's deck and
/// `deck_opponent_text` becomes player 1's deck (and vice-versa for
/// `agent_player == 1`). During setup and initial forced sequences the
/// opponent's actions are driven by `opponent_config`; control returns to the
/// caller the first time the agent must decide, or at the terminal state if
/// the match somehow ended before the agent ever acted.
#[wasm_bindgen]
pub fn init_match(
    deck_agent_text: &str,
    deck_opponent_text: &str,
    opponent_config_json: &str,
    agent_player: usize,
    seed: u64,
) -> Result<JsValue, JsError> {
    if agent_player > 1 {
        return Err(JsError::new("agent_player must be 0 or 1"));
    }
    let deck_agent = Deck::from_string(deck_agent_text)
        .map_err(|e| JsError::new(&format!("deck_agent: {e}")))?;
    let deck_opponent = Deck::from_string(deck_opponent_text)
        .map_err(|e| JsError::new(&format!("deck_opponent: {e}")))?;
    let opponent_config: HeuristicConfig = serde_json::from_str(opponent_config_json)
        .map_err(|e| JsError::new(&format!("opponent_config: {e}")))?;

    let mut rng = StdRng::seed_from_u64(seed);
    let (deck_a, deck_b) = if agent_player == 0 {
        (&deck_agent, &deck_opponent)
    } else {
        (&deck_opponent, &deck_agent)
    };

    let pre = State::initialize(deck_a, deck_b, &mut rng);
    let mut state = pre.clone();
    let mut actions_applied: Vec<Action> = Vec::new();

    advance_until_agent_decision(
        &mut state,
        &mut rng,
        agent_player,
        &opponent_config,
        &mut actions_applied,
    )?;

    let shaping_signals = compute_shaping_signals(&pre, &state, agent_player, actions_applied);
    let result = StepResult {
        done: state.winner.is_some(),
        winner: state.winner,
        state,
        shaping_signals,
    };
    to_js(&result)
}

/// Apply the agent's action, then let the opponent (and any forced stack
/// sequences) resolve until the legal-action list is once again owned by the
/// agent or the game ends.
#[wasm_bindgen]
pub fn step_until_agent_decision(
    state_json: &str,
    agent_action_json: &str,
    opponent_config_json: &str,
    agent_player: usize,
    seed: u64,
) -> Result<JsValue, JsError> {
    if agent_player > 1 {
        return Err(JsError::new("agent_player must be 0 or 1"));
    }
    let pre: State =
        serde_json::from_str(state_json).map_err(|e| JsError::new(&format!("state: {e}")))?;
    let agent_action: Action = serde_json::from_str(agent_action_json)
        .map_err(|e| JsError::new(&format!("agent_action: {e}")))?;
    let opponent_config: HeuristicConfig = serde_json::from_str(opponent_config_json)
        .map_err(|e| JsError::new(&format!("opponent_config: {e}")))?;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut state = pre.clone();
    let mut actions_applied: Vec<Action> = Vec::with_capacity(8);

    apply_action(&mut rng, &mut state, &agent_action);
    actions_applied.push(agent_action);

    advance_until_agent_decision(
        &mut state,
        &mut rng,
        agent_player,
        &opponent_config,
        &mut actions_applied,
    )?;

    let shaping_signals = compute_shaping_signals(&pre, &state, agent_player, actions_applied);
    let result = StepResult {
        done: state.winner.is_some(),
        winner: state.winner,
        state,
        shaping_signals,
    };
    to_js(&result)
}

/// For each legal agent action at the current decision point, compute the
/// `StepResult` of applying that action followed by opponent play until the
/// next agent decision (or terminal). One WASM crossing per decision point.
///
/// Each branch uses a deterministic sub-seed so results are reproducible.
#[wasm_bindgen]
pub fn enumerate_one_step(
    state_json: &str,
    opponent_config_json: &str,
    agent_player: usize,
    seed: u64,
) -> Result<JsValue, JsError> {
    if agent_player > 1 {
        return Err(JsError::new("agent_player must be 0 or 1"));
    }
    let pre: State =
        serde_json::from_str(state_json).map_err(|e| JsError::new(&format!("state: {e}")))?;
    let opponent_config: HeuristicConfig = serde_json::from_str(opponent_config_json)
        .map_err(|e| JsError::new(&format!("opponent_config: {e}")))?;

    let (actor, possible) = pre.generate_possible_actions();
    if actor != agent_player {
        return Err(JsError::new(
            "enumerate_one_step: state is not at an agent decision point",
        ));
    }
    if possible.is_empty() {
        return Err(JsError::new(
            "enumerate_one_step: agent has no legal actions, but game is not over",
        ));
    }

    let mut branches = Vec::with_capacity(possible.len());
    for (i, action) in possible.iter().enumerate() {
        let branch_seed = mix_seed(seed, i);
        let mut rng = StdRng::seed_from_u64(branch_seed);
        let mut state = pre.clone();
        let mut actions_applied: Vec<Action> = Vec::with_capacity(8);

        apply_action(&mut rng, &mut state, action);
        actions_applied.push(action.clone());

        advance_until_agent_decision(
            &mut state,
            &mut rng,
            agent_player,
            &opponent_config,
            &mut actions_applied,
        )?;

        let shaping_signals = compute_shaping_signals(&pre, &state, agent_player, actions_applied);
        branches.push(EnumerationBranch {
            action: action.clone(),
            result: StepResult {
                done: state.winner.is_some(),
                winner: state.winner,
                state,
                shaping_signals,
            },
        });
    }

    to_js(&EnumerationResult { branches })
}

// =============================================================================
// Internal helpers
// =============================================================================

/// Drive the heuristic until control returns to the agent or the game ends.
/// Returns an error if the cap is exceeded — a guardrail against a heuristic
/// routing bug that could otherwise produce an infinite loop.
fn advance_until_agent_decision(
    state: &mut State,
    rng: &mut StdRng,
    agent_player: usize,
    opponent_config: &HeuristicConfig,
    actions_applied: &mut Vec<Action>,
) -> Result<(), JsError> {
    for _ in 0..MAX_FOLLOWUP_ACTIONS {
        if state.winner.is_some() {
            return Ok(());
        }
        let (actor, possible) = state.generate_possible_actions();
        if actor == agent_player {
            return Ok(());
        }
        if possible.is_empty() {
            return Err(JsError::new(
                "advance_until_agent_decision: no legal actions for non-agent player, but game is not over",
            ));
        }
        let action = apply_heuristic(opponent_config, state, &possible);
        apply_action(rng, state, &action);
        actions_applied.push(action);
    }
    Err(JsError::new(&format!(
        "advance_until_agent_decision: exceeded {MAX_FOLLOWUP_ACTIONS} follow-up actions without returning to the agent"
    )))
}

/// Compute shaping signals over the pre → post delta. Precise accounting of
/// damage-across-heals is not required — these are training shaping signals,
/// so an approximate diff with consistent semantics on both on-policy and
/// counterfactual branches is what matters.
fn compute_shaping_signals(
    pre: &State,
    post: &State,
    agent: usize,
    intermediate_actions: Vec<Action>,
) -> ShapingSignals {
    let opp = 1 - agent;

    let agent_prize_delta = (post.points[agent] as i32) - (pre.points[agent] as i32);
    let opponent_prize_delta = (post.points[opp] as i32) - (pre.points[opp] as i32);

    let agent_damage_dealt = total_damage_done_to(pre, post, opp);
    let agent_damage_taken = total_damage_done_to(pre, post, agent);

    let agent_hp_preserved =
        total_remaining_hp(post, agent) as i32 - total_remaining_hp(pre, agent) as i32;

    let agent_energy_attached = intermediate_actions
        .iter()
        .filter(|a| {
            a.actor == agent
                && matches!(
                    a.action,
                    crate::actions::SimpleAction::Attach {
                        is_turn_energy: true,
                        ..
                    }
                )
        })
        .count() as u32;

    // Phase 2 Step 0j — HP-remaining potential shaping.
    let agent_v_opp_delta = board_value_for_player(post, opp) - board_value_for_player(pre, opp);
    let agent_v_self_delta =
        board_value_for_player(post, agent) - board_value_for_player(pre, agent);

    ShapingSignals {
        agent_prize_delta,
        opponent_prize_delta,
        agent_damage_dealt,
        agent_damage_taken,
        agent_hp_preserved,
        agent_energy_attached,
        agent_v_opp_delta,
        agent_v_self_delta,
        agent_prize_total: post.points[agent] as u32,
        opponent_prize_total: post.points[opp] as u32,
        intermediate_actions,
    }
}

/// Damage done to `target_player`'s Pokémon between `pre` and `post`.
/// Counts both additional damage counters on still-alive Pokémon and full
/// effective-HP credit for any Pokémon that was discarded (KO'd).
fn total_damage_done_to(pre: &State, post: &State, target_player: usize) -> u32 {
    let mut damage: u32 = 0;
    for i in 0..4 {
        match (
            &pre.in_play_pokemon[target_player][i],
            &post.in_play_pokemon[target_player][i],
        ) {
            (Some(pre_p), Some(post_p)) => {
                // Additional damage counters accumulated.
                damage = damage.saturating_add(
                    post_p
                        .get_damage_counters()
                        .saturating_sub(pre_p.get_damage_counters()),
                );
            }
            (Some(pre_p), None) => {
                // KO'd and discarded between pre and post. Credit the remaining
                // HP this Pokémon had when the step started; damage counters
                // already on it before the step don't count (they belong to
                // an earlier step).
                damage = damage.saturating_add(pre_p.get_remaining_hp());
            }
            (None, Some(_)) | (None, None) => {
                // A new Pokémon being placed (or nothing) attributes no damage.
            }
        }
    }
    damage
}

fn total_remaining_hp(state: &State, player: usize) -> u32 {
    state.in_play_pokemon[player]
        .iter()
        .filter_map(|x| x.as_ref().map(|p| p.get_remaining_hp()))
        .sum()
}

/// Deterministic per-branch sub-seed derivation. Same `(seed, i)` → same
/// sub-seed, across runs and platforms.
fn mix_seed(seed: u64, i: usize) -> u64 {
    let mut h = seed.wrapping_mul(0x9e3779b97f4a7c15);
    h ^= (i as u64).wrapping_mul(0xbf58476d1ce4e5b9);
    h ^= h >> 30;
    h.wrapping_mul(0x94d049bb133111eb)
}
