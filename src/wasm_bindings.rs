use rand::{rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use crate::{
    actions::{apply_action, Action},
    deck::Deck,
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
