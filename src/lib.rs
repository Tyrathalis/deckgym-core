pub mod actions;
pub mod card_ids;
pub mod card_logic;
pub mod card_validation;
pub mod combinatorics;
pub mod data_exporter;
pub mod database;
pub mod deck;
pub mod effects;
pub mod example_utils;
pub mod game;
pub mod gameplay_stats_collector;
pub mod heuristics;
mod hooks;
pub mod models;
pub mod move_generation;
pub mod optimize;
pub mod players;
pub mod shaping;
pub mod simulate;
pub mod simulation_event_handler;
pub mod stadiums;
pub mod state;
pub mod temp_deck;
pub mod tools;

#[cfg(any(test, feature = "test-utils"))]
pub mod test_support;

pub use deck::Deck;
pub use game::Game;
pub use hooks::to_playable_card;
pub use move_generation::generate_possible_trainer_actions;
#[cfg(feature = "cli")]
pub use optimize::cli_optimize;
pub use optimize::{
    optimize, optimize_with_configs, EnemyDeckConfig, OptimizationConfig, ParallelConfig,
    SimulationConfig,
};
#[cfg(feature = "cli")]
pub use simulate::simulate;
pub use simulate::{Simulation, SimulationCallbacks};
pub use simulation_event_handler::ComputedStats;
pub use state::State;

#[cfg(feature = "python")]
pub mod python_bindings;

#[cfg(feature = "tui")]
pub mod tui;

#[cfg(feature = "wasm")]
pub mod wasm_bindings;

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyModule;

#[cfg(feature = "python")]
#[pymodule]
fn deckgym(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    python_bindings::deckgym(py, m)
}
