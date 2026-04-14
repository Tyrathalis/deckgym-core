pub(crate) mod abilities;
mod apply_abilities_action;
mod apply_action;
mod apply_action_helpers;
mod apply_attack_action;
mod apply_trainer_action;
pub(crate) mod attack_helpers;
mod attacks;
mod effect_ability_mechanic_map;
mod effect_mechanic_map;
mod mutations;
mod outcomes;
mod shared_mutations;
mod types;

pub(crate) use apply_action::apply_action;
/// Test/wasm-binding re-export of `apply_action`. The core `apply_action` is
/// `pub(crate)` to keep the normal public API tight (`Game::play` is the
/// entry point for orchestrated play), but Ditto Trainer's heuristic
/// integration tests and wasm bindings drive the engine one action at a
/// time, so we need it exposed when either gate applies.
#[cfg(any(test, feature = "test-utils", feature = "wasm"))]
pub use apply_action::apply_action as apply_action_public;
pub(crate) use apply_action::apply_evolve;
pub(crate) use apply_action::apply_place_card;
pub(crate) use apply_action::forecast_action;
pub(crate) use apply_action_helpers::handle_damage;
pub(crate) use apply_action_helpers::handle_damage_only;
pub(crate) use apply_action_helpers::handle_knockouts;
pub use apply_trainer_action::may_effect;
pub use effect_ability_mechanic_map::ability_mechanic_from_effect;
pub(crate) use effect_ability_mechanic_map::get_ability_mechanic;
pub use effect_ability_mechanic_map::has_ability_mechanic;
pub use effect_ability_mechanic_map::EFFECT_ABILITY_MECHANIC_MAP;
pub use effect_mechanic_map::EFFECT_MECHANIC_MAP;
pub use types::Action;
pub use types::SimpleAction;
