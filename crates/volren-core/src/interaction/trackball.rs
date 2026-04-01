//! Trackball interaction style for 3D volume exploration.
//!
//! Implements an arcball-style camera:
//! - **Left drag**   → orbit (arcball rotation)
//! - **Right drag**  → dolly (zoom in/out)
//! - **Middle drag** → pan
//! - **Scroll**      → dolly
//! - **`r` key**     → reset to initial camera (requires consumer to set)
//!
//! # VTK Equivalent
//! `vtkInteractorStyleTrackballCamera`

use super::{
    events::{InteractionContext, InteractionResult, MouseEventKind},
    InteractionStyle, KeyEvent, MouseButton, MouseEvent,
};
use crate::camera::Camera;

/// State of active drag.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum DragState {
    #[default]
    None,
    Orbiting,
    Dollying,
    Panning,
}

/// Trackball (arcball) camera interaction style.
///
/// **Sensitivity** values control the mapping from pixel deltas to angles/units:
/// - `orbit_sensitivity` — radians per pixel (default 0.005)
/// - `pan_sensitivity`   — world units per pixel (default 0.001 × distance)
/// - `zoom_sensitivity`  — factor per scroll tick (default 0.1)
#[derive(Debug)]
pub struct TrackballStyle {
    drag: DragState,
    last_pos: (f64, f64),
    /// Radians per pixel for orbit.
    pub orbit_sensitivity: f64,
    /// Scroll zoom: factor per scroll unit (subtracted from 1.0).
    pub zoom_sensitivity: f64,
}

impl TrackballStyle {
    /// Create with default sensitivities.
    #[must_use]
    pub fn new() -> Self {
        Self {
            drag: DragState::None,
            last_pos: (0.0, 0.0),
            orbit_sensitivity: 0.005,
            zoom_sensitivity: 0.1,
        }
    }
}

impl Default for TrackballStyle {
    fn default() -> Self {
        Self::new()
    }
}

impl InteractionStyle for TrackballStyle {
    fn on_mouse_event(
        &mut self,
        event: &MouseEvent,
        _ctx: &InteractionContext,
        camera: &mut Camera,
    ) -> InteractionResult {
        match event.kind {
            MouseEventKind::Press(button) => {
                self.last_pos = event.position;
                self.drag = match button {
                    MouseButton::Left => DragState::Orbiting,
                    MouseButton::Right => DragState::Dollying,
                    MouseButton::Middle => DragState::Panning,
                };
                InteractionResult::nothing()
            }

            MouseEventKind::Release(_) => {
                self.drag = DragState::None;
                InteractionResult::nothing()
            }

            MouseEventKind::Move => {
                let dx = event.position.0 - self.last_pos.0;
                let dy = event.position.1 - self.last_pos.1;
                self.last_pos = event.position;

                if dx == 0.0 && dy == 0.0 {
                    return InteractionResult::nothing();
                }

                match self.drag {
                    DragState::None => return InteractionResult::nothing(),

                    DragState::Orbiting => {
                        let angle_h = dx * self.orbit_sensitivity;
                        let angle_v = dy * self.orbit_sensitivity;
                        camera.orbit(angle_h, angle_v);
                    }

                    DragState::Dollying => {
                        let delta = dy * camera.distance() * 0.01;
                        camera.dolly(delta);
                    }

                    DragState::Panning => {
                        let scale = camera.distance() * 0.001;
                        let right = camera.right();
                        let up = camera.view_up_ortho();
                        camera.pan(-right * dx * scale + up * dy * scale);
                    }
                }

                InteractionResult::camera_only()
            }

            MouseEventKind::Scroll(delta) => {
                let factor = if delta > 0.0 {
                    1.0 - self.zoom_sensitivity * delta.abs()
                } else {
                    1.0 + self.zoom_sensitivity * delta.abs()
                };
                camera.zoom(factor.max(0.01));
                InteractionResult::camera_only()
            }
        }
    }

    fn on_key_event(
        &mut self,
        _event: &KeyEvent,
        _ctx: &InteractionContext,
        _camera: &mut Camera,
    ) -> InteractionResult {
        InteractionResult::nothing()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interaction::Modifiers;
    use approx::assert_abs_diff_eq;
    use glam::DVec3;

    fn default_camera() -> Camera {
        Camera::new(DVec3::new(0.0, 0.0, 10.0), DVec3::ZERO, DVec3::Y)
    }

    fn ctx() -> InteractionContext {
        InteractionContext { viewport_width: 800.0, viewport_height: 600.0, volume_bounds: None }
    }

    fn mouse(pos: (f64, f64), kind: MouseEventKind) -> MouseEvent {
        MouseEvent { position: pos, kind, modifiers: Modifiers::default() }
    }

    #[test]
    fn press_and_move_orbits() {
        let mut style = TrackballStyle::new();
        let mut cam = default_camera();
        let d0 = cam.distance();

        let r1 = style.on_mouse_event(
            &mouse((0.0, 0.0), MouseEventKind::Press(MouseButton::Left)),
            &ctx(),
            &mut cam,
        );
        assert!(!r1.camera_changed);

        let r2 = style.on_mouse_event(
            &mouse((50.0, 0.0), MouseEventKind::Move),
            &ctx(),
            &mut cam,
        );
        assert!(r2.camera_changed);
        // Distance preserved after orbit
        assert_abs_diff_eq!(cam.distance(), d0, epsilon = 1e-6);
    }

    #[test]
    fn scroll_zooms_camera() {
        let mut style = TrackballStyle::new();
        let mut cam = default_camera();
        let d0 = cam.distance();

        style.on_mouse_event(
            &mouse((400.0, 300.0), MouseEventKind::Scroll(1.0)),
            &ctx(),
            &mut cam,
        );
        assert!(cam.distance() < d0, "scroll should zoom in");
    }

    #[test]
    fn no_drag_move_does_nothing() {
        let mut style = TrackballStyle::new();
        let mut cam = default_camera();
        let pos0 = cam.position();

        style.on_mouse_event(
            &mouse((100.0, 100.0), MouseEventKind::Move),
            &ctx(),
            &mut cam,
        );
        assert_abs_diff_eq!(cam.position().x, pos0.x, epsilon = 1e-10);
    }
}
