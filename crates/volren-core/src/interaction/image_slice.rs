//! Image-slice interaction style for 2D MPR viewing.
//!
//! Controls:
//! - **Left drag horizontal** → adjust window width
//! - **Left drag vertical**   → adjust window center (level)
//! - **Middle drag**          → pan
//! - **Right drag / Scroll**  → scroll through slices (Z translation)
//!
//! # VTK Equivalent
//! `vtkInteractorStyleImage`

use super::{
    events::{InteractionContext, InteractionResult, MouseEventKind},
    InteractionStyle, KeyEvent, MouseButton, MouseEvent,
};
use crate::{camera::Camera, window_level::WindowLevel};

/// Drag action in progress.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum DragState {
    #[default]
    None,
    WindowLevel,
    Panning,
    Slicing,
}

/// Image-slice interaction style.
///
/// Maintains an internal [`WindowLevel`] that the consumer can read after each
/// event via [`ImageSliceStyle::window_level`].
///
/// Slice scrolling is communicated through a `slice_delta` value that the
/// consumer should read and apply to its [`crate::reslice::SlicePlane`].
#[derive(Debug)]
pub struct ImageSliceStyle {
    drag: DragState,
    last_pos: (f64, f64),
    window_level: WindowLevel,
    slice_delta: f64,
    /// Pixels per unit of window width change (default 4.0).
    pub window_sensitivity: f64,
    /// Pixels per unit of window center change (default 2.0).
    pub level_sensitivity: f64,
    /// World units per scroll tick for slicing (default 1.0).
    pub slice_scroll_step: f64,
}

impl ImageSliceStyle {
    /// Create with default sensitivities and the given initial window/level.
    #[must_use]
    pub fn new(window_level: WindowLevel) -> Self {
        Self {
            drag: DragState::None,
            last_pos: (0.0, 0.0),
            window_level,
            slice_delta: 0.0,
            window_sensitivity: 4.0,
            level_sensitivity: 2.0,
            slice_scroll_step: 1.0,
        }
    }

    /// The current window/level (updated by drag events).
    #[must_use]
    pub fn window_level(&self) -> WindowLevel {
        self.window_level
    }

    /// Consume and return the accumulated slice translation since the last call.
    ///
    /// The consumer should apply this to its `SlicePlane::offset_along_normal`.
    pub fn take_slice_delta(&mut self) -> f64 {
        let d = self.slice_delta;
        self.slice_delta = 0.0;
        d
    }
}

impl InteractionStyle for ImageSliceStyle {
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
                    MouseButton::Left => DragState::WindowLevel,
                    MouseButton::Middle => DragState::Panning,
                    MouseButton::Right => DragState::Slicing,
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
                    DragState::None => InteractionResult::nothing(),

                    DragState::WindowLevel => {
                        self.window_level.width =
                            (self.window_level.width + dx * self.window_sensitivity).max(1.0);
                        self.window_level.center += dy * self.level_sensitivity;
                        InteractionResult::window_level_only()
                    }

                    DragState::Panning => {
                        let scale = camera.distance() * 0.001;
                        let right = camera.right();
                        let up = camera.view_up_ortho();
                        camera.pan(-right * dx * scale + up * dy * scale);
                        InteractionResult::camera_only()
                    }

                    DragState::Slicing => {
                        self.slice_delta += dy * self.slice_scroll_step * 0.1;
                        InteractionResult::slice_only()
                    }
                }
            }

            MouseEventKind::Scroll(delta) => {
                self.slice_delta += delta * self.slice_scroll_step;
                InteractionResult::slice_only()
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
    use crate::window_level::presets;
    use glam::DVec3;

    fn default_camera() -> Camera {
        Camera::new(DVec3::new(0.0, 0.0, 10.0), DVec3::ZERO, DVec3::Y)
    }

    fn ctx() -> InteractionContext {
        InteractionContext {
            viewport_width: 800.0,
            viewport_height: 600.0,
            volume_bounds: None,
        }
    }

    fn mouse(pos: (f64, f64), kind: MouseEventKind) -> MouseEvent {
        MouseEvent {
            position: pos,
            kind,
            modifiers: Modifiers::default(),
        }
    }

    #[test]
    fn left_drag_horizontal_changes_window_width() {
        let mut style = ImageSliceStyle::new(presets::SOFT_TISSUE);
        let mut cam = default_camera();
        let w0 = style.window_level().width;

        style.on_mouse_event(
            &mouse((0.0, 0.0), MouseEventKind::Press(MouseButton::Left)),
            &ctx(),
            &mut cam,
        );
        let r = style.on_mouse_event(&mouse((50.0, 0.0), MouseEventKind::Move), &ctx(), &mut cam);

        assert!(r.window_level_changed);
        assert!(style.window_level().width > w0, "width should increase");
    }

    #[test]
    fn left_drag_vertical_changes_center() {
        let mut style = ImageSliceStyle::new(presets::SOFT_TISSUE);
        let mut cam = default_camera();
        let c0 = style.window_level().center;

        style.on_mouse_event(
            &mouse((0.0, 0.0), MouseEventKind::Press(MouseButton::Left)),
            &ctx(),
            &mut cam,
        );
        style.on_mouse_event(&mouse((0.0, 30.0), MouseEventKind::Move), &ctx(), &mut cam);

        assert_ne!(style.window_level().center, c0, "center should change");
    }

    #[test]
    fn scroll_accumulates_slice_delta() {
        let mut style = ImageSliceStyle::new(presets::SOFT_TISSUE);
        let mut cam = default_camera();

        style.on_mouse_event(
            &mouse((400.0, 300.0), MouseEventKind::Scroll(3.0)),
            &ctx(),
            &mut cam,
        );
        let d = style.take_slice_delta();
        assert!(d > 0.0, "scroll up should give positive delta");
        assert_eq!(style.take_slice_delta(), 0.0, "delta consumed");
    }
}
