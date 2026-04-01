//! Abstract input event types.
//!
//! The library is windowing-system agnostic. Consumers bridge their platform
//! events (winit, SDL2, egui, etc.) to the abstract types defined here.

// ── Mouse ─────────────────────────────────────────────────────────────────────

/// Mouse button identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MouseButton {
    /// Left (primary) button.
    Left,
    /// Right (secondary) button.
    Right,
    /// Middle (scroll wheel) button.
    Middle,
}

/// The specific mouse action that occurred.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum MouseEventKind {
    /// A button was pressed.
    Press(MouseButton),
    /// A button was released.
    Release(MouseButton),
    /// The cursor moved.
    Move,
    /// Mouse wheel scrolled. Positive = scroll up (zoom in convention).
    Scroll(f64),
}

/// Abstract mouse event.
///
/// All positions are in **logical pixels** relative to the viewport origin
/// (top-left). Sub-pixel precision is preserved as `f64`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MouseEvent {
    /// Cursor position in logical pixels.
    pub position: (f64, f64),
    /// What kind of mouse action occurred.
    pub kind: MouseEventKind,
    /// Modifier keys held during this event.
    pub modifiers: Modifiers,
}

// ── Keyboard ──────────────────────────────────────────────────────────────────

/// Platform-independent key identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Key {
    /// Reset.
    R,
    /// X axis.
    X,
    /// Y axis.
    Y,
    /// Z axis.
    Z,
    /// Arrow up.
    Up,
    /// Arrow down.
    Down,
    /// Arrow left.
    Left,
    /// Arrow right.
    Right,
    /// Zoom in / increase.
    Plus,
    /// Zoom out / decrease.
    Minus,
    /// Cancel / exit.
    Escape,
    /// Any other printable character.
    Char(char),
}

/// Abstract keyboard event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum KeyEvent {
    /// A key was pressed.
    Pressed {
        /// Which key.
        key: Key,
    },
    /// A key was released.
    Released {
        /// Which key.
        key: Key,
    },
}

/// Modifier keys held during an event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Modifiers {
    /// Shift key held.
    pub shift: bool,
    /// Control key held (Cmd on macOS).
    pub ctrl: bool,
    /// Alt/Option key held.
    pub alt: bool,
}

// ── Context ───────────────────────────────────────────────────────────────────

/// Viewport and volume context passed alongside every event.
#[derive(Debug, Clone, Copy)]
pub struct InteractionContext {
    /// Viewport width in logical pixels.
    pub viewport_width: f64,
    /// Viewport height in logical pixels.
    pub viewport_height: f64,
    /// Volume bounding box, if a volume is loaded.
    pub volume_bounds: Option<crate::math::Aabb>,
}

impl InteractionContext {
    /// Aspect ratio (width / height).
    #[must_use]
    pub fn aspect(&self) -> f64 {
        if self.viewport_height > 0.0 {
            self.viewport_width / self.viewport_height
        } else {
            1.0
        }
    }
}

// ── Result ────────────────────────────────────────────────────────────────────

/// Result of processing an interaction event.
///
/// Tells the consumer what changed so it can decide whether to re-render.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct InteractionResult {
    /// The camera was modified.
    pub camera_changed: bool,
    /// The window/level was modified (for 2D slice styles).
    pub window_level_changed: bool,
    /// The slice plane was modified.
    pub slice_changed: bool,
    /// Convenience: true if any of the above changed.
    pub needs_redraw: bool,
}

impl InteractionResult {
    /// Convenience: only the camera changed.
    #[must_use]
    pub fn camera_only() -> Self {
        Self {
            camera_changed: true,
            needs_redraw: true,
            ..Self::default()
        }
    }

    /// Convenience: nothing changed.
    #[must_use]
    pub fn nothing() -> Self {
        Self::default()
    }

    /// Convenience: window/level changed.
    #[must_use]
    pub fn window_level_only() -> Self {
        Self {
            window_level_changed: true,
            needs_redraw: true,
            ..Self::default()
        }
    }

    /// Convenience: slice changed.
    #[must_use]
    pub fn slice_only() -> Self {
        Self {
            slice_changed: true,
            needs_redraw: true,
            ..Self::default()
        }
    }
}
