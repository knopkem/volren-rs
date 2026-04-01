//! Abstract input events and interaction styles.
//!
//! The library is windowing-system agnostic. Consumers bridge their platform
//! events (winit, SDL2, etc.) to the abstract types defined here.

mod events;
mod image_slice;
mod trackball;

pub use events::{
    InteractionContext, InteractionResult, Key, KeyEvent, Modifiers, MouseButton, MouseEvent,
    MouseEventKind,
};
pub use image_slice::ImageSliceStyle;
pub use trackball::TrackballStyle;

use crate::camera::Camera;

/// An interaction style drives a [`Camera`] in response to abstract input events.
///
/// # VTK Equivalent
/// `vtkInteractorStyle` — specifically `vtkInteractorStyleTrackballCamera`
/// and `vtkInteractorStyleImage`.
pub trait InteractionStyle: Send + Sync {
    /// Handle a mouse event and potentially mutate the camera.
    ///
    /// Returns an [`InteractionResult`] describing what changed so the
    /// consumer knows whether to trigger a re-render.
    fn on_mouse_event(
        &mut self,
        event: &MouseEvent,
        ctx: &InteractionContext,
        camera: &mut Camera,
    ) -> InteractionResult;

    /// Handle a key event.
    fn on_key_event(
        &mut self,
        event: &KeyEvent,
        ctx: &InteractionContext,
        camera: &mut Camera,
    ) -> InteractionResult;
}
