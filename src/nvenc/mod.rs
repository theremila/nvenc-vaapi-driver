//! CUDA/NVENC session ownership, H.264 configuration and preset selection.

#![allow(dead_code)]

mod api;
pub mod h264_config;
pub mod preset;
pub mod session;

pub use session::NvencSession;
