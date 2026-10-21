//! Encoding completes synchronously in vaEndPicture. Synchronisation still
//! validates IDs/readiness so invalid or unsubmitted buffers cannot look ready.
use super::{guard, state_from};
use crate::driver::state::BufferStorage;
use crate::error::{DriverError, VAStatus};
use crate::ids::ContainsKey;
use va_sys as va;

pub unsafe extern "C" fn sync_surface(
    ctx: va::VADriverContextP,
    target: va::VASurfaceID,
) -> VAStatus {
    guard(|| {
        // SAFETY: caller supplies a live driver context.
        let state = unsafe { state_from(ctx) }.ok_or(DriverError::InvalidParameter)?;
        state
            .pools
            .lock()
            .surfaces
            .find_by_low_bits(target)
            .ok_or(DriverError::InvalidSurface)?;
        Ok(())
    })
}

pub unsafe extern "C" fn sync_surface2(
    ctx: va::VADriverContextP,
    target: va::VASurfaceID,
    _timeout_ns: u64,
) -> VAStatus {
    // SAFETY: synchronous implementation has the same preconditions.
    unsafe { sync_surface(ctx, target) }
}

pub unsafe extern "C" fn sync_buffer(
    ctx: va::VADriverContextP,
    id: va::VABufferID,
    _timeout_ns: u64,
) -> VAStatus {
    guard(|| {
        // SAFETY: caller supplies a live driver context.
        let state = unsafe { state_from(ctx) }.ok_or(DriverError::InvalidParameter)?;
        let pools = state.pools.lock();
        let key = pools
            .buffers
            .find_by_low_bits(id)
            .ok_or(DriverError::InvalidBuffer)?;
        let rec = &pools.buffers[key];
        if matches!(rec.storage, BufferStorage::Coded(_)) && !*rec.coded_ready.lock() {
            return Err(DriverError::OperationFailed(
                "coded buffer has no submitted picture",
            ));
        }
        Ok(())
    })
}
