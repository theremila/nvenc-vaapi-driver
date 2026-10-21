//! Live backend-ABI regression: changing RC, two contexts, surface reuse and teardown.
mod common;
use libloading::Library;
use std::ffi::c_void;
use std::fs::File;
use std::os::fd::AsRawFd;
use va_sys as va;

struct Driver {
    ctx: Box<va::VADriverContext>,
    vt: Box<va::VADriverVTable>,
    _drm: Box<va::drm_state>,
    _node: File,
    _lib: Library,
}

impl Driver {
    fn new() -> Self {
        let node = File::options()
            .read(true)
            .write(true)
            .open(common::render_node())
            .expect("open NVIDIA render node (override NVENC_TEST_RENDER_NODE)");
        // SAFETY: zeroed backend structs contain only scalar/pointer fields.
        let mut ctx: Box<va::VADriverContext> = Box::new(unsafe { std::mem::zeroed() });
        let mut vt = Box::new(unsafe { std::mem::zeroed::<va::VADriverVTable>() });
        let mut drm = Box::new(unsafe { std::mem::zeroed::<va::drm_state>() });
        drm.fd = node.as_raw_fd();
        ctx.vtable = &mut *vt;
        ctx.display_type = va::VA_DISPLAY_DRM as _;
        ctx.drm_state = (&mut *drm as *mut va::drm_state).cast();
        // SAFETY: load our built ABI with a matching backend context.
        let lib = unsafe { Library::new(common::build_driver()) }.unwrap();
        let init = unsafe {
            lib.get::<unsafe extern "C" fn(va::VADriverContextP) -> i32>(b"__vaDriverInit_1_23")
        }
        .unwrap();
        assert_eq!(unsafe { init(&mut *ctx) }, 0);
        Self {
            ctx,
            vt,
            _drm: drm,
            _node: node,
            _lib: lib,
        }
    }

    fn upload(&mut self, surface: u32, frame: usize) {
        let ctx = &mut *self.ctx as *mut va::VADriverContext;
        let mut image: va::VAImage = unsafe { std::mem::zeroed() };
        unsafe {
            assert_eq!(self.vt.vaDeriveImage.unwrap()(ctx, surface, &mut image), 0);
            let mut ptr = std::ptr::null_mut();
            assert_eq!(self.vt.vaMapBuffer.unwrap()(ctx, image.buf, &mut ptr), 0);
            let data = std::slice::from_raw_parts_mut(ptr.cast::<u8>(), image.data_size as usize);
            data.fill(128);
            for y in 0..288usize {
                for x in 0..512usize {
                    data[y * image.pitches[0] as usize + x] = ((x + y + frame) % 200 + 16) as u8;
                }
            }
            assert_eq!(self.vt.vaUnmapBuffer.unwrap()(ctx, image.buf), 0);
            assert_eq!(self.vt.vaDestroyImage.unwrap()(ctx, image.image_id), 0);
        }
    }

    fn buffer<T>(&mut self, context: u32, kind: u32, value: &mut T) -> u32 {
        let mut id = va::VA_INVALID_ID;
        // SAFETY: value is a writable, live parameter struct for this call.
        assert_eq!(
            unsafe {
                self.vt.vaCreateBuffer.unwrap()(
                    &mut *self.ctx,
                    context,
                    kind,
                    std::mem::size_of::<T>() as u32,
                    1,
                    (value as *mut T).cast(),
                    &mut id,
                )
            },
            0
        );
        id
    }
}
impl Drop for Driver {
    fn drop(&mut self) {
        // SAFETY: context, vtable and library are still live.
        if let Some(term) = self.vt.vaTerminate {
            assert_eq!(unsafe { term(&mut *self.ctx) }, 0);
        }
    }
}

#[test]
#[ignore = "requires a live NVIDIA GPU, render node and ffmpeg"]
fn dynamic_rate_control_two_contexts_and_surface_lifetime() {
    let mut d = Driver::new();
    let ctx = &mut *d.ctx as *mut va::VADriverContext;
    let mut config = va::VA_INVALID_ID;
    let mut surface = va::VA_INVALID_SURFACE;
    let mut contexts = [va::VA_INVALID_ID; 2];
    let mut coded = [va::VA_INVALID_ID; 2];
    // SAFETY: every array/output has the size passed to the backend.
    unsafe {
        assert_eq!(
            d.vt.vaCreateConfig.unwrap()(
                ctx,
                va::VAProfileH264Main,
                va::VAEntrypointEncSlice,
                std::ptr::null_mut(),
                0,
                &mut config
            ),
            0
        );
        assert_eq!(
            d.vt.vaCreateSurfaces2.unwrap()(
                ctx,
                va::VA_RT_FORMAT_YUV420,
                512,
                288,
                &mut surface,
                1,
                std::ptr::null_mut(),
                0
            ),
            0
        );
        for i in 0..2 {
            assert_eq!(
                d.vt.vaCreateContext.unwrap()(
                    ctx,
                    config,
                    512,
                    288,
                    0,
                    &mut surface,
                    1,
                    &mut contexts[i]
                ),
                0
            );
            assert_eq!(
                d.vt.vaCreateBuffer.unwrap()(
                    ctx,
                    contexts[i],
                    va::VAEncCodedBufferType,
                    512 * 288,
                    1,
                    std::ptr::null_mut(),
                    &mut coded[i]
                ),
                0
            );
        }
    }
    let mut output = [Vec::<u8>::new(), Vec::<u8>::new()];
    let frames: usize = std::env::var("NVENC_TEST_FRAMES")
        .unwrap_or_else(|_| "360".into())
        .parse()
        .unwrap();
    assert!((1..=36000).contains(&frames));
    for frame in 0..frames {
        d.upload(surface, frame);
        for i in 0..2 {
            let mut ids = Vec::new();
            if frame == 0 {
                let mut seq: va::VAEncSequenceParameterBufferH264 = unsafe { std::mem::zeroed() };
                seq.picture_width_in_mbs = 32;
                seq.picture_height_in_mbs = 18;
                seq.bits_per_second = 1_000_000;
                seq.intra_idr_period = 60;
                seq.time_scale = 120;
                seq.num_units_in_tick = 1;
                ids.push(d.buffer(contexts[i], va::VAEncSequenceParameterBufferType, &mut seq));
            }
            let mut pic: va::VAEncPictureParameterBufferH264 = unsafe { std::mem::zeroed() };
            pic.coded_buf = coded[i];
            pic.pic_init_qp = 28;
            unsafe {
                pic.pic_fields
                    .bits
                    .as_mut()
                    .set_idr_pic_flag(u32::from(frame % 60 == 0));
            }
            ids.push(d.buffer(contexts[i], va::VAEncPictureParameterBufferType, &mut pic));
            // Small deltas every frame plus large changes and fractional FPS.
            let mut rc = [
                1u32,
                [1_000_000, 3_000_000, 500_000][(frame / 60) % 3] + frame as u32,
            ];
            ids.push(d.buffer(contexts[i], va::VAEncMiscParameterBufferType, &mut rc));
            let mut fps = [
                0u32,
                if frame < 90 {
                    60 | (1 << 16)
                } else {
                    30000 | (1001 << 16)
                },
            ];
            ids.push(d.buffer(contexts[i], va::VAEncMiscParameterBufferType, &mut fps));
            unsafe {
                assert_eq!(d.vt.vaBeginPicture.unwrap()(ctx, contexts[i], surface), 0);
                assert_eq!(
                    d.vt.vaRenderPicture.unwrap()(
                        ctx,
                        contexts[i],
                        ids.as_mut_ptr(),
                        ids.len() as i32
                    ),
                    0
                );
                assert_eq!(
                    d.vt.vaEndPicture.unwrap()(ctx, contexts[i]),
                    0,
                    "frame={frame}, context={i}"
                );
                assert_eq!(d.vt.vaSyncBuffer.unwrap()(ctx, coded[i], u64::MAX), 0);
                let mut ptr: *mut c_void = std::ptr::null_mut();
                assert_eq!(d.vt.vaMapBuffer.unwrap()(ctx, coded[i], &mut ptr), 0);
                let segment = &*ptr.cast::<va::VACodedBufferSegment>();
                assert!(segment.size > 0, "empty frame {frame}");
                output[i].extend_from_slice(std::slice::from_raw_parts(
                    segment.buf.cast::<u8>(),
                    segment.size as usize,
                ));
                assert_eq!(d.vt.vaUnmapBuffer.unwrap()(ctx, coded[i]), 0);
                for id in ids {
                    assert_eq!(d.vt.vaDestroyBuffer.unwrap()(ctx, id), 0);
                }
            }
        }
    }
    // Destroy the input while sessions still hold registrations, then
    // encode a replacement and let vaTerminate clean up live resources.
    let old_surface = surface;
    unsafe {
        assert_eq!(d.vt.vaDestroySurfaces.unwrap()(ctx, &mut surface, 1), 0);
        assert_eq!(
            d.vt.vaCreateSurfaces2.unwrap()(
                ctx,
                va::VA_RT_FORMAT_YUV420,
                512,
                288,
                &mut surface,
                1,
                std::ptr::null_mut(),
                0
            ),
            0
        );
        assert_ne!(surface, old_surface);
        assert_ne!(d.vt.vaSyncSurface.unwrap()(ctx, old_surface), 0);
    }
    d.upload(surface, 0);
    let mut pic: va::VAEncPictureParameterBufferH264 = unsafe { std::mem::zeroed() };
    pic.coded_buf = coded[0];
    let mut param = d.buffer(contexts[0], va::VAEncPictureParameterBufferType, &mut pic);
    unsafe {
        assert_eq!(d.vt.vaBeginPicture.unwrap()(ctx, contexts[0], surface), 0);
        assert_eq!(
            d.vt.vaRenderPicture.unwrap()(ctx, contexts[0], &mut param, 1),
            0
        );
        assert_eq!(d.vt.vaEndPicture.unwrap()(ctx, contexts[0]), 0);
    }
    for (i, bytes) in output.iter().enumerate() {
        let path = std::env::temp_dir().join(format!("nvenc_rc_{}_{i}.h264", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        let probe = std::process::Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "csv=p=0",
            ])
            .arg(&path)
            .output()
            .unwrap();
        assert!(probe.status.success());
        assert!(
            probe.stderr.is_empty(),
            "{}",
            String::from_utf8_lossy(&probe.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&probe.stdout).trim(),
            frames.to_string()
        );
        std::fs::remove_file(path).unwrap();
    }
}

#[test]
#[ignore = "requires a live NVIDIA GPU and render node"]
fn nv12_image_readback_crops_and_mapping_access() {
    let mut d = Driver::new();
    let ctx = &mut *d.ctx as *mut va::VADriverContext;
    let mut surface = va::VA_INVALID_SURFACE;
    let mut derived: va::VAImage = unsafe { std::mem::zeroed() };
    let mut image: va::VAImage = unsafe { std::mem::zeroed() };
    let mut format: va::VAImageFormat = unsafe { std::mem::zeroed() };
    format.fourcc = va::VA_FOURCC_NV12;
    // SAFETY: all output structs, buffers and rectangles are sized as passed.
    unsafe {
        assert_eq!(
            d.vt.vaCreateSurfaces2.unwrap()(
                ctx,
                va::VA_RT_FORMAT_YUV420,
                66,
                34,
                &mut surface,
                1,
                std::ptr::null_mut(),
                0
            ),
            0
        );
        assert_eq!(d.vt.vaDeriveImage.unwrap()(ctx, surface, &mut derived), 0);
        let mut ptr = std::ptr::null_mut();
        assert_eq!(
            d.vt.vaMapBuffer2.unwrap()(ctx, derived.buf, &mut ptr, 4),
            va::VA_STATUS_ERROR_INVALID_PARAMETER as i32
        );
        assert_eq!(
            d.vt.vaMapBuffer2.unwrap()(ctx, derived.buf, &mut ptr, va::VA_MAPBUFFER_FLAG_WRITE),
            0
        );
        let bytes = std::slice::from_raw_parts_mut(ptr.cast::<u8>(), derived.data_size as usize);
        bytes.fill(0xa5);
        for plane in 0..2 {
            let height = if plane == 0 { 34 } else { 17 };
            for y in 0..height {
                for x in 0..66 {
                    bytes[derived.offsets[plane] as usize
                        + y * derived.pitches[plane] as usize
                        + x] = (x + 3 * y + 80 * plane) as u8;
                }
            }
        }
        assert_eq!(d.vt.vaUnmapBuffer.unwrap()(ctx, derived.buf), 0);
        // A read map sees current device bytes; unmap must not upload them.
        assert_eq!(
            d.vt.vaMapBuffer2.unwrap()(ctx, derived.buf, &mut ptr, va::VA_MAPBUFFER_FLAG_READ),
            0
        );
        assert_eq!(*ptr.cast::<u8>().add(5), 5);
        // Deliberately modify the host shadow to detect an accidental upload.
        *ptr.cast::<u8>().add(5) = 222;
        assert_eq!(d.vt.vaUnmapBuffer.unwrap()(ctx, derived.buf), 0);
        assert_eq!(d.vt.vaMapBuffer.unwrap()(ctx, derived.buf, &mut ptr), 0);
        assert_eq!(*ptr.cast::<u8>().add(5), 5);
        *ptr.cast::<u8>().add(5) = 111;
        assert_eq!(d.vt.vaUnmapBuffer.unwrap()(ctx, derived.buf), 0);
        assert_eq!(
            d.vt.vaCreateImage.unwrap()(ctx, &mut format, 66, 34, &mut image),
            0
        );
        assert_eq!(
            d.vt.vaGetImage.unwrap()(ctx, surface, 0, 0, 66, 34, image.image_id),
            0
        );
        assert_eq!(
            d.vt.vaMapBuffer2.unwrap()(ctx, image.buf, &mut ptr, va::VA_MAPBUFFER_FLAG_READ),
            0
        );
        let bytes = std::slice::from_raw_parts(ptr.cast::<u8>(), image.data_size as usize);
        for plane in 0..2 {
            let height = if plane == 0 { 34 } else { 17 };
            for y in 0..height {
                for x in 0..66 {
                    let expected = if plane == 0 && y == 0 && x == 5 {
                        111
                    } else {
                        (x + 3 * y + 80 * plane) as u8
                    };
                    assert_eq!(
                        bytes
                            [image.offsets[plane] as usize + y * image.pitches[plane] as usize + x],
                        expected
                    );
                }
            }
        }
        assert_ne!(
            d.vt.vaGetImage.unwrap()(ctx, surface, 0, 0, 66, 34, image.image_id),
            0
        );
        assert_eq!(d.vt.vaUnmapBuffer.unwrap()(ctx, image.buf), 0);
        assert_eq!(d.vt.vaDestroyImage.unwrap()(ctx, image.image_id), 0);
        assert_eq!(
            d.vt.vaCreateImage.unwrap()(ctx, &mut format, 22, 12, &mut image),
            0
        );
        assert_eq!(
            d.vt.vaGetImage.unwrap()(ctx, surface, 8, 6, 22, 12, image.image_id),
            0
        );
        assert_eq!(d.vt.vaMapBuffer.unwrap()(ctx, image.buf, &mut ptr), 0);
        let bytes = std::slice::from_raw_parts(ptr.cast::<u8>(), image.data_size as usize);
        for plane in 0..2 {
            let height = if plane == 0 { 12 } else { 6 };
            let source_y = if plane == 0 { 6 } else { 3 };
            for y in 0..height {
                for x in 0..22 {
                    assert_eq!(
                        bytes
                            [image.offsets[plane] as usize + y * image.pitches[plane] as usize + x],
                        (x + 8 + 3 * (y + source_y) + 80 * plane) as u8
                    );
                }
            }
        }
        assert_eq!(d.vt.vaUnmapBuffer.unwrap()(ctx, image.buf), 0);
        for (x, y, w, h) in [
            (-2, 0, 22, 12),
            (1, 0, 22, 12),
            (0, 1, 22, 12),
            (46, 0, 22, 12),
            (0, 24, 22, 12),
            (0, 0, 20, 12),
        ] {
            assert_eq!(
                d.vt.vaGetImage.unwrap()(ctx, surface, x, y, w, h, image.image_id),
                va::VA_STATUS_ERROR_INVALID_PARAMETER as i32
            );
        }
        assert_eq!(
            d.vt.vaGetImage.unwrap()(ctx, surface, 0, 0, 66, 34, derived.image_id),
            va::VA_STATUS_ERROR_SURFACE_BUSY as i32
        );
        assert_eq!(
            d.vt.vaPutImage.unwrap()(ctx, surface, derived.image_id, 0, 0, 66, 34, 0, 0, 66, 34),
            va::VA_STATUS_ERROR_SURFACE_BUSY as i32
        );
        assert_eq!(d.vt.vaDestroyImage.unwrap()(ctx, image.image_id), 0);
        assert_eq!(d.vt.vaDestroyImage.unwrap()(ctx, derived.image_id), 0);
        // Reject invalid image sizes before allocating or inserting objects.
        for (w, h) in [(3, 2), (2, 3), (65536, 2)] {
            assert_eq!(
                d.vt.vaCreateImage.unwrap()(ctx, &mut format, w, h, &mut image),
                va::VA_STATUS_ERROR_INVALID_PARAMETER as i32
            );
        }
        assert_eq!(d.vt.vaDestroySurfaces.unwrap()(ctx, &mut surface, 1), 0);
    }
}

#[test]
#[ignore = "requires a live NVIDIA GPU and render node"]
fn derived_images_copy_between_surfaces_and_reject_stale_surface() {
    let mut d = Driver::new();
    let ctx = &mut *d.ctx as *mut va::VADriverContext;
    let mut surfaces = [va::VA_INVALID_SURFACE; 2];
    let mut images: [va::VAImage; 2] = unsafe { std::mem::zeroed() };
    // SAFETY: arrays and image buffers remain live and are sized as passed.
    unsafe {
        assert_eq!(
            d.vt.vaCreateSurfaces2.unwrap()(
                ctx,
                va::VA_RT_FORMAT_YUV420,
                66,
                34,
                surfaces.as_mut_ptr(),
                2,
                std::ptr::null_mut(),
                0
            ),
            0
        );
        let mut ptr = std::ptr::null_mut();
        for i in 0..2 {
            assert_eq!(
                d.vt.vaDeriveImage.unwrap()(ctx, surfaces[i], &mut images[i]),
                0
            );
            assert_eq!(
                d.vt.vaMapBuffer2.unwrap()(
                    ctx,
                    images[i].buf,
                    &mut ptr,
                    va::VA_MAPBUFFER_FLAG_WRITE
                ),
                0
            );
            std::slice::from_raw_parts_mut(ptr.cast::<u8>(), images[i].data_size as usize)
                .fill(40 + i as u8);
            assert_eq!(d.vt.vaUnmapBuffer.unwrap()(ctx, images[i].buf), 0);
        }
        assert_eq!(
            d.vt.vaGetImage.unwrap()(ctx, surfaces[0], 0, 0, 66, 34, images[1].image_id),
            0
        );
        assert_eq!(
            d.vt.vaMapBuffer2.unwrap()(ctx, images[1].buf, &mut ptr, va::VA_MAPBUFFER_FLAG_READ),
            0
        );
        assert_eq!(*ptr.cast::<u8>(), 40);
        // Leave a stale host shadow; vaPutImage must read the derived surface.
        *ptr.cast::<u8>() = 200;
        assert_eq!(d.vt.vaUnmapBuffer.unwrap()(ctx, images[1].buf), 0);
        assert_eq!(
            d.vt.vaMapBuffer2.unwrap()(ctx, images[0].buf, &mut ptr, va::VA_MAPBUFFER_FLAG_WRITE),
            0
        );
        std::slice::from_raw_parts_mut(ptr.cast::<u8>(), images[0].data_size as usize).fill(60);
        assert_eq!(d.vt.vaUnmapBuffer.unwrap()(ctx, images[0].buf), 0);
        assert_eq!(
            d.vt.vaPutImage.unwrap()(
                ctx,
                surfaces[0],
                images[1].image_id,
                0,
                0,
                66,
                34,
                0,
                0,
                66,
                34
            ),
            0
        );
        assert_eq!(
            d.vt.vaMapBuffer2.unwrap()(ctx, images[0].buf, &mut ptr, va::VA_MAPBUFFER_FLAG_READ),
            0
        );
        assert_eq!(*ptr.cast::<u8>(), 40);
        assert_eq!(d.vt.vaUnmapBuffer.unwrap()(ctx, images[0].buf), 0);
        assert_eq!(
            d.vt.vaDestroySurfaces.unwrap()(ctx, surfaces.as_mut_ptr(), 2),
            0
        );
        assert_eq!(
            d.vt.vaMapBuffer2.unwrap()(ctx, images[0].buf, &mut ptr, va::VA_MAPBUFFER_FLAG_READ),
            va::VA_STATUS_ERROR_INVALID_SURFACE as i32
        );
        // A failed read leaves the buffer unmapped and destroyable.
        for image in images {
            assert_eq!(d.vt.vaDestroyImage.unwrap()(ctx, image.image_id), 0);
        }
    }
}
