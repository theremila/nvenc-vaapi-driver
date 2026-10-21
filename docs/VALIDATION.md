# Validation

## CPU and ABI checks

```sh
cargo test --locked --all-targets --all-features
cargo clippy --locked --all-targets --all-features -- -D warnings
git diff --check
python3 -m unittest discover -s tools/tests -v
```

The default suite does not encode on a GPU. ABI tests load the release shared
library and exercise capability queries, invalid arguments, mapping, identifier
lifetime and termination. Hardware tests are marked ignored by default.

## Hardware integration

Requires FFmpeg with `h264_vaapi`, ffprobe, vainfo, NVIDIA runtime libraries and
an accessible NVIDIA render node. Set `NVENC_TEST_RENDER_NODE` to select a node
when discovery is unsuitable. `NVENC_TEST_VAINFO` can point to a vainfo binary.

```sh
NVENC_TEST_FRAMES=1800 cargo test --release --locked -- --include-ignored --nocapture
```

Tests must fail when required tools or hardware are unavailable. Validate exact
frame counts, visible dimensions, profile and full-stream decoding. Fractional
frame rate is checked against SPS timing. ABI integration covers dynamic rate
changes, shared input across contexts, stale IDs and live-resource termination.
Image ABI tests compare full-frame and cropped NV12 readback byte for byte,
exercise mapping access flags and copies between derived surfaces, and reject
mapped buffers, invalid rectangles and stale surface handles.
Increase `NVENC_TEST_FRAMES` for longer stability runs.

## Performance

```sh
python3 tools/benchmark-encode.py --help
```

Use `--drivers-path DIR` to compare release builds without replacing the
installed library. Alternate old/new build order and compare repeated runs.

Compare VA-API, native NVENC and software encoding with identical input, output
geometry, bitrate, frame count and pacing. Record throughput, latency, CPU time,
resident memory and GPU measurements. Input generation and upload contribute
to process CPU time. Whole-GPU power includes other workloads and is not
whole-system energy. Battery operation, thermal limits and concurrent graphics
work invalidate comparisons made under different conditions.

Do not infer minimum energy use from GPU utilization alone. Keep raw results
outside tracked documentation, for example under `target/validation/`.

## Stage profiling

Build with
`cargo build --release --features profiling --target-dir target/profiling`
to enable process-wide timing counters. Create the usual libva symlink in that release
directory, then pass it to the benchmark with `--drivers-path`. On termination
the driver writes `nvenc-vaapi-profile` counters to stderr: call counts and
aggregate nanoseconds for host images, CUDA copies and NVENC operations.
These counters contain no device identity or image data.

Timings include CPU waiting and are not GPU kernel timings. End-picture time
includes its nested stages; do not sum them together. Counters accumulate
across displays in the process. Ordinary builds compile out the timers and
atomic counters. Use ordinary builds for final throughput comparisons.

## Client release checks

FFmpeg success does not establish compatibility with capture or browser clients.
Before releasing, verify actual VA-API encoder selection, codec negotiation,
long-running capture, stream decoding, rate changes, reconnects and resource
cleanup in each supported client. Validate Wayland DMA-BUF import, sandboxed
packaging and hardware generations independently. Test that decode clients
retain their intended backend when choosing deployment scope.
