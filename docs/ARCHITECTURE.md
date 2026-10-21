# Architecture

## Loading and state

A VA-API client calls libva, which loads `nvidia_nvenc_drv_video.so` and resolves
an exported `__vaDriverInit_1_*` entry point. The backend is a userspace library,
not a kernel driver or a daemon. Driver state belongs to `VADriverContext` and
is released by `vaTerminate`.

`DriverState` owns lazy CUDA selection and `Mutex<Pools>`. Typed object pools
store configs, surfaces, contexts, buffers and images. IDs are monotonically
allocated nonzero 32-bit values and are never reused during the display's
lifetime. Allocation fails before the invalid-ID sentinel or wraparound.

## Encode lifecycle

1. Config creation validates profile, entry point and rate-control mode.
2. Surfaces own pitched CUDA NV12 memory or imported external memory.
3. Context creation retains geometry and configuration. NVENC remains unopened.
4. Begin/render collect a picture's input and parameter buffers.
5. End validates output, merges sequence and misc controls, lazily opens or
   reconfigures NVENC, and registers input in that session.
6. NVENC maps input, submits, locks output, copies compressed bytes, unlocks
   output and unmaps input. No B-frames or lookahead are enabled.
7. Successful completion publishes a matching coded segment. Sync validates
   object/readiness; map exposes stable storage until unmap.

Encoding is synchronous. The global pool lock also serialises independent
contexts within a display. This simplifies resource lifetime but limits
parallel throughput. An asynchronous implementation must retain a distinct
coded-buffer/input association until completion; delivering an older picture
into a newer coded buffer is invalid.

## Resources and teardown

Each `NvencSession` owns one reusable NVENC output buffer and a cache of input
registrations. Registrations belong to a session, so two contexts may share
one surface safely. Surface destruction unregisters it from all sessions before
freeing CUDA memory. Driver termination destroys contexts before surfaces.

VA images use host memory for CPU upload. Mapping a generic buffer retains its
storage lock until unmap. Mutating, rendering or destroying mapped buffers must
be rejected before attempting to take that lock.

## Rate control and format limits

Sequence parameters establish dimensions, cropping, bitrate, GOP and rational
frame rate. Misc rate-control and frame-rate parameters take precedence. VBR
keeps average and peak bitrate separate; CQP uses picture QP. Equal configs
skip NVENC reconfiguration. Rate changes do not force IDR.

NVENC generates H.264 headers. Arbitrary client packed headers, exact reference
picture programming, all VA misc controls and arbitrary slice layouts are not
fully implemented. Capability reporting must not promise those semantics.

## External memory

The import path validates a single-object, single-layer linear NV12 descriptor
with two tightly packed planes, matching pitch, offsets and bounded allocation.
Duplicated file descriptors transfer ownership only after successful CUDA
import; the owning CUDA context remains alive through external-memory teardown.
Descriptor validation alone does not establish interoperability with arbitrary
DMA-BUF exporters or compositor modifiers. Surface export is unsupported.

## Device selection

An explicit PCI address takes precedence, followed by the display's DRM PCI
address. A sole CUDA device may be selected without a DRM match; ambiguous
selection fails. No application names or GPU models participate in routing.
