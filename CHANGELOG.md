Change Log
=====

## 0.3.0
- Merge the `llama.cpp` release line (0.2.x source of truth) into `develop`.
- Pin `genai_core` to `~> 0.3.5` (from `~> 0.3`).

## 0.2.3
Complete the Linux build fix started in 0.2.2:

- Pass -DCMAKE_POSITION_INDEPENDENT_CODE=ON so static archives can be
  linked into the NIF .so without "relocation ... can not be used when
  making a shared object" errors on any Linux arch.
- Add -lopenblas -lgomp -ldl to the Linux link line so sgemm_/dgemm_
  (from libggml-blas.a), OpenMP symbols (from libggml-cpu.a), and
  dlopen (from libggml.a) resolve at both link time and runtime.

Darwin link line unchanged.

## 0.2.2
Fix Linux build: enable GGML_BLAS=ON in the cmake step so libggml-blas.a
is produced, matching what the NIF link line already expects. On Darwin
this flag picks Apple Accelerate (prior default); on Linux it selects
OpenBLAS (requires libopenblas-dev at build time).

## 0.2.0
C NIF rewrite: switch from the outdated Rust wrapper to a C NIF built
against the latest llama.cpp. (Published to Hex as 0.2.0–0.2.1 from the
`llama.cpp` branch; in-tree changelog backfilled here.)

## 0.1.0
Update to use GenAI Core structs
