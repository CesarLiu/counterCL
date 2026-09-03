# Migrating the GPUDrive/CL4AD stack to an H200 server

Target container: **CUDA 13.1, PyTorch 2.10, Python 3.12**.
Source machine (where every number in `docs/results/` was measured): **NVIDIA L4
23 GB, CUDA 12.4, PyTorch 2.6.0+cu124, Python 3.11.15**.

The container's stack cannot run this code as-is. Section 1 says why, section 2
gives the two ways forward, sections 3-6 are the build, section 7 is the
verification gate, sections 8-10 are the experiments.

---

## 1. What actually blocks a drop-in run

GPUDrive is not a pip package. It is a CUDA C++ simulator (Madrona) with a
compiled Python extension, so both the Python ABI and the CUDA toolkit are
load-bearing.

### 1.1 The simulator binary is Python-3.11 ABI

```
cl4ad/madrona_gpudrive.cpython-311-x86_64-linux-gnu.so
```

`cpython-311` is not a label, it is the ABI tag. Under Python 3.12 this will not
import. There is no wheel; the only fix is a full CMake + CUDA rebuild
(section 5). `pufferlib` 2.0.6 has the same problem — it ships Cython
extensions (`extensions.cpython-311-*.so`, `puffernet.cpython-311-*.so`, and
~15 `ocean/*` modules) that are compiled at install time.

### 1.2 CUDA 13.1 is a real risk, and it fails late

Madrona *will attempt* the build: the only version gate is

```cmake
# external/madrona/src/mw/CMakeLists.txt:10
if (NOT CUDAToolkit_FOUND OR CUDAToolkit_VERSION_MAJOR LESS 12)
    return()
endif()
```

and 13 passes it. The problem is downstream. Madrona's device code includes
libcu++ headers directly from the toolkit:

```
external/madrona/src/mw/device/  ->  <cuda/barrier>, <cuda/std/tuple>,
                                     <cuda/std/type_traits>, <cuda/std/complex>,
                                     <cuda/std/cmath>, cuda::atomic,
                                     cuda::thread_scope_system
```

pulled in via `${CUDAToolkit_INCLUDE_DIRS}` (`src/mw/CMakeLists.txt:88`). This
Madrona fork is pinned at `4bda334` (2025-04-15, `m-naumann/madrona`), written
against **CCCL 2.x** as shipped with CUDA 12.x. **CUDA 13 ships CCCL 3.x**, a
breaking major release.

Two mitigations are already in the tree, and neither covers this: CUB is
vendored (`external/cub`, so the toolkit's CUB is bypassed) and Madrona brings
its own libc++ (`madrona_libcxx`). libcu++ is still taken from the toolkit.

**The failure mode is nasty.** That header set is compiled twice:

1. host-side at build time — a normal compile error, easy to see;
2. **again at runtime via NVRTC**, when Madrona JITs the megakernel with
   `-std=c++20 -default-device -rdc=true -use_fast_math -arch=sm_90` and
   `-I${CUDAToolkit_INCLUDE_DIRS}` (`src/mw/cuda_exec.cpp:957-1000`).

So `cmake && make` can succeed cleanly and the stack still dies on the first
`GPUDriveTorchEnv(...)` construction. **Do not treat a green build as a pass.**
The gate is section 7.1, not the build log.

### 1.3 PyTorch 2.10 is *not* a blocker — verified

Madrona and torch are decoupled. The bridge is nanobind + DLPack
(`external/madrona/src/python/bindings.cpp:52`, `tensor_to_pytorch`), and
`madrona_python_utils` links `CUDA::cudart` but **never libtorch**
(`src/python/CMakeLists.txt:23-27`). The only thing the two halves share is the
driver.

Consequence, and this is what makes Path A work: **a Madrona built against CUDA
12.4 runs fine in the same process as a torch built against CUDA 13.0.** You do
not have to move Madrona to CUDA 13 just because the container's torch is on it.

Untested, and the one thing to watch: `pufferlib` 2.0.6 predates torch 2.10.
Nothing in its API surface is exotic, but it has not been run against 2.10 here.

### 1.4 Smaller things that will still cost you an hour each

| # | Issue | Fix |
|---|---|---|
| a | `[tool.uv] default-groups = "all"` means a bare `uv sync` also installs the `vbd` group, which pins `jaxlib==0.5.3` and pulls `waymo-waymax` from git. Nothing in the curriculum path uses it. | `uv sync --no-group vbd` |
| b | Madrona downloads its own clang toolchain from GitHub releases at configure time (`external/madrona-toolchain/cmake/set_toolchain.cmake:51`), keyed by the submodule's git short hash. | Build with network, or pre-seed `MADRONA_TOOLCHAIN_VERSION` + `MADRONA_TOOLCHAIN_HASH` against a vendored tarball |
| c | `cmake==4.0.0` is a hard dependency, and CMake 4 dropped compatibility with `cmake_minimum_required(<3.5)` used by a transitive dep. | Keep `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` (already in the Dockerfile) |
| d | `cl4ad/.kernel_cache` (5.9 MB) is a **compiled megakernel for the L4's sm_89**. | Delete it before the first H200 run. Also unset/repoint `MADRONA_MWGPU_KERNEL_CACHE` |
| e | `libcuda.so.1` is absent at build time in a container without a GPU attached. | The Dockerfile's stub symlink dance (`/usr/local/cuda/lib64/stubs/`) — keep it |

---

## 2. Two paths. Take A unless you have a week to spare.

### Path A — CUDA 12.4 + Python 3.11 inside the CUDA 13.1 container (recommended)

Add a second CUDA toolkit and a second Python to the image. Both are just
packages; the container's CUDA 13.1 and torch 2.10 stay installed and unused by
this stack. Because of 1.3 you could even keep the container's torch for
unrelated work in the same image.

- **Risk:** low. This reproduces a configuration that is running right now.
- **Cost:** ~4 GB image growth, ~30 min build.
- **Deadline math:** ICRA is **2026-09-15**. Path A spends hours; Path B can
  spend days on a CCCL-3 port with no guarantee.

### Path B — native Python 3.12 + CUDA 13.1 + torch 2.10

Only worth it if the server's driver refuses CUDA 12.4 (it will not — see 3.1)
or if policy forbids a second toolkit. Expect to port Madrona's device code to
CCCL 3.x. Budget days, not hours, and hold section 7 as the gate.

### 2.1 The comparability rule — this constrains the plan, not just the build

Moving to sm_90 re-JITs the megakernel and changes float reduction order. Under
Path B, libcu++, torch RNG and the CUDA math library all change too. Results
will not be bit-reproducible across machines under **either** path.

That is survivable, because the protocol adopted after the PLR-vs-DR
reproduction is curves and AUC rather than endpoints — but only if arms are
compared **within one machine**.

> **Never compare an arm trained on the L4 against an arm trained on the H200.**

Practical consequence for Phase 3: `plr_n1000_s42` is already banked from the
L4 run. If the remaining arms move to the H200, **that cell must be re-run
there**, and the L4 copy kept only as a sanity reference. Budget the extra run.

---

## 3. Server prerequisites

### 3.1 Driver

```bash
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv
```

Expect `NVIDIA H200`, compute capability **9.0**, 143771 MiB. Any driver new
enough to ship in a CUDA-13.1 image (r580+) also runs CUDA 12.4 binaries —
CUDA is backward compatible on the driver side, which is what licenses Path A.

If compute capability is not 9.0, stop: Madrona derives `-arch=sm_<major><minor>`
from the live device (`cuda_exec.cpp:957`) and the cache in 1.4d is keyed to it.

### 3.2 Container flags

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /path/on/host/diffCurricula:/workspace/diffCurricula \
  -w /workspace/diffCurricula <image> bash
```

`--ipc=host` matters: PufferLib's vectorisation uses shared memory and the
default 64 MB `/dev/shm` is not enough.

### 3.3 Storage budget

| item | size | note |
|---|---|---|
| `cl4ad/data/gpudrive_mini` | 2.4 GB | 1000 training / 150 validation / 150 testing scenes |
| `cl4ad/data/subsets` | 35 MB | symlink trees, regenerate rather than copy |
| `cl4ad/data/pools` | ~3.3 GB **per** (N=1000, seed) | transient; the sweep builds and deletes one at a time |
| `cl4ad/runs` | ~9 MB per run | checkpoints are 636 KB; curriculum pickles are the bulk |
| build tree + toolchain | ~6 GB | |

**~20 GB free is comfortable.** Storage was the binding constraint on the L4
box (15 GB free); it should not be one here. Keep `--min_free_gb` set anyway.

---

## 4. Environment build (Path A)

```dockerfile
# Dockerfile.h200 — layered on the existing CUDA 13.1 image
ARG BASE=<your-cuda13.1-torch2.10-py3.12-image>
FROM ${BASE}

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y -q --no-install-recommends \
      build-essential git curl ca-certificates wget vim ffmpeg \
      libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
      mesa-common-dev libc++1 libjpeg-dev libpng-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# --- second CUDA toolkit: 12.4, side by side with 13.1 -----------------------
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
 && dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb \
 && apt-get update && apt-get install -y --no-install-recommends cuda-toolkit-12-4 \
 && apt-get clean && rm -rf /var/lib/apt/lists/*
# NOTE: do NOT repoint /usr/local/cuda. Leave 13.1 as the default and select
# 12.4 explicitly for the Madrona build only (section 5).

RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/bin sh
RUN uv python install 3.11
```

Then, inside the container:

```bash
cd /workspace/diffCurricula/cl4ad
git submodule update --init --recursive
uv venv --python 3.11 .venv
uv sync --frozen --no-group vbd        # see 1.4a
```

If `--frozen` fails because `uv.lock` disagrees with the environment, drop it
once and re-lock — but **pin torch to `2.6.0+cu124`** to match the source
machine. Mixing a cu13 torch into Path A adds an unnecessary variable; 1.3 says
it would work, but there is no reason to spend the risk here.

---

## 5. Building the simulator

```bash
cd /workspace/diffCurricula/cl4ad
rm -f .kernel_cache                       # 1.4d — sm_89 artefact from the L4
rm -rf build && mkdir build && cd build

export CUDAToolkit_ROOT=/usr/local/cuda-12.4
export PATH=/usr/local/cuda-12.4/bin:$PATH

uv run cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.4 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc

find external -type f -name "*.tar" -delete   # reclaims ~2 GB

# libcuda.so.1 is missing at build time without a GPU in the build container
ln -sf /usr/local/cuda-12.4/lib64/stubs/libcuda.so \
       /usr/local/cuda-12.4/lib64/stubs/libcuda.so.1
LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64/stubs:$LD_LIBRARY_PATH uv run make -j$(nproc)
rm -f /usr/local/cuda-12.4/lib64/stubs/libcuda.so.1
```

Confirm the ABI tag now matches the interpreter:

```bash
ls /workspace/diffCurricula/cl4ad/madrona_gpudrive.cpython-*.so
# expect cpython-311 under Path A, cpython-312 under Path B
```

**Path B differs only in dropping the two `cuda-12.4` exports and the
`-DCUDAToolkit_ROOT` / `-DCMAKE_CUDA_COMPILER` flags.** If the NVRTC device
compile then fails on `<cuda/std/*>`, that is 1.2 and the fix is a CCCL-3 port.

---

## 6. The parent repo

The sweep and evaluation scripts live in the **parent** repo, not in `cl4ad`
(`run_gpudrive_sweep.py` defaults to `--eval_script ../scripts/eval_heldout.py`
and `--mutate_script ../scripts/mutate_scenes_random.py`). Clone
`diffCurricula` with `cl4ad` inside it, exactly as on the source machine:

```
diffCurricula/
├── scripts/            eval_heldout.py, run_gpudrive_sweep.py,
│                       mutate_scenes_random.py, make_scene_subsets.py,
│                       bench_gpudrive_raw.py, eval_curve.py, salvage_sweep_evals.py
└── cl4ad/
    ├── .venv/          the 3.11 env from section 4
    ├── baselines/ppo/  ppo_pufferlib_cl4ad.py + config/
    └── data/
```

`diffCurricula`'s own `.venv` (Python 3.8, SAFE-SIM) is a **separate**
environment and is **not** needed for any GPUDrive experiment. Do not create it
on the server unless you are also moving the diffusion-mutation work — and if
you do, note that its Python 3.8 will not coexist with CUDA 13 torch either.

> Recurring failure on the source machine: invoking `.venv/bin/python` from the
> `diffCurricula` root (Python 3.8, SAFE-SIM) instead of `cl4ad/.venv/bin/python`,
> giving `ModuleNotFoundError: No module named 'gpudrive'`. Always run GPUDrive
> work with `cwd=cl4ad` and `cl4ad/.venv/bin/python`.

### 6.1 Data

Stage `cl4ad/data/gpudrive_mini/{training,validation,testing}` (2.4 GB) by
rsync. Then regenerate the subsets on the server rather than copying symlink
trees:

```bash
cd /workspace/diffCurricula/cl4ad
.venv/bin/python ../scripts/make_scene_subsets.py \
  --root data/gpudrive_mini/training \
  --sizes 50 1000 --seeds 42 --out_dir data/subsets
```

This exists because `SceneDataLoader` truncates **before** it shuffles, so
`dataset_size=N` always takes the same first N files and neither
`shuffle_dataset` nor `seed` changes which. Different-N subsets are nested, and
different-seed subsets are genuinely different scenes.

---

## 7. Verification gate — run all four before any experiment

A green build proves nothing (1.2). These do.

### 7.1 The megakernel actually JITs on sm_90

```bash
cd /workspace/diffCurricula/cl4ad
time .venv/bin/python -c "
import gpudrive, torch
print('torch', torch.__version__, 'cuda', torch.version.cuda)
from gpudrive.env.config import EnvConfig
print('gpudrive import OK')
"
```

The **first** run pays a several-minute NVRTC compile and writes the kernel
cache; later runs start in seconds. If this dies with template errors mentioning
`cuda/std/...`, you have hit 1.2 — go back to Path A.

### 7.2 Raw simulator throughput

```bash
.venv/bin/python ../scripts/bench_gpudrive_raw.py
```

Reference from the L4 at `num_worlds=100`: **9,902 agent-steps/s** bare, and the
per-step `d2lt`/`d2la` the curriculum consumes costs **8.2%** of that loop
(~5.9% end-to-end). Record the H200 number here before choosing `num_worlds`
(section 9).

### 7.3 A short training run completes and checkpoints

```bash
cd /workspace/diffCurricula/cl4ad
.venv/bin/python ../scripts/run_gpudrive_sweep.py \
  --arms plr --sizes 50 --seeds 42 \
  --total_timesteps 2000000 --checkpoint_interval 5 \
  --heldout_dir data/gpudrive_mini/validation \
  --results_dir results/smoke --min_free_gb 10
```

Expect a `[done] plr_n50_s42 goal=... scored=...` line and
`results/smoke/plr_n50_s42.json`. A `[FAIL] ... produced no checkpoint` means
checkpoint discovery broke again — the trainer rewrites `exp_id` to
`<tag>__<mode>__R_<size>__<timestamp>` (`ppo_pufferlib_cl4ad.py:335`), so the
sweep globs `runs/<tag>__*`; anything that changes that naming breaks the glob.

### 7.4 Held-out evaluation uses the right split

```bash
.venv/bin/python ../scripts/eval_heldout.py \
  --model_cpt runs/<some-run>/model_*_000010.pt \
  --data_dir data/gpudrive_mini/validation \
  --config results/smoke/plr_n50_s42.yaml \
  --num_worlds 100 --max_scenes 100
```

> **`--data_dir` must be `validation`, never `testing`.** The testing split is
> not distribution-matched: median controlled goal distance is **6.05 m** there
> versus **28.80 m** (training) and **28.19 m** (validation). One checkpoint
> scores 97.4% on testing, 15.7% on training and 14.4% on validation. Reporting
> testing numbers would be meaningless.

Report `clean_success_rate` (goal ∧ no collision ∧ no off-road) as the headline;
`goal_achieved_rate` alone hides collisions.

---

## 8. Reproducing the L4 baseline first

Before spending H200 hours on new arms, re-run the one comparison whose answer
is already known. If the H200 disagrees with the shape below, the port is wrong.

```bash
cd /workspace/diffCurricula/cl4ad
.venv/bin/python ../scripts/run_gpudrive_sweep.py \
  --arms plr dr --sizes 1000 --seeds 42 \
  --total_timesteps 100000000 --checkpoint_interval 10 \
  --heldout_dir data/gpudrive_mini/validation \
  --results_dir results/repro_h200 --min_free_gb 10
```

L4 reference (N=1000, seed 42, 100M steps, 12 checkpoints each, validation):

| quantity | value |
|---|---|
| checkpoints where PLR leads | **10 of 12** (mean +0.035, median +0.020) |
| steps to 0.20 clean success | PLR **28.9M** vs DR **47.3M** → **1.64×** |
| AUC ratio | **1.15×** |
| endpoint at 100M | DR 0.5828 vs PLR 0.5338 |

The endpoint row is *why the protocol is curves and AUC*. Both learning curves
are staircases — DR plateaus 10M-38M, jumps at 47M, plateaus to 66M, jumps at
75M — so any single-checkpoint comparison reports where the staircase happened
to be, not which method is better. An earlier 30M-step comparison landed on the
first plateau and its 0.11-0.31 spread carried no signal at all.

> **Protocol: report curves and area under them, never endpoints. No run below
> ~50M steps.** Use `scripts/eval_curve.py` to recover a curve from a run's
> checkpoints.

---

## 9. Hyperparameters — what to change for the H200, and what not to

### 9.1 Do not expect an 8× speedup. The arithmetic:

| machine | worlds | agent-steps/s | source |
|---|---|---|---|
| RTX A5000 24 GB | 100 | **2,525** | paper Appendix D: 1B steps, >110 h |
| **L4 23 GB (ours)** | 100 | **4,700-4,900** | measured |
| H200 141 GB | 800 | **9,259** | paper Appendix D: 2B steps, ~60 h |

Our L4 is already **1.9× faster than the hardware the paper ran its own
ablation on**. And the paper's H200 figure, at **8× the worlds**, is only ~2×
our L4 number — so throughput is badly sublinear in `num_worlds`; the megakernel
and the PPO update, not world count, set the ceiling. **Plan for roughly 2×, so
~3 h per 100M-step run instead of ~5.7 h.** Measure with 7.2 before believing
any of this.

### 9.2 Keep the optimisation hyperparameters exactly as they are

`batch_size: 131_072`, `minibatch_size: 8192`, `learning_rate: 3e-4`,
`update_epochs: 4`, `ent_coef: 1e-4`, `vf_coef: 0.3` — all unchanged. Arms must
differ **only** in how scenes are chosen or what the pool contains; per-arm or
per-machine tuning would confound the exact comparison the sweep exists to make
(`run_gpudrive_sweep.py:build_config` enforces this).

### 9.3 If you raise `num_worlds`, you have started a new baseline

`batch_size` would have to scale with it or the PPO update starves, and that
changes the algorithm. Every arm would need re-running. Given 9.1 says the
return is ~2× at best, **the better use of 141 GB is concurrency, not width.**

### 9.4 Concurrency

One Madrona instance sizes its megakernel to the whole device, so two training
processes cannot share a GPU normally. On an H200 there are two ways out:

- **MIG**: slice into 3-4 instances and run one arm per slice at `num_worlds=100`
  each. This keeps every hyperparameter identical to the L4 configuration, so
  results stay comparable, and cuts Phase 3 wall-clock ~3×. Each slice reports
  its own SM count, which is what Madrona sizes against — but **verify with 7.1
  and 7.2 inside a slice before committing**; this has not been tested here.
- **Multiple GPUs**: one run per GPU, no caveats.

Under either, section 2.1 still holds: compare arms only against arms from the
same configuration.

---

## 10. Phase 3 — the experiment the migration is for

Generation vs selection under data scarcity: does adding generated variants help
*more* when real scenes are scarce?

```bash
cd /workspace/diffCurricula/cl4ad
nohup .venv/bin/python ../scripts/run_gpudrive_sweep.py \
  --arms plr random_mut --sizes 50 1000 --seeds 42 \
  --total_timesteps 100000000 --checkpoint_interval 10 \
  --heldout_dir data/gpudrive_mini/validation \
  --results_dir results/phase3 --min_free_gb 10 \
  --variants_per_scene 3 > phase3.log 2>&1 &
```

- `plr` = PLR over N real scenes (selection).
- `random_mut` = PLR over N real + 3 ACCEL-style variants each (generation).
  Variants shift each non-expert vehicle's whole logged track by (dx, dy) with a
  heading/speed perturbation and move its goal with it — verified to preserve
  goal distance to 0.000000 m while displacing positions by 2.64 m mean.
- 4 cells × ~3 h (H200 estimate) ≈ **12 h** on one GPU, ~4 h across 3 MIG slices.
- **One seed (42) per arm** — the paper's own first seed. Add seeds only once an
  effect is visible; multi-seed sweeps multiply wall-clock before answering the
  question that gates the project.
- Runs whose `results/phase3/<tag>.json` exists are skipped, so the sweep
  resumes after an interruption. Under section 2.1, **do not** copy the L4's
  `plr_n1000_s42.json` into this directory — that would silently skip the cell
  and mix machines.

**Decision criterion.** Not "does `random_mut` beat `plr`" — that would just be
ACCEL's result. The claim is that generation matters *specifically when data is
scarce*, so the test is whether the gap widens as N shrinks:

```
( random_mut − plr ) at N=50   >   ( random_mut − plr ) at N=1000
```

compared as AUC over the learning curve, not as endpoints. Equal gaps at both N
falsify the framing even if both are positive.

---

## 11. Not migrating: the SAFE-SIM guided-mutation work

`docs/guided_mutation_blocker.md` covers the diffusion-mutation arm. It needs
the Python 3.8 SAFE-SIM environment, nuScenes via trajdata, and the SAFE-SIM
checkpoint — a separate stack that shares nothing with GPUDrive. Current state:
the partial-diffusion injection chain is fixed and active, but variants are
still not adversarial (difficulty −41.05 against a −38.4 baseline), and the
identified fix is to generate variants by closed-loop rollout rather than a
single `sample()` call. **Leave it on the L4 box**; moving it would cost a day
and it is not on Phase 3's critical path.
