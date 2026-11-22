# Model catalog

!!! warning "Read this first"
    Specs for third-party models are approximate figures from public sources and have NOT been benchmarked by Qortex. Only entries with evidence=measured have been run through the Qortex sim suite. Verify licences before commercial use.

The same data is served by `GET /v1/models` and drives [`/v1/recommend`](router.md).

| Model | Access | Params | Inference memory | Language | Trainable here | Evidence |
|---|---|---|---|---|---|---|
| [`qortex/mlp-bc`](#qortexmlp-bc) | open-weights | 70k | ~0.01 GB | no | yes | measured |
| [`lerobot/act`](#lerobotact) | open-weights | 0.08 B | ~1.5 GB | no | no | published-specs |
| [`lerobot/smolvla`](#lerobotsmolvla) | open-weights | 0.45 B | ~3 GB | yes | no | published-specs |
| [`openvla/openvla-7b`](#openvlaopenvla-7b) | open-weights | 7.0 B | ~16 GB | yes | no | published-specs |
| [`physical-intelligence/openpi-pi0`](#physical-intelligenceopenpi-pi0) | open-weights | 3.3 B | ~8 GB | yes | no | published-specs |
| [`nvidia/gr00t-n1.7`](#nvidiagr00t-n17) | open-weights | 3.0 B | ~16 GB | yes | no | published-specs |
| [`physical-intelligence/pi-api`](#physical-intelligencepi-api) | external-api | n/a | vendor-hosted | yes | no | published-specs |
| [`dream-labs/dream-1`](#dream-labsdream-1) | external-api | n/a | vendor-hosted | yes | no | published-specs |

## qortex/mlp-bc

**Qortex MLP-BC** from Qortex. Licence: Apache-2.0.

Small state-based behavior-cloning policy. Trains in seconds on CPU, distills to about 1.5k parameters, runs on any computer including a Raspberry Pi. No vision, no language.

Robot types: `any`.

Caveats:

- State observations only: no camera input
- One task per policy

## lerobot/act

**ACT (Action Chunking Transformer)** from Open source (LeRobot implementation). Licence: MIT (verify per checkpoint).

Compact vision policy for one precise manipulation task learned from a few dozen demonstrations. Usually runs on-robot without distillation.

Robot types: `arm-2dof`, `arm`, `bimanual`, `custom`.

Caveats:

- No language conditioning
- Needs a GPU worker to train

## lerobot/smolvla

**SmolVLA** from Hugging Face. Licence: Apache-2.0.

Small language-conditioned VLA designed for consumer hardware. A reasonable on-robot default for Jetson Orin NX class boards, and a common distillation student.

Robot types: `arm-2dof`, `arm`, `bimanual`, `mobile-base`, `custom`.

Caveats:

- Trained mostly on low-cost arm data
- Needs a GPU worker to fine-tune

## openvla/openvla-7b

**OpenVLA 7B** from Open source (Stanford, Berkeley and others). Licence: MIT.

Large open VLA with LoRA fine-tuning recipes. Too slow for most on-robot use without compression, so it fits best as a cloud teacher.

Robot types: `arm-2dof`, `arm`, `custom`.

Caveats:

- Single-arm focus
- Low control rate without action chunking

## physical-intelligence/openpi-pi0

**pi0 (openpi)** from Physical Intelligence. Licence: Apache-2.0 (verify per checkpoint).

Strong open-weights flow-matching VLA for dexterous manipulation. Good teacher: fine-tune in the cloud, then distill for the robot.

Robot types: `arm-2dof`, `arm`, `bimanual`, `mobile-base`, `custom`.

Caveats:

- Full fine-tune needs about 70 GB of GPU memory, LoRA about 22.5 GB
- Needs a GPU worker

## nvidia/gr00t-n1.7

**Isaac GR00T N1.7** from NVIDIA. Licence: NVIDIA open model licence (verify commercial terms).

Open VLA aimed at humanoids, with a slow reasoning system and a fast action system. Targets Jetson Thor class hardware.

Robot types: `humanoid`, `bimanual`, `arm`, `custom`.

Caveats:

- Parameter count here is an estimate carried over from N1.5
- Needs a GPU worker

## physical-intelligence/pi-api

**Pi API (hosted)** from Physical Intelligence. Licence: Commercial API terms.

Vendor-hosted fine-tuning and inference on Physical Intelligence's latest models. Beta.

Robot types: `arm-2dof`, `arm`, `bimanual`, `mobile-base`, `humanoid`, `custom`.

Caveats:

- Cloud only: inference runs on the vendor's servers
- Not integrated: Qortex only points you to it
- Weights are not exported, so no on-robot distillation

## dream-labs/dream-1

**Dream-1 (Dream API)** from Dream Labs. Licence: Commercial API terms.

Cross-embodiment model sold as a cloud API: camera frame plus instruction in, motor commands out. A cheaper flash variant is listed at 2 USD per hour.

Listed price: 3.0 USD per hour.

Robot types: `arm-2dof`, `arm`, `bimanual`, `mobile-base`, `humanoid`, `custom`.

Caveats:

- Cloud only
- Not integrated: Qortex only points you to it
- Network latency limits control rate

## Compute targets

Memory and power figures are approximate.

| Id | Device | Memory | Max power |
|---|---|---|---|
| `raspberry-pi-5` | Raspberry Pi 5 (8 GB) | 8 GB | 10 W |
| `laptop-cpu` | Laptop CPU | 16 GB | 30 W |
| `jetson-orin-nano` | Jetson Orin Nano (8 GB) | 8 GB | 25 W |
| `jetson-orin-nx-16` | Jetson Orin NX (16 GB) | 16 GB | 40 W |
| `jetson-agx-orin-64` | Jetson AGX Orin (64 GB) | 64 GB | 60 W |
| `jetson-thor` | Jetson Thor | 128 GB | 130 W |
| `rtx-4090` | RTX 4090 workstation | 24 GB | 450 W |
| `rtx-5090` | RTX 5090 workstation | 32 GB | 575 W |
| `cloud-h100` | Cloud H100 (80 GB) | 80 GB | n/a |
