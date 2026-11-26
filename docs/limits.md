# Status and limits

This is version 0.1. It proves the pipeline end to end. Here is what it does not do.

## Models

- The only model family that trains inside the API is `qortex/mlp-bc`: small state-based behavior-cloning networks. No vision, no language.
- ACT trains on a [GPU worker you connect](gpu-worker.md). That path is tested against a stand-in trainer only and has not run on a real GPU. ACT policies have no evaluation here (training loss only), so the deploy gate never passes them without `force`, and the Qortex runtime cannot run them.
- SmolVLA, pi0, GR00T and OpenVLA are in the catalog and the router. Fine-tune jobs for them are accepted and then wait, because no GPU worker backend exists yet.
- Pi API and Dream API are listed for comparison only. Qortex does not call them.

## Router

- Third-party model figures (parameters, memory, licences, prices) are approximate and taken from public sources. They have not been verified or benchmarked here.
- `estimated_hz` is a rough heuristic.
- Check each model's licence yourself before commercial use.

## Sim

- Three small tasks. They are kinematic or near-kinematic and have no obstacles, sensor noise, latency or contact.
- The drone task does not simulate rotors or attitude.
- You can [upload your own MuJoCo model](custom-sim.md), but only as one self-contained MJCF file with primitive geoms (no meshes), and only for reach-a-goal tasks. URDF is not accepted.
- Custom robots without an uploaded sim are validated offline on held-out episodes only.
- Eval latency is measured on the eval host, not on your robot's computer.

## Runtime

- Python only. No ROS 2 node, no C++ runtime.
- No safety layer. The policy output is clipped to `[-1, 1]` and nothing else. Velocity limits, geofences, collision checks and e-stops are your responsibility.
- The hardware adapters in these docs have not been run on physical robots.

## Platform

- On Vercel, jobs run inside the request, so each one must finish within 300 seconds. That is fine for the built-in policies and rules out GPU-scale training there.
- Updates are read-modify-write with no row locking. Two simultaneous writes to the same object can lose one.
- Open signup, no rate limits, no billing.
- No dashboard. Everything is API and SDK.
