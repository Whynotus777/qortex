# Sim evals and the deploy gate

Every policy gets a closed-loop sim eval as soon as training finishes. You can run more:

```python
ev = qx.evaluate(policy_id, episodes=200)
print(ev["scorecard"])
```

```json
{
  "mode": "sim",
  "episodes": 200,
  "success_rate": 0.97,
  "mean_final_distance_m": 0.012,
  "latency_ms_p50": 0.02,
  "latency_ms_p99": 0.05,
  "max_control_hz_on_eval_host": 20000,
  "success_tolerance_m": 0.03,
  "n_params": 1474,
  "quantize": "int8"
}
```

Evals use seeds that were not used to generate training data.

`latency_ms_*` is measured on the eval host's CPU. It is not a measurement on your robot's computer. The runtime reports real on-robot latency through telemetry.

## The gate

```python
qx.deploy(policy_id, robot_id, min_success_rate=0.9)
```

- The policy's latest scorecard must meet the gate, or the call returns HTTP 412 and nothing is deployed.
- `force=True` overrides the gate. The deployment records `forced: true`.
- The deployment stores the scorecard it was approved on.

## Sim tasks

| Robot type | Task | Success means |
|---|---|---|
| `arm-2dof` | Planar two-joint arm reaches a random target | Tip within 3 cm at the end of the episode |
| `mobile-base` | Holonomic base drives to a random goal | Within 8 cm |
| `drone` | Vehicle flies to a random 3D waypoint by velocity setpoints | Within 10 cm |

These tasks are small on purpose so the pipeline runs on a CPU in seconds. Passing one says the pipeline works and the policy does that task in sim. It does not say the policy is safe on your hardware.
