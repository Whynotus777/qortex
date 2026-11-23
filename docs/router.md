# Choosing a model

`POST /v1/recommend` ranks every model in the [catalog](models.md) for your robot, task and constraints.

```python
rec = qx.recommend(
    robot={"type": "arm", "compute": "jetson-orin-nx-16"},      # or robot_id=...
    task={"kind": "language", "needs_vision": True},
    constraints={"deploy": "on_robot", "min_hz": 5, "allow_paid": False},
)
```

## Inputs

| Field | Values | Effect |
|---|---|---|
| `task.kind` | `single_task`, `multi_task`, `language`, `teacher` | What you need the model to be good at. `teacher` means a cloud model you will distill from. |
| `task.needs_vision` | bool | Drops models with no camera input |
| `constraints.deploy` | `on_robot`, `on_prem`, `cloud` | Vendor-hosted APIs are only eligible for `cloud` |
| `constraints.min_hz` | number | Target control rate |
| `constraints.max_power_w` | number | Flags compute boards that exceed it. Defaults to the robot's `power_budget_w` |
| `constraints.commercial` | bool | Drops models whose licence forbids commercial use |
| `constraints.allow_paid` | bool | Drops paid APIs |

## Output

Each recommendation has:

- `fit`: `direct`, `needs_quantization`, `needs_distillation`, `on_prem_gpu`, `cloud` or `cloud_teacher`
- `evidence`: `measured` if this account has a closed-loop sim scorecard for that base model on that robot type, `offline-check` if it only has an offline eval (custom robots), otherwise `published-specs`
- `estimated_hz`: a rough figure from parameter count and device class. It is an estimate, not a benchmark.
- `why` and `caveats`: the reasons behind the score
- `runnable_here`: whether this platform can train the model today

The response also lists `excluded` models with the reason, and a `suggested_pipeline` such as "fine-tune pi0 in the cloud as teacher, distill into SmolVLA, evaluate, deploy".

The suggested pipeline is the best plan on paper. It is not always one this platform can run today, so check:

- `pipeline_executable_here`: `true` only if every model in the pipeline can be trained here now.
- `pipeline_blocked_by`: the models that cannot, with the reason.
- `executable_alternative`: the best plan that does run here today, with its trade-offs, or `null` with a note when nothing runnable meets the requirements.

!!! warning "How much to trust the ranking"
    Only `qortex/mlp-bc` has been run through the sim suite. Every other entry is scored from public specs. Treat those ranks as a shortlist to test, not a verdict.
