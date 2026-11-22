# Distilling for the robot

Distillation trains a small student to copy a larger teacher, so the student fits the robot's onboard computer.

```python
student = qx.distill(teacher_id, hidden=(32, 32), quantize="int8")
```

| Parameter | Default | Meaning |
|---|---|---|
| `hidden` | `[32, 32]` | Student hidden layer sizes |
| `quantize` | `int8` | Store weights as 8-bit integers. `none` keeps float32 |
| `epochs` | 60 | Training passes |

The student is trained on the dataset's observations plus jittered copies, all labelled by the teacher. It then gets its own sim eval, so you compare teacher and student on the same task:

```python
print(qx.policy(teacher_id)["latest_eval"]["success_rate"])
print(student["latest_eval"]["success_rate"], student["training"]["compression"])
```

In our end-to-end test a 69k-parameter teacher distills to about 1.5k parameters (a 4 KB file) and keeps 96 to 100% sim success on the three built-in tasks.

!!! note
    Today this works for the built-in policy family. Distilling a large VLA into a small VLA uses the same API shape but needs a GPU worker backend that is not built yet.
