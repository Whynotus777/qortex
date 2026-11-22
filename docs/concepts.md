# Concepts

The API has five objects. Everything else is an action on one of them.

| Object | What it is | Created by |
|---|---|---|
| **Robot** | The body: type, onboard computer, power budget, observation and action shapes | `POST /v1/robots` |
| **Dataset** | Episodes of `(observation, action)` pairs for one robot | `POST /v1/datasets` |
| **Policy** | A trained, versioned model with lineage: base model, dataset, parent policy | `POST /v1/policies`, `POST /v1/policies/{id}/distill` |
| **Eval** | A sim run that produces a scorecard for one policy | `POST /v1/evals` (also runs automatically after training) |
| **Deployment** | A policy assigned to a robot, with the gate result recorded | `POST /v1/deployments` |

## Lineage

A distilled policy points at its teacher through `parent_policy_id`, and both point at the dataset and base model. A deployment stores the scorecard it was approved on. You can always answer "what is running on this robot and why was it allowed there".

## Jobs

Dataset generation, fine-tuning, distillation and evals run as jobs. Each call returns a `job_id`. Poll `GET /v1/jobs/{id}?wait=30`, or let the SDK wait for you (it does by default).

## Actions and observations

Actions are always normalised to `[-1, 1]`. Each robot type documents how to scale them. Observations are flat float vectors; the robot's `obs_spec` field says what each entry means.

## Control rate

The built-in sim tasks run at 20 Hz. Run the real control loop at the same rate.
