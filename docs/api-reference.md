# API reference

Base URL: your server, for example `https://YOUR-HOST`. All endpoints are under `/v1`.

An interactive version of this page, generated from the same schema, is at [`/reference`](/reference).

## Authentication

Send your key as a bearer token on every request except `POST /v1/keys`, `GET /v1/models` and `GET /v1/devices`.

```bash
curl $QORTEX_URL/v1/robots -H "Authorization: Bearer $QORTEX_API_KEY"
```

## Errors

Errors return JSON `{"error": "message"}` with these status codes:

| Code | Meaning |
|---|---|
| 401 | Missing or invalid API key |
| 404 | Object not found, or it belongs to another key |
| 409 | Object is not in the right state yet, for example a policy still training |
| 412 | Deployment blocked by the eval gate |
| 422 | Invalid request body |

## Auth

### Create an API key

`POST /v1/keys`

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `email` | string | yes |  |  |

## Router

### List the model catalog

`GET /v1/models`

### List known compute targets

`GET /v1/devices`

### Rank models for a robot, task and constraints

`POST /v1/recommend`

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `robot_id` | string/null | no |  |  |
| `robot` | object/null | no |  | Inline robot spec if you have not registered one: {type, compute, power_budget_w} |
| `task` | object: `kind`, `description`, `needs_vision` | no | `{'kind': 'single_task', 'description': '', 'needs_vision': False}` |  |
| `constraints` | object: `deploy`, `min_hz`, `max_power_w`, `commercial`, `allow_paid` | no | `{'deploy': 'on_robot', 'min_hz': 10.0, 'commercial': True, 'allow_paid': True}` |  |

## Robots

### Register a robot

`POST /v1/robots`

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | string | yes |  |  |
| `type` | string | yes |  | arm-2dof | mobile-base | drone | custom |
| `compute` | string/null | no |  | A device id from GET /v1/devices |
| `power_budget_w` | number/null | no |  |  |
| `obs_dim` | integer/null | no |  | Required for type=custom |
| `action_dim` | integer/null | no |  | Required for type=custom |
| `notes` | string | no |  |  |

### List robots

`GET /v1/robots`

### Get a robot

`GET /v1/robots/{id}`

## Datasets

### Create an empty dataset

`POST /v1/datasets`

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `robot_id` | string | yes |  |  |
| `name` | string | no | `dataset` |  |

### Upload recorded episodes

`POST /v1/datasets/{id}/episodes`

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `episodes` | array | yes |  |  |

### Generate demonstrations in sim

`POST /v1/datasets/{id}/generate`

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `episodes` | integer | no | `150` |  |
| `noise` | number | no | `0.1` | Std of noise added to executed actions, to teach recovery |
| `seed` | integer | no | `0` |  |

### Get a dataset

`GET /v1/datasets/{id}`

## Policies

### Fine-tune a policy from a base model

`POST /v1/policies`

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `robot_id` | string | yes |  |  |
| `dataset_id` | string | yes |  |  |
| `base_model` | string | no | `qortex/mlp-bc` |  |
| `name` | string | no | `policy` |  |
| `hidden` | array | no | `[256, 256]` |  |
| `epochs` | integer | no | `40` |  |
| `auto_eval` | boolean | no | `True` |  |

### List policies

`GET /v1/policies`

Query parameters: `robot_id`.

### Distill a smaller student policy

`POST /v1/policies/{id}/distill`

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `hidden` | array | no | `[32, 32]` | Student hidden layer sizes |
| `quantize` | string | no | `int8` | int8 | none |
| `epochs` | integer | no | `60` |  |
| `auto_eval` | boolean | no | `True` |  |

### Get a policy and its lineage

`GET /v1/policies/{id}`

### Download the policy artifact (.npz)

`GET /v1/policies/{id}/artifact`

### Hosted inference: observation in, action out

`POST /v1/policies/{id}/act`

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `observation` | array | yes |  |  |

## Evals

### Evaluate a policy in sim

`POST /v1/evals`

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `policy_id` | string | yes |  |  |
| `episodes` | integer | no | `50` |  |
| `seed` | integer | no | `10000` |  |

### Get an eval and its scorecard

`GET /v1/evals/{id}`

## Deployments

### Deploy a policy to a robot, gated on its eval

`POST /v1/deployments`

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `policy_id` | string | yes |  |  |
| `robot_id` | string | yes |  |  |
| `target` | string | no | `on_robot` | on_robot | cloud |
| `gate` | object: `min_success_rate`, `max_action_mse` | no | `{'min_success_rate': 0.8}` |  |
| `force` | boolean | no | `False` | Deploy even if the gate fails. Recorded on the deployment. |

### Current deployment for a robot (polled by the runtime)

`GET /v1/robots/{id}/deployment`

### Roll back to the previous deployment

`POST /v1/robots/{id}/rollback`

### Deployment history for a robot

`GET /v1/robots/{id}/deployments`

## Telemetry

### Report runtime stats from the robot

`POST /v1/robots/{id}/telemetry`

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `policy_id` | string/null | no |  |  |
| `deployment_id` | string/null | no |  |  |
| `steps` | integer | no | `0` |  |
| `loop_hz` | number/null | no |  |  |
| `inference_ms_p50` | number/null | no |  |  |
| `inference_ms_p99` | number/null | no |  |  |
| `episodes` | integer/null | no |  |  |
| `successes` | integer/null | no |  |  |
| `extra` | object | no | `{}` |  |

### Recent telemetry for a robot

`GET /v1/robots/{id}/telemetry`

Query parameters: `limit`.

## Jobs

### Get a job. Use ?wait=30 to long-poll until it finishes

`GET /v1/jobs/{id}`

Query parameters: `wait`.

