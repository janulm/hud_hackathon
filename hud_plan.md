# HUD Integration Plan — Real SO101 Gold-Bar → Jar (no sim)

We are committing to the **physical robot only**. We already have a real SO101, real
teleop data, and an ACT policy; crossing the sim2real gap is the biggest risk, so we cut
sim entirely. HUD's job here is **live tracking, scoring, dataset recording, and the
trace viewer** wrapped around the real arm — not simulation.

Companion to `PLAN.md`. This doc is the HUD-specific build + reasoning.

---

## 0. TL;DR

- Run our **ACT** policy on the real SO101 inside HUD's `robot` capability: a thin
  **`RobotBridge` that wraps LeRobot's SO101 hardware driver** (motors + cameras).
- **HUD's payoff:** automatic **per-episode telemetry → the hud.ai web trace viewer**
  (cameras + state/action channels + inference ticks), per-episode **scoring** from our
  own gold-bar-in-jar detector, and optional **LeRobot v3 dataset recording** for the
  retrain loop.
- **Latency:** the hud.ai platform is **not** in the control loop. Telemetry uploads
  out-of-band; recording buffers locally and finalizes at exit. The only hot-path cost is
  the **local** agent↔bridge WebSocket — keep them co-located and run the policy locally.
- **Honest scope:** `lerobot-record --policy.path` already deploys + records on hardware.
  HUD earns its place via the **trace viewer + scoring + one unified harness**, not by
  re-implementing hardware I/O (the bridge delegates that to LeRobot).

---

## 1. Basics of HUD (the mental model)

HUD is an agent **evaluation + training** platform. Four nouns:

- **Environment** — a reset-and-reproduce world the agent acts in. The docs are explicit
  that the env side *"translates incoming actions into changes in the digital **or
  physical** environment and serves observations"* — so **a real SO101 is a first-class
  HUD environment**, not a hack.
- **Task** — a unit of work with a prompt and a **grader** (the reward).
- **Capability** — how the agent talks to the env. For robots it's the **`robot`
  capability**: a continuous observe→act loop over **WebSocket** (wire format `openpi/0`,
  msgpack + numpy), because a high-rate policy can't stream actions as tool calls.
- **Agent** = a **model** (our ACT checkpoint) + a **harness** (the loop that drives it).
  HUD provides the harness (`RobotAgent`); we supply the model + a small adapter.

Each run produces a **Run/Trace** — every frame, action, and score — replayable and
gradable. That's the loop: environment → task → agent acts → reward → (evaluate / train).

### The robot pieces we touch
- **`RobotBridge`** (env side) — one class, three methods (`reset` / `step` /
  `get_observation`) plus `result()`. Framework owns the serve loop, wire codec, and
  control-rate stepping. **Here it wraps the physical SO101.**
- **Contract** — one shared JSON (control rate + observation/action feature names). Wires
  the policy to the env *and* defines the recorded-dataset schema. No other shared config.
- **`RobotAgent` + `Model` + `Adapter`** — our ACT policy. ACT is a stock LeRobot policy,
  so this is ~15 lines via the bundled `LeRobotModel` + `LeRobotAdapter`.

---

## 2. The web trace viewer (the centerpiece)

Lives on the platform at **hud.ai**, and it's automatic: with `HUD_API_KEY` set, **every
episode streams to the browser trace viewer** — no extra wiring. When a run finishes, open
hud.ai, click the job, and the viewer replays the episode end to end:

- **Camera feeds** scrubbed on a timeline (overhead + wrist).
- **State & action channels** you can toggle (the 6 joints + gripper).
- **Inference ticks** — when ACT predicted each action chunk and for which steps.

Scrub between ticks to watch a chunk execute; click a tick to jump to the decision point.
This is our **curation tool** (decide which autonomous rollouts to keep) and our **demo**
(real arm beside its HUD replay). Telemetry streams whether or not we also record a
dataset — recording is the opt-in extra.

---

## 3. Architecture — real-robot-only

```
   ACT checkpoint (local GPU)                     physical world
        │                                               │
   RobotAgent(ACT)  ──openpi/0 WebSocket (LOCAL)──▶  SO101RealBridge
   LeRobotModel + Ensembler                            reset()  → home arm, prompt
   LeRobotAdapter   ◀──── observations ──────────       step()  → write 6 joint targets
        │                                               get_observation() → cams + joints
        │                                               result() → CV gold-bar-in-jar score
        ▼
   HUD_API_KEY ───────▶ hud.ai web trace viewer (async, out-of-band)
   agent.save=True ───▶ LeRobot v3 dataset (RECORD_DIR, local) ──▶ filter ──▶ retrain
```

**Key synthesis:** the HUD bridge **delegates motor/camera I/O to LeRobot's
`SO101Follower`**. LeRobot owns hardware; HUD owns the loop, telemetry, trace, scoring,
and dataset. Minimal new code, no reinvented drivers.

---

## 4. Control loop & latency (does HUD break a high-rate loop? No.)

There are **two separate data flows** — do not conflate them:

1. **Control loop** — `observe → infer → act` between the agent and the bridge over the
   `openpi/0` WebSocket. Latency-critical, lockstep (the bridge steps once per action).
2. **Telemetry / trace upload** — frames + actions + scores up to hud.ai. **Out-of-band.**

What this means:

- **The hud.ai platform is NOT in the control loop.** Per the docs, telemetry "streams
  either way" and recording is buffered locally and "finalized at process exit" — so
  server/network delay to the platform does **not** sit between camera read and motor
  write. *(Verify on the installed `hud-python` build that telemetry emission is
  queued/non-blocking, not an awaited send per step — a quick local timing test confirms.)*
- **The hot path is the LOCAL agent↔bridge WebSocket.** Keep them co-located (loopback or
  same LAN) → sub-ms to low-ms hop, fine for a 16.7 ms (60 Hz) budget. **Run ACT locally;
  do NOT use a cloud GPU (`--remote`) for a real-time real-arm loop** — that puts WAN
  latency inside the loop. (`--remote` is for sim throughput, not hardware real-time.)
- **The real 60 Hz constraints are the usual robotics ones, not HUD telemetry:**
  - **Inference** > 16 ms, so we rely on **action chunking** — ACT predicts a chunk of T
    actions, the harness plays them out and re-infers only when spent (use an `Ensembler`).
  - **Per-tick frame serialization** is the one place the local WebSocket adds cost
    (msgpack of camera frames every tick). Two 640×480 RGB frames ≈ 1.8 MB; at 60 Hz that's
    ~110 MB/s over loopback — jitter risk. Mitigate: send **smaller frames** (policy
    resizes to ~224 anyway) and/or run the loop at **30 Hz** (likely our recording rate —
    confirm against the dataset).

### Live vs. post — three options
| Option | HUD in the loop? | Get | Cost |
|---|---|---|---|
| **Live, telemetry on** (recommended) | agent↔bridge local hop only | live streaming + scoring + recording | per-tick frame serialize |
| **Live, telemetry off** | local hop only, no `HUD_API_KEY` | local control + local dataset; upload trace after | none beyond the hop |
| **Post-hoc** | none | `lerobot-record --policy.path` at native rate, then replay into HUD | no *live* streaming |

**Recommendation:** run the policy local, bridge co-located, telemetry on (background), at
our recorded control rate (likely 30 Hz). If a loop-timing test shows frame-serialize
jitter, drop frame resolution or fall back to post-hoc — the trace viewer + scoring work
identically either way.

---

## 5. What we build

A small package (copy `robot-template/inventory/envs/scaffold/` + `agents/scaffold/`):

### 5.1 `so101_joint_pos.json` — the contract (NEW)
```json
{
  "robot_type": "so101",
  "control_rate": 30,
  "features": {
    "observation/overhead_image": { "role": "observation", "type": "rgb",
      "names": ["height", "width", "channel"], "shape": [480, 640, 3] },
    "observation/wrist_image":     { "role": "observation", "type": "rgb",
      "names": ["height", "width", "channel"], "shape": [480, 640, 3] },
    "observation/state": { "role": "observation", "type": "joint_pos",
      "names": ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"],
      "shape": [6] },
    "action": { "role": "action", "type": "joint_pos",
      "names": ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"],
      "shape": [6] }
  }
}
```
- `control_rate` **must equal the fps our ACT data was recorded at** (agent uses it as the
  record fps + trace playback speed).
- `type:"rgb"` on images is load-bearing (how the adapter spots a camera vs. the state).
- Feature keys are verbatim what the bridge returns *and* the recorded dataset columns:
  `observation/overhead_image` → `observation.images.overhead_image`, etc.

### 5.2 `so101_real_bridge.py` — the bridge (wraps LeRobot hardware)
```python
from hud.environment.robot import RobotBridge, ThreadSimRunner

class SO101RealBridge(RobotBridge):
    def __init__(self, robot, scorer, **kw):
        super().__init__(sim_runner=ThreadSimRunner(), **kw)  # blocking serial/cam off the loop
        self._robot = robot       # lerobot SO101Follower (owns motor bus + cameras)
        self._scorer = scorer     # gold-bar-in-jar CV detector

    async def reset(self, task_id="pick_goldbar", seed=0):
        self._robot.send_action(HOME_POSE)        # home; optionally wait for human to place bar
        self._scorer.reset()
        self.task_description = "pick up the gold bar and put it in the jar"
        return self.task_description

    def step(self, action):                        # action: 6-D joint targets from ACT
        self._robot.send_action(dict(zip(JOINT_NAMES, action)))
        score, success = self._scorer.update(self._last_overhead)  # CV on latest frame
        self.total_reward, self.success = score, success
        self.terminated = success or self._step >= MAX_STEPS

    def get_observation(self):
        obs = self._robot.get_observation()        # joints + camera frames
        self._last_overhead = obs["overhead"]
        data = {
            "observation/overhead_image": obs["overhead"],   # HWC uint8
            "observation/wrist_image":    obs["wrist"],       # HWC uint8
            "observation/state": joints6(obs).astype("float32"),
        }
        return data, self.terminated

    def result(self):
        return {"score": self._scorer.score, "success": bool(self.success),
                "total_reward": float(self.total_reward)}
```
*(Import path for `SO101Follower` and the camera/obs key names to confirm against our
installed LeRobot + how we recorded.)*

### 5.3 `so101_act_agent.py` — the policy (~15 lines)
```python
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from hud.agents.robot import RobotAgent, LeRobotModel, LeRobotAdapter

class SO101ACTAgent(RobotAgent):
    max_steps = 400
    save = True   # record a LeRobot v3 dataset (needs: pip install 'lerobot[dataset]')
    def __init__(self, checkpoint, device="cuda"):
        policy = ACTPolicy.from_pretrained(checkpoint).to(device).eval()
        pre, post = make_pre_post_processors(policy.config, checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}})
        self.model = LeRobotModel(policy, pre, post)   # + Ensembler (ACT is chunked)
        self.adapter = LeRobotAdapter(model_image_keys=list(policy.config.image_features))
```

### 5.4 `gold_bar_scorer.py` — the reward (NEW; build now, no model needed)
Fixed jar → calibrated pixel ROI. Per frame: gold/yellow **HSV mask** → centroid + area.
- **success** = centroid inside jar ROI **AND** area ≥ min **AND** `was_lifted` (bar seen
  above table height earlier in the episode → rules out "shoved over").
- **score** (shaping) = `0.5*lift_progress + 0.5*float(success)`.
- Optional **VLM keyframe audit** on the final frame ("is the gold bar in the jar?") as a
  tie-breaker — not per-frame (cost/latency/spatial flakiness).

### 5.5 `env.py` + runner
`RobotEndpoint(SO101RealBridge(...))`, publish the capability with the contract, one
`pick_goldbar` task; a small runner that builds the agent and calls `Taskset(...).run(...)`.

### 5.6 (optional) `loop_timing.py`
Logs per-tick `get_observation` / `infer` / `send_action` / serialize latency so we can
**measure live-vs-post on the real rig** before committing.

---

## 6. Recording — how HUD writes the dataset

- Set `agent.save = True`. HUD records every `(observation, executed action)` tick into a
  **LeRobot v3 dataset, agent-side** (in our process; only our machine needs
  `pip install 'lerobot[dataset]'`).
- **The contract drives the schema** with no extra wiring: image features →
  `observation.images.<name>` (per-episode video), the state vector → `observation.state`,
  the action → `action`, the prompt → each frame's `task`.
- Env vars: `RECORD_DIR` (root, default `./data`), `HF_REPO` (+ `HF_TOKEN` to push),
  `HF_PRIVATE`. *(The `robot-template` README calls these `HUD_RECORD_DIR`/`HUD_HF_REPO` —
  verify which the installed build expects.)*
- **One dataset per run, finalized at exit.** For filtering, label episodes by their
  `result()` score and drop the failures by index (v3 is episode-aware: `meta/episodes/`),
  or run scoring in per-batch processes.

---

## 7. The improvement loop (real arm)

ACT is at ~15% success. Pure **success-filtered BC** on autonomous rollouts is low-yield
(≈200 rollouts with manual resets for ~30 successes) and clones the policy's own lucky
trajectories. So:

- **Prefer HG-DAgger (human-gated intervention):** let the policy run; the moment it's
  about to fail, take over with the leader arm and demonstrate the recovery; record the
  correction. Targets the policy's weak states — far more sample-efficient.
- **Combine:** autonomous rollouts (HUD-scored by the CV detector) for easy wins + human
  takeover on failures. Reward-*weight* near-misses instead of hard filtering.
- Retrain ACT co-trained on real teleop demos + curated rollouts/corrections; repeat;
  watch real eval for regressions.
- HUD's role in the loop: record the rollouts (or use `lerobot-record`), **auto-score**
  each episode, and **curate in the trace viewer** before retraining.

---

## 8. Gotchas (priority order)

1. **Run the policy LOCAL; bridge co-located.** No cloud GPU in a real-time loop.
2. **Match `control_rate` to the recorded fps** (likely 30 Hz; 60 Hz is aggressive for a
   camera-in-the-loop policy — confirm the dataset rate).
3. **Frame serialization at high rate** — shrink frames / lower rate if the timing harness
   shows jitter.
4. **Safety** — joint limits in `send_action`, an e-stop, bounded `reset()` motion: this is
   real hardware running autonomously.
5. **Env-var/API drift** (`agent.save`/`RECORD_DIR` vs README's `HUD_RECORD_DIR`; the
   `SO101Follower` import path) — verify against installed `hud-python` + `lerobot`.
6. **`ThreadSimRunner`** so blocking serial/camera reads don't stall the asyncio loop.

---

## 9. Build order (while the checkpoint finishes / hardware is free)

1. `gold_bar_scorer.py` + **jar-ROI calibration** on a few captured overhead frames (no
   robot needed).
2. `so101_joint_pos.json` (set `control_rate` from the dataset).
3. `so101_real_bridge.py` wrapping `SO101Follower` — prove `reset/step/get_observation`
   drive the arm + return frames.
4. `so101_act_agent.py` + runner; **NoopAgent** smoke test → confirm a trace appears on
   hud.ai and `agent.save=True` writes a valid v3 dataset.
5. `loop_timing.py` → decide **live vs. post** from real numbers.
6. First scored autonomous rollouts → curate in the trace viewer → start the HG-DAgger loop.

---

## 10. Risk register

| Risk | Mitigation / fallback |
|---|---|
| Latency breaks the loop | Local policy + co-located bridge; telemetry is async; timing harness; drop res / 30 Hz; else post-hoc |
| Telemetry secretly blocking | Verify queued/non-blocking on installed build; else telemetry-off + upload after |
| Filtered BC shows no gain at 15% | HG-DAgger interventions + reward-weighting + more teleop demos |
| `SO101Follower`/env-var API drift | Confirm imports + record vars against installed versions; NoopAgent smoke test |
| HUD adds little over lerobot on real | Accept HUD = trace viewer + scoring + unified harness; that's the demo value |
| Autonomous run unsafe | Joint limits + e-stop + bounded reset; human supervises |
