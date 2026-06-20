# SO101 Gold-Bar Pickup: Sim-Improved Imitation Learning (24h Hackathon Plan)

## TL;DR

Train an imitation policy on **real teleop data** (safety net), bring up a **matched SO101 sim** with a Gizmo gold-bar scene + reward, then run a **reward-filtered self-improvement loop** in sim (rollout → score with HUD → keep the good episodes → retrain, co-trained with real data + domain randomization), and finally **deploy back to the real arm**. We ship something working at the end of Phase 1 no matter what; everything after that improves robustness and makes the demo.

- **Models:** ACT (fast workhorse + safety deploy + loop iteration) → SmolVLA base (hero/robust, leverages SO101 pretraining).
- **Substrate:** Everything is LeRobot v3 datasets + checkpoints, so sim and real share one format.
- **Key risk to manage:** the real↔sim *visual* gap. Anchor with a little sim teleop data + Gizmo domain randomization; always co-train with real.

---

## 1. Assets & setup

| Thing | Detail |
|---|---|
| Arms | SO101 leader + follower (calibrated, can teleop) |
| Cameras | Overhead/front static = `camera1` (**critical**), wrist = `camera2` (robustness). Keep the same layout + viewpoint across real / sim / deploy. |
| Object | Real gold bar + a matching sim asset (Gizmo-generated or a sized/colored primitive) |
| Compute | Laptop RTX 5090 (24 GB) — enough for ACT + SmolVLA finetune locally. Sponsor cloud only as overflow. |
| Action space | **Absolute joint positions, 6D incl. gripper.** Make the sim record the same so checkpoints map 1:1 to real. |
| `observation.state` | 6D follower joint readback, same order/units in sim and real. |

**Tooling to lean on (don't build from scratch):**
- `huggingface/lerobot` — teleop, record, train (ACT + SmolVLA), deploy.
- `hud-evals/robot-template` — HUD eval harness, LeRobot dataset recording, Isaac integration (`inventory/envs/robolab/`), trace viewer.
- SO101 sim options: your existing **Isaac Lab SO101 env**, or `johnsutor/so101-nexus` (MuJoCo/ManiSkill `PickLift`/`PickAndPlace` + a teleop recorder that outputs LeRobot v3), or `liorbenhorin/lerobot_so101_teleop` (Isaac Lab teleop-to-dataset, built for domain randomization).
- **Gizmo** (`gizmo.antimlabs.com`) — generates the gold-bar scene + randomized backgrounds/lighting for Isaac/MuJoCo. (Get early-access sorted early; fallback = primitive bar + procedural randomization.)

---

## 2. Why this architecture (the reasoning)

- **Imitation first, RL-flavored improvement second.** True online RL on SmolVLA's flow-matching head needs specialized methods (FPO / ReinFlow / DPPO) — research-grade, not a 24h job. We instead use **reward-filtered behavior cloning** (a.k.a. reward-weighted regression / filtered BC): roll out, keep successes, retrain. Stable, reuses `lerobot-train`, transfers cleanly, and still gives the honest "the policy improved from its own scored rollouts" story.
- **Real→sim is NOT free.** A real-only policy sees sim renders as out-of-distribution. Mitigate with (a) a small sim teleop set to anchor the sim domain, (b) Gizmo domain randomization, (c) matched camera pose/FOV/lighting, and (d) co-training with real data every retrain.
- **Embodiment match matters.** The sim must be an **SO101**, not the templates' default Franka/LIBERO, or the checkpoint won't transfer to the real arm.

---

## 3. Pipeline

```
[Real SO101 teleop] --40 demos--> [Train ACT] --> deploy real (SAFETY-NET DEMO)
                              \--> [Finetune SmolVLA base] (hero model, in parallel)

[Sim: SO101 + Gizmo gold-bar scene + reward] <-- ~15 sim teleop demos (anchor domain)
        |
        v   self-improvement loop (x2-3):
   rollout policy in sim (Gizmo domain randomization)
        -> HUD scores each episode (env reward = bar lifted/grasped)
        -> keep successes / top-K by reward
        -> add to dataset, RETRAIN co-trained with real + sim-teleop demos
        |
        v
[Deploy improved hero policy to real SO101] --> final demo video (real + HUD trace)
```

---

## 4. Phase-by-phase (with acceptance criteria)

### Phase 0 — Setup (H0–2)
- Mount overhead + wrist cams; confirm teleop and both camera streams.
- Create the LeRobot env on the 5090 (ACT + SmolVLA extras). Clone `robot-template`; run its `run.py` once to prove the eval/trace path.
- Stand up the SO101 sim (your Isaac env or `so101-nexus`), confirm reset/step.
- **Done when:** leader→follower teleop works, cameras stream, `robot-template` eval runs, sim boots.

### Phase 1 — Real imitation baseline (H2–6) — THE SAFETY NET
- Record 30–50 gold-bar pickup demos:

```bash
lerobot-record \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower \
  --teleop.type=so101_leader  --teleop.port=/dev/ttyACM1 --teleop.id=leader \
  --robot.cameras='{ overhead: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30},
                     wrist:    {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30} }' \
  --dataset.repo_id=$HF_USER/so101_goldbar_real --dataset.num_episodes=40 \
  --dataset.single_task="Pick up the gold bar"
```

- Train ACT (fast) and deploy to validate the loop end-to-end:

```bash
lerobot-train --policy.type=act \
  --dataset.repo_id=$HF_USER/so101_goldbar_real \
  --batch_size=8 --steps=40000 \
  --output_dir=outputs/train/act_goldbar --job_name=act_goldbar \
  --policy.device=cuda
```

- In parallel, kick off the SmolVLA hero finetune:

```bash
lerobot-train --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=$HF_USER/so101_goldbar_real \
  --batch_size=48 --steps=20000 \
  --output_dir=outputs/train/smolvla_goldbar --job_name=smolvla_goldbar \
  --policy.device=cuda \
  --rename_map='{"observation.images.overhead":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'
```

- Deploy via the same record/inference command with `--policy.path=outputs/train/<run>/checkpoints/last/pretrained_model`.
- **Done when:** ACT picks up the bar on the real arm at a non-trivial success rate, and we have a recorded clip. **We now have a demo regardless of what happens next.**

### Phase 2 — Matched sim bring-up (H4–9, overlaps Phase 1)
- Put the SO101 in sim with the gold-bar scene (Gizmo asset or primitive). Match the **overhead camera pose/FOV/lighting** to the real rig.
- Implement the **reward / success detector** (prerequisite for everything downstream): success = bar center-of-mass above height `H` AND gripper closed on it; partial credit = gripper→bar distance + lift height. Expose it through the HUD env (wrap Isaac via `inventory/envs/robolab/`).
- Collect ~10–20 **sim teleop demos** by driving the sim with the real leader arm (anchors the sim visual domain). Sanity-check whether the Phase-1 real policy runs in sim at all — expect it to be rough; that's exactly why we anchor + randomize.
- **Done when:** sim resets/steps, the reward fires on a scripted success, a policy can run an episode, and we have ~15 sim teleop demos.

### Phase 3 — Reward-filtered self-improvement loop (H9–16)
Repeat 2–3 iterations:
1. Roll out the current policy in sim with **Gizmo domain randomization** (backgrounds, lighting, bar pose), e.g. 100–300 episodes.
2. Score every episode with the env reward; HUD logs success rate + reward breakdown + trace replays.
3. **Filter:** keep successes (reward above threshold) and a few informative near-misses; optionally take top-K by reward. Cap per-iteration additions so real data isn't drowned out.
4. **Retrain co-trained** on: real demos + sim teleop demos + filtered sim rollouts. Iterate with ACT (cheap); do one SmolVLA pass near the end.
- Guardrails: keep real+sim-teleop data in every retrain (prevents drift), and stop early if real-world eval starts regressing (catastrophic forgetting check).
- **Done when:** sim success rate climbs across iterations and the policy is robust to background/lighting changes.

### Phase 4 — Deploy hero policy to real + demo (H16–21)
- Deploy the improved hero policy (SmolVLA, or best ACT) to the real SO101.
- Record the demo: real arm next to the HUD trace viewer; show the Phase-1 baseline vs the sim-improved policy (the "it got better and still works on hardware" arc).

### Phase 5 — Buffer + edit (H21–24)
- Reserve for the inevitable USB/serial/calibration gremlin and video editing.

---

## 5. Domain-gap mitigations (read before Phase 3)
- **Anchor:** ~15 sim teleop demos so the policy isn't blind to sim visuals.
- **Randomize:** Gizmo backgrounds/lighting + bar pose/material so sim data teaches *robustness*, not sim-overfitting.
- **Match:** overhead camera pose/FOV/lighting and the action space (absolute joints) across real/sim/deploy.
- **Co-train + don't forget:** every retrain includes real data; early-stop on real-eval regression.

---

## 6. Risk register & fallbacks
| Risk | Mitigation / fallback |
|---|---|
| Real↔sim visual gap | Sim teleop anchor + Gizmo DR + matched camera; co-train with real |
| SmolVLA too slow to iterate | Use ACT for the loop; SmolVLA only for 1–2 final passes |
| Isaac wiring eats time | Fall back to `so101-nexus` MuJoCo `PickLift` |
| Gizmo access blocked | Primitive gold bar + procedural randomization |
| Self-improvement unstable | We use filtered BC (not policy-gradient RL) — inherently stable |
| Everything sim fails | Ship the Phase-1 real ACT policy — still a complete demo |

**Hard rule:** Phase 1 (real BC) must be done and recorded before sinking time into the sim loop.

---

## 7. Suggested division of labor
- **Person A (hardware/real):** calibration, camera mounting, real teleop recording, real deploys, final video capture.
- **Person B (sim/loop):** SO101 sim + Gizmo scene, reward/success detector, HUD `robolab` wiring, the rollout→score→filter→retrain loop.

---

## 8. Open items to confirm
1. Is the existing Isaac Lab env actually SO101 + does it support a graspable object, or do we use `so101-nexus` PickLift?
2. Second camera available (wrist+overhead) or overhead-only for v1?
3. Gizmo early-access granted? Does the organizer's "SO101 SmolVLA checkpoint" mean plain `smolvla_base` or a task-specific one?
