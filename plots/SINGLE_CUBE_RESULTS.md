# BC vs IFQL on `visual-block-single-task1-v0`

**Task.** A UR5e arm must pick up one cube and place it at a fixed goal (within 4 cm), using only images from
3 cameras (front, side, wrist; 224×224). An episode lasts at most 1,250 steps (25 s). The policy sees an image
every 25 steps and outputs the next 25 actions (one "chunk").

**Data.** 10,000 scripted demonstrations (88.2% successful; failures run out the time limit). The demonstrator
deliberately randomizes its grasps (offsets, tilts, retries).

## Methods

| | MSE BC | IFQL | flow BC |
|---|---|---|---|
| Policy | CNN + residual MLP → 25×7 actions, MSE loss | flow-matching actor + IQL critic Q(s, chunk) and value V(s) | the IFQL actor alone |
| Acting | one deterministic chunk | sample 32 chunks, run the one with the highest Q | sample 1 chunk |
| Data | 100 shards, successful episodes only (235,673 samples) | 60 shards (RAM limit), all episodes (176,052 samples; the actor imitates the 142,052 from successful episodes) | – |
| Size / batch | 154M params, batch 256 | 293M trained params, batch 128 | – |

All three use the same image encoder (IMPALA-small, shared across cameras) and were trained for 70k steps.

## Evaluation

Every checkpoint (every 10k steps) was rolled out in the simulator from the **exact first states of the
1,000 held-out (validation) episodes**. Every model and the demonstrator are therefore compared on identical
starting layouts. Each success rate below is ±1.5 points (1 standard error).

## Results

**Success on the 1,000 held-out starts.** Figures: `single_cube/00_success_all_checkpoints.png` and
`single_cube/04_success_action_error_grasp_failures.png`.

| | 10k | 20k | 30k | 40k | 50k | 60k | 70k | best |
|---|---|---|---|---|---|---|---|---|
| MSE BC | 10% | 48% | 62% | 66% | 65% | **70.5%** | 65% | 70.5% |
| IFQL (best of 32) | 6% | 34% | 34% | 62% | 69% | 62% | **71.5%** | 71.5% |
| flow BC | 4% | 17% | 20% | 40% | **56%** | 45% | 53% | 56% |
| demonstrator | | | | | | | | 87.6% |

- **BC and IFQL end up tied** at about 70%. BC gets there much sooner (62% by 30k); IFQL needs 50–70k.
- **Choosing by Q adds 8–25 points** over IFQL's own actor at every checkpoint from 20k on (IFQL vs flow BC).
- **No model reaches the demonstrator.** On about 27% of all starts, the demonstrator succeeded and the model failed
  (IFQL 50k; 34% for BC 30k).

## Where the models fail

Failed episodes are sorted into categories using the gripper and cube positions recorded at every step
(`single_cube/01–03_*.png`):

| best checkpoint | success | **tried to grasp, never lifted** | lifted, then failed | other |
|---|---|---|---|---|
| MSE BC 60k | 70.5% | **17.8%** | 10.2% | 1.5% |
| IFQL 70k | 71.5% | **16.3%** | 10.8% | 1.4% |
| flow BC 50k | 56.3% | **23.6%** | 14.1% | 6.0% |
| demonstrator | 87.6% | **0.5%** | 11.6% | 0.3% |

- **Failed grasps make up the whole gap to the demonstrator.** All models drop or misplace a lifted cube about as
  often as the demonstrator does (10–14% vs 11.6%). But they fail to pick the cube up in 16–24% of episodes,
  versus 0.5% for the demonstrator.
- In most failed grasps the gripper closed **on nothing beside the cube** (85% of failed-grasp attempts for BC 30k). This is a
  precision problem, not a failure to reach the cube.
- Even successful episodes retry often: 37–47% need more than one grasp attempt (demonstrator: 25%), which makes
  them slower (14–15 s vs 13.2 s).

## What the IFQL critic (Q) does

Q values below are for IFQL at 50k, on the held-out starts.

| | Q at start | Q at 10 s | Q at the end of the episode |
|---|---|---|---|
| successful episodes | 0.942 | 0.968 | **0.992** |
| lifted, then failed | 0.936 | 0.955 | 0.977 |
| tried to grasp, never lifted | 0.934 | 0.944 | **0.934** |

- **Q tracks progress, but predicts poorly in advance.** At the start, Q barely separates outcomes (correlation
  with success 0.13–0.21). After the grasp, the outcomes separate clearly: Q climbs to about 0.99 once the cube
  is held, and stays flat around 0.93 when the grasp fails.
- **Q reacts within seconds of a grasp attempt.** When the first attempt lifts the cube, Q rises from 0.957 to
  0.983 within 2 s. When it fails, Q stays flat (0.948 → 0.953).
- **The values are too optimistic.** Q is around 0.93–0.99 even though actual success is about 70%. Q estimates
  how likely the demonstrator is to succeed from a state, and the demonstrator recovers from most bad grasps.
  Picking the best of 32 candidates only needs the ranking to be right, and it is.

## Action error vs success

The middle panel of `single_cube/04_*.png` compares each model's predicted chunk with the demonstrator's on the
held-out demonstration states. BC's actions are closest to the demonstrator's (MSE about 0.07), and IFQL's are
furthest (about 0.09). Yet IFQL succeeds as often as BC. Choosing by Q deliberately moves away from imitation,
and closeness to the demonstrator doesn't predict success on this task.

## Takeaways

1. On this short, single-strategy task with good demonstrations, **plain MSE BC is a strong baseline**. It
   averages out the demonstrator's random grasp variations, and offline RL (IFQL) only matches it.
2. **The critic is useful**: it consistently improves on its own actor and correctly identifies whether a grasp
   worked.
3. **The remaining gap is failed grasps.** The most promising fixes target grasp precision: re-observing more
   often during the approach, giving the policy the gripper/arm state, and image augmentation.
4. Differences between the methods should be larger on longer, multi-stage tasks. That is the next experiment,
   on `visual-block-double-task1-v0`.

*Caveats:* each model is a single training run (seed). IFQL saw less data and a smaller batch than BC, because of
RAM limits. The grasp categories use hand-chosen thresholds: within 6 cm counts as reaching the cube, a 2 cm rise
counts as lifting it, and closure values of 0.2 and 0.7 separate a closed gripper from one that closed on nothing.
