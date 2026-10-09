# TicTacToe selfplay lr 비교 (Option 1 / Option 2)

|          | Option 1 | Option 2 |
|----------|----------|----------|
| job (ID) | `ttt-lr5e-6` (65165343) | `ttt-lr1e-6` (65165344) |
| learning_rate | 5e-6 | 1e-6 |
| 최소 lr  | 5e-7 | 1e-7 |
| 로그     | `slurm_logs/ttt-lr5e-6-65165343.out` | `slurm_logs/ttt-lr1e-6-65165344.out` |
| 출력 폴더 | `runs/tictactoe_selfplay_opt1/` | `runs/tictactoe_selfplay_opt2/` |

## 추가 스케줄 (2026-10-07) — Option 3 / Option 4

opt1(5e-6)은 step ~90부터 붕괴 → step 111 NaN, m1b(1e-6, min_lr 0)는 거의 안 움직임 → 그 사이 값 + 실제 warmup.

|          | Option 3 | Option 4 |
|----------|----------|----------|
| job (ID) | `ttt-lr2e-6` (65341503) | `ttt-lr3e-6` (65341504) |
| learning_rate | 2e-6 | 3e-6 |
| 최소 lr  | 2e-7 | 3e-7 |
| warmup_steps | 320 (optimizer step = pipeline ~10 step) | 320 |
| 로그     | `slurm_logs/ttt-lr2e-6-65341503.out` | `slurm_logs/ttt-lr3e-6-65341504.out` |
| 출력 폴더 | `runs/tictactoe_selfplay_opt3/` | `runs/tictactoe_selfplay_opt4/` |

나머지는 Option 1/2와 동일(save_steps 20, rollout 128, LoRA r=32, kl_loss_coef 0.2). run 스크립트에 `set -o pipefail` 추가 → 크래시하면 Slurm에 FAILED로 찍힘.
