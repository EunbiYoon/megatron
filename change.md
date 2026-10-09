# agentic_val_tictactoe_selfplay.yaml 변경사항 정리

## 지금(최종) 설정 요약

| 항목 | 원래(리포지토리 기본) | 지금 |
|---|---|---|
| `num_gpus_per_node` / `device_mapping` | 8 / `range(0,8)` | **4 / `range(0,4)`** |
| `tensor_model_parallel_size` | 4 | **4** (GPU 3장이던 중간 단계에서 1로 내렸다가 4장으로 복원하며 4로 되돌림) |
| `sequence_parallel` | true | **true** |
| `use_distributed_optimizer` | true | **false** (TP=4/DP=1이라 분산 저장할 DP가 없어서 끔 — 체크포인트 버그 회피, 메모리 손해 없음) |
| `pretrain` 경로 | `/mnt/public/...` (존재 안 함) | `/datasets/ai/qwen3/...` |
| `rollout_batch_size` | 128 | **128**(메인 yaml, "진짜" 학습용) / **12**(quickcheck 전용 yaml, 버그 확인용) |
| `per_device_train_batch_size` / `gradient_accumulation_steps` | 2 / 2 | **2 / 2** (TP=1이던 중간 단계에서 1/4로 낮췄다가 TP=4 복원 후 원복) |
| `actor_train.infer_batch_size` | 2 | **1** |
| `reference.infer_batch_size` | 2 | **1** |
| `actor_infer.strategy_config.gpu_memory_utilization` | 0.8 | **0.8** (0.9로 올렸다가 vLLM 재onload OOM 나서 원복) |
| `train_env_manager.env_groups` | 64 | **256** |
| `val_batch_size` | 1500 | **16** |
| `val_env_manager.env_groups` | 480 | **128** |
| `val_env_manager.tags` | 15개 게임 전체 | **TicTacToe 4개만** |
| step 0 검증 실행 | 무조건 실행 | **건너뜀** (`global_step > 0` 조건 추가) |
| `sequence_length` | 32768 | **32768** (안 건드림 — 줄이지 말라는 요청 있었음) |

GPU 3장/TP=1로 잠깐 내렸던 시기, `rollout_batch_size=132`로 조정했던 것, PP=3 실험 등은 전부 **폐기된 중간 단계**이고 지금은 위 표가 최종 상태. 아래 "문제 해결 히스토리"에 왜 그런 시도들을 했었는지만 기록으로 남겨둠.

## 지금 학습 실행 방법: `sbatch` (배치 job)

`salloc`(인터랙티브)은 파티션과 무관하게 **클러스터 전체 정책으로 8시간 제한**이 걸려있어서(`--time`을 아무리 크게 줘도 거부됨: `ERROR: Interactive jobs limited to 8 hours`), 8시간 넘게 도는 실제 학습에는 못 씀. **`sbatch`(배치 job)는 이 제한이 없고 파티션 최대치(14일)까지 가능** — 지금은 이 방식으로 돌림.

- `examples/tictactoe/sbatch_m1_gpu.sh`(job명 `m1b`), `sbatch_m2_gpu.sh`(job명 `m2b`): `gpu`(비선점형) 파티션만 사용. **2026-09-28: `gpu-preempt` 파티션용 `sbatch_m1.sh`/`sbatch_m2.sh`(job명 `m1`/`m2`)는 계속 preempt당해서 완전히 폐기 — 스크립트 삭제, 실행 중이던 job(`m1` 64803131, `m2` 64803132) `scancel`, 관련 `slurm_logs/m1-*.out`/`m2-*.out` 삭제. 이제 `gpu` 파티션 job(`m1b`/`m2b`)만 사용.**

제출: `sbatch examples/tictactoe/sbatch_m1.sh` (conda 환경 활성화까지 스크립트 안에 포함되어 있어야 함 — sbatch는 인터랙티브 셸이 아니라서 `conda activate`가 자동으로 안 되고, 직접 `source .../conda.sh && conda activate <mg 환경 경로>`를 스크립트에 넣어야 함). 출력은 `./slurm_logs/<job-name>-<jobid>.out`에 쌓이고, 터미널/tmux/SSH를 다 닫아도 SLURM이 독립적으로 계속 돌림. 이미 떠 있는 job 안에 들어가서 보려면 `srun --jobid=<id> --overlap bash`.

인터랙티브로 잠깐 디버깅만 할 때는 여전히 `salloc`(8시간 이내) 사용:
```bash
salloc --account=pi_dagarwal_umass_edu --job-name=m1 --partition=gpu-preempt --nodes=1 \
  --gres=gpu:l40s:4 --cpus-per-task=32 --mem=256G --time=08:00:00 \
  --exclude=gpu037,gpu038 --mail-user=eunbiyoon@umass.edu --mail-type=BEGIN,END,FAIL,TIME_LIMIT_10
```
- `--cpus-per-task=32`: L40S 노드는 32 CPU가 맥스(원래 이 옵션이 없어서 기본값 1로 잡혀 있었고, 그게 속도 느렸던 원인이었음)
- `--exclude=gpu037,gpu038`: gpu037은 GPU 하드웨어/드라이버 문제(NVML 에러) 있던 노드
- `scancel -u $USER` / `scancel --all`은 **이 계정의 모든 job**을 취소함(m1b, m2b 전부) — 하나만 멈추려면 `scancel <jobid>`. GPU 할당은 유지한 채 프로세스만 재시작하려면 `scancel` 자체가 필요 없고 Ctrl+C 후 스크립트만 다시 실행하면 됨.

`m1b`(job 64803199)는 2026-09-24T00:13부터 gpu031에서 한 번도 안 끊기고 계속 돌아가는 중이라, 실행 결과 확인용 TensorBoard는 이 job의 출력 디렉토리(`runs/tictactoe_selfplay/20260924-001348/tensorboard`)를 대상으로 6006 포트에 띄워둠(`tensorboard --logdir runs/tictactoe_selfplay/20260924-001348/tensorboard --port 6006 --host 0.0.0.0`).

## 빠른 확인용 설정/스크립트 분리 (`_quickcheck`)

"진짜" 학습(rollout_batch_size=128)과 "버그/체크포인트만 빨리 확인"(12)을 같은 yaml로 왔다갔다 하다가 실수(타이밍 놓쳐서 잘못된 값으로 시작)가 몇 번 나서, PP=3 테스트 때처럼 **완전히 별도 파일로 분리**:

- `examples/tictactoe/agentic_val_tictactoe_selfplay_quickcheck.yaml`: 메인 yaml 복사본, `rollout_batch_size: 12`만 다름
- `examples/tictactoe/run_agentic_pipeline_tictactoe_selfplay_quickcheck.sh`: 출력 디렉토리도 `runs/tictactoe_selfplay_quickcheck/`로 분리

메인 yaml(`agentic_val_tictactoe_selfplay.yaml`)은 항상 "진짜 설정" 상태로 유지, 빠른 확인이 필요하면 quickcheck 쪽을 씀.

## 코드 수정 사항 (현재 적용된 것들)

- **`roll/utils/functionals.py`**: `log_probs_from_logits`, `entropy_from_logits` — 긴 시퀀스(32768)에서 vocab 전체 분포를 한 번에 만들다 OOM 나던 문제를 시퀀스 청크(2048) 처리로 수정. `log_probs_from_logits`는 `log_softmax` 전체 대신 `logits[정답] - logsumexp(logits)`로 수학적 동치 계산(메모리 추가 절약).
- **`roll/third_party/megatron/tensor_parallel.py`**: `vocab_parallel_entropy`(TP 랭크 간 `all_reduce` 있는 `train_step` 전용 entropy 계산) — 같은 방식으로 청크 처리(2048행). TP 랭크마다 청크 수가 동일해야 collective가 안 어긋나므로, 입력을 `reshape(-1, vocab_shard_size)`로 먼저 2D로 펴서 처리한 뒤 원래 shape로 복원(2D/3D 어떤 입력이 와도 안전하도록).
- **`roll/third_party/megatron/mcore_adapter_weight_sync_patch.py`** (신규 파일): mcore_adapter가 삭제한 `VirtualModels.all_gather_weights_as_hf_bucket`을 재구현(가중치를 vLLM으로 스트리밍). `gather_tensor_parallel`(tp_rank==0만 결과 받음) 대신 `all_gather_tensors`(모든 랭크가 결과 받음) 사용 — TP>1에서 모든 랭크가 각자 자기 몫의 브로드캐스트를 수행해야 하기 때문.
- **`roll/distributed/scheduler/resource_manager.py`**: GPU 워커용 Ray placement group에 노드 CPU 절반만(`node_cpu/2`) 반영되던 것 → 전체(`node_cpu`)로 수정.
- **`roll/distributed/strategy/megatron_strategy.py`**: `mpu.get_data_modulo_expert_parallel_rank()`(삭제된 API) → `mpu.get_expert_data_parallel_rank()`(새 이름, 동일 로직)로 체크포인트 저장 코드 수정.
- **`roll/distributed/scheduler/initialize.py`**: Ray 소켓 경로 107바이트 제한 초과 방지용 `--temp-dir=/tmp/ray` 추가.
- **`roll/third_party/megatron/offload_states_patch.py`**: 삭제된 `legacy_a2a_token_dispatcher` fallback import.
- **`roll/third_party/megatron/optimizer.py`**: 삭제된 `no_weight_decay_cond`/`scale_lr_cond`/`lr_mult` 인자 제거, `config_overrides`로 대체.
- **`roll/agentic/env/tictactoe/env.py`**: `matplotlib` 3.10에서 삭제된 `tostring_rgb()` → `buffer_rgba()`로 교체.
- **`roll/pipeline/agentic/agentic_pipeline.py`**: step 0 검증 무조건 실행되던 것 → `global_step > 0` 조건 추가(건너뜀).

## 공유 환경(conda `mg`) 버전 호환성 수정 (설치된 패키지/의존성 자체)

- `click`/`ray` 버전 충돌 (Sentinel 에러) → `click<8.2.0`으로 다운그레이드
- Ray `LogMonitor` API 변경 (`gcs_publisher` → `gcs_client`) → 환경 재설치로 해결
- `flash_attn`이 다른 torch 버전으로 빌드되어 있던 문제 → 재설치로 해결
- `apex` 최신 버전에 레거시 `apex.amp` 서브모듈이 없어서 `transformers.trainer`의 무조건적 import가 깨짐 → 더미 `apex.amp` 모듈 추가
- SLURM `--mem` 부족으로 OOM → `--mem=64G` → `--mem=256G`

---

## 문제 해결 히스토리 (시간순, 참고용 — 지금은 위 "최종 설정"이 정답)

<details>
<summary>GPU 3장 시절의 시행착오 (지금은 4장으로 해결됨, 펼쳐서 보기)</summary>

### `actor_train` logits OOM — TP=1일 때 vocab이 안 쪼개짐
```
File "megatron/core/models/gpt/gpt_model.py", line 712, in _postprocess
    return logits.transpose(0, 1).contiguous()
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 18.55 GiB.
```
GPU 3장이라 `tensor_model_parallel_size`를 4→1로 낮췄었는데, TP=1이면 vocab을 GPU들로 안 쪼개서 `logits` 텐서(시퀀스 32768 × vocab 15만)를 GPU 한 장이 통째로 만들어야 함. 당시 `actor_train.infer_batch_size: 2→1`로 배치 차원을 줄여 임시 대응(컨텍스트 길이는 유지).

### A/B 테스트 실패: batch 더 줄이기 vs PP=3
- batch=1로도 `output_layer`에서 재차 OOM (9.27 GiB 부족) — 더 줄일 배치가 없어서 막힘.
- PP=3(레이어를 GPU 3장에 분산)도 실패 — PP는 레이어만 나누고 vocab은 안 나눠서, output_layer 담당 마지막 스테이지가 여전히 전체 logits(18.55 GiB)를 만들어야 함.
- **결론**: GPU 3장 + TP=1(head수 32라 TP=3 불가, TP는 1/2/4/8/16/32만 가능) 조합에서는 vocab을 쪼갤 방법이 없음 → GPU 4장으로 늘려서 TP=4 복원(`32÷4=8`로 정확히 나눠짐).

### `rollout_batch_size` DP 분할 나머지 버그
```
AssertionError: 1 % 4 != 0
```
`DataProto.chunk()`가 `np.array_split()`으로 DP 랭크별 배치를 나누는데, GPU 8→3장 전환으로 DP가 2→3이 되면서 128을 3으로 나누면 `[43,43,42]`로 불균등 분할 → `mini_batch_size=4`로 다시 못 나눠서 assert 실패. 당시 `rollout_batch_size`를 `DP(3)×4=12`의 배수인 132로 조정. **GPU 4장(DP=1) 복원 후에는 이 제약이 사라져서 다시 128 사용 — 132는 더 이상 안 씀.**

### `reference` 워커 causal mask OOM
```
File ".../modeling_qwen3.py", line 745, in _prepare_4d_causal_attention_mask_with_cache_position
    causal_mask = causal_mask.clone()
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.00 GiB.
```
`reference` 워커는 Megatron이 아니라 HF `transformers`를 직접 씀. `sequence_length=32768`이라 causal mask가 `(batch,1,32768,32768)`로 큰데 `infer_batch_size=2`라 OOM. → `reference.infer_batch_size` 2→1로 수정(이건 GPU 4장 이후에도 계속 유지).

</details>

<details>
<summary>GPU 4장/TP=4 전환 직후 시행착오 (펼쳐서 보기)</summary>

### TP=4 가중치 동기화 NCCL 데드락 (30분 타임아웃)
```
[Rank 3] Watchdog caught collective operation timeout: WorkNCCL(... OpType=GATHER ...)
Fatal Python error: Aborted
```
TP>1일 때 모든 TP 랭크가 각자 자기 몫의 actor_infer 타겟에 독립적으로 `collective.broadcast()`를 호출해야 하는데, `mcore_adapter_weight_sync_patch.py`가 `gather_tensor_parallel`(tp_rank==0만 결과 받음)을 써서 나머지 랭크가 자기 브로드캐스트를 한 번도 안 함 → 반대편이 30분 대기 후 타임아웃. `all_gather_tensors`(모든 랭크가 결과 받음)로 교체해서 해결. (LoRA는 무관 — TP=1에서는 `dist.gather`가 no-op이라 안 드러났을 뿐.)

### `train_step` entropy 계산 OOM + 그 수정 자체의 버그
`roll/third_party/megatron/tensor_parallel.py`의 `vocab_parallel_entropy`가 청크 처리 안 돼있어서 OOM → 청크 처리 추가하는 과정에서 처음엔 입력을 2D로 잘못 가정해 shape 에러 발생 → `reshape(-1, vocab_shard_size)`로 수정. (현재 코드 상태는 위 "코드 수정 사항"에 반영됨.)

### 체크포인트 저장 실패 2건 연속
1. `ShardedTensor.flattened_range is not supported` — `use_distributed_optimizer=true`가 이 megatron-core 버전에서 막혀있는 기능을 씀 → DP=1이라 켤 필요도 없어서 `false`로 변경.
2. `mpu.get_data_modulo_expert_parallel_rank` 없음 — megatron-core 버전에서 `get_expert_data_parallel_rank`로 이름만 바뀐 것 → 새 이름으로 교체.

### vLLM 재onload 시 OOM (`cumem_allocator.cpp`)
`gpu_memory_utilization=0.9`로 올렸다가, 반복되는 onload/offload 사이클에서 다른 프로세스들 잔여 메모리 때문에 vLLM의 90% 예약이 실패 → 0.8로 되돌림.

</details>

## 전체 파이프라인 단계 순서 (`agentic_pipeline.py`)

```
for global_step in 0..max_steps(200):
    rollout                      — train_env로 트라젝토리(게임) 수집, rollout_batch_size개
    cal_ref_log_probs            — reference 모델로 KL penalty용 log_probs 계산
    cal_old_log_probs_values     — actor_train 자신의 (업데이트 전) log_probs + critic values 계산
    adv                          — 리워드 정규화, advantage 계산
    (gradient 업데이트)           — 실제 PPO loss 계산 + actor_train 파라미터 업데이트
                                    ← 여기까지 끝나야 "스텝 1 완료"
    (save_steps=20마다 체크포인트 저장)
```
`val`(검증)은 이 루프 안에서 `eval_steps=5`마다 한 번씩 끼어들 뿐, 학습(gradient 업데이트) 경로와는 완전히 분리되어 있음.

## 2026-09-28: KuhnPoker selfplay (8 GPU, LoRA) 추가

tictactoe selfplay / selfplay_quickcheck 구조 그대로 KuhnPoker 버전 2개 생성. yaml은 **현재 tictactoe selfplay yaml(LoRA r=32/alpha=32, 위에서 한 버그 수정 반영된 값들)을 복사**해서 아래만 바꿈 (원래 upstream kuhn yaml은 LoRA 없고 옛날 설정이라 덮어씀 — 원본은 git에 있음).

| 항목 | tictactoe selfplay | **kuhn_poker selfplay** |
|---|---|---|
| `num_gpus_per_node` / `device_mapping` | 4 / `range(0,4)` | **8 / `range(0,8)`** |
| 병렬화 | TP=4, DP=1 | **TP=4, DP=2** (TP 설정은 그대로) |
| `use_distributed_optimizer` | false | **false** 유지 (DP=2지만 `flattened_range` 체크포인트 버그 때문에) |
| `rollout_batch_size` | 128 / quickcheck 12 | **128 / quickcheck 16** — DP=2라 `DP(2)×per_device(2)×grad_acc(2)=8`의 배수여야 함 (12면 `6 % 4 != 0` assert 실패) |
| train env | `TicTacToe`, env_groups 256 | **`KuhnPoker`**(built_in_opponent none), env_groups **64** (원래 kuhn yaml 값) |
| val env | TicTacToe-first/second-100/1000, 128 groups | **KuhnPoker-first/second(CFR 상대)**, n_groups [64,64] = 128 |
| `val_batch_size` | 16 | **128** (Kuhn은 카드 운 분산이 커서 16판으로는 의미 없음, 게임이 짧아 비용도 작음) |

GPU: `gpu` 파티션에서 L40S는 노드당 4장이 최대라 **A100-80G×8 노드(gpu022~024)** 사용 (`--gres=gpu:a100:8`, cpus 64, mem 512G, `--exclude` 제거 — L40S 노드용이었음). `gpupod-l40s`는 우리 계정 접근 불가.

새 파일 (`examples/kuhn_poker/`):
- `agentic_val_kuhn_poker_selfplay.yaml`, `agentic_val_kuhn_poker_selfplay_quickcheck.yaml`
- `run_agentic_pipeline_kuhn_poker_selfplay.sh`(기존 파일을 tictactoe 버전 기반으로 교체 — cuda/cudnn/triton 환경 설정 포함), `run_agentic_pipeline_kuhn_poker_selfplay_quickcheck.sh`
- `sbatch_k1_gpu.sh`(job `k1b`, 본 학습) / `sbatch_k2_gpu.sh`(job `k2b`, quickcheck)
- 출력: `runs/kuhn_poker_selfplay/<timestamp>/`, `runs/kuhn_poker_selfplay_quickcheck/<timestamp>/`, slurm 로그 `slurm_logs/k1b-<jobid>.out`, `k2b-<jobid>.out`

제출: `k1b` 64999724, `k2b` 64999725 (제출 시점 PENDING/Priority — A100×8 노드 3대 중 1대 사용 중, 2대 재부팅/drain 상태).

### 2026-09-28 (수정): A100×8 → L40S 4장 × 2노드 멀티노드로 변경
- 이유: L40S 노드는 클러스터 전체 40대가 **전부 노드당 4장**이라 `gpu:l40s:8`은 존재하지 않는 구성(`Requested node configuration is not available`로 즉시 거절, 기다려도 안 잡힘). A100×8은 `gpu` 파티션에 3대뿐이라 대기 길어서, L40S 4장 노드 2대를 Ray로 묶는 방식으로 전환.
- A100 job `k1b` 64999724 / `k2b` 64999725 `scancel` → 재제출 **`k1b` 65001653**, **`k2b` 65001654** (`--nodes=2 --ntasks-per-node=1 --gres=gpu:l40s:4 --cpus-per-task=32 --mem=256G --exclude=gpu037,gpu038`).
- yaml: `num_gpus_per_node: 8 → 4` (`device_mapping`은 `range(0,8)` 유지 → `num_nodes`가 자동으로 2로 계산됨). TP=4 그룹은 노드 안(rank 0-3 / 4-7), DP=2가 노드 간(IB).
- 신규 `examples/kuhn_poker/ray_multinode.sh`: head 노드에서 `ray start --head`(job별 포트 `6379 + jobid%1000`), 나머지 노드는 `srun`으로 `ray start --address ... --block`, GPU 8장 다 붙을 때까지 대기(최대 10분) 후 `RAY_ADDRESS` export하고 run 스크립트 실행.
- 신규 `examples/kuhn_poker/env_setup.sh`: run 스크립트에 있던 cuda/cudnn/cublas/triton 설정을 분리. Ray 워커는 드라이버가 아니라 **각 노드의 raylet 환경을 물려받기 때문에** `ray start` 전에 설정돼 있어야 함.
- run 스크립트 2개: env 설정을 `env_setup.sh` source로 교체, `RAY_ADDRESS`가 설정돼 있으면(멀티노드) `ray stop` 안 함(안 그러면 방금 띄운 클러스터를 스스로 죽임).

### 2026-09-29 (수정): KuhnPoker를 tictactoe와 동일하게 L40S 4장 1노드로 되돌림
- 사용자 요청: 8 GPU 대신 **tictactoe(m1b/m2b)와 완전히 같은 GPU 분배**로 실행.
- 멀티노드 job `k1b` 65001653 / `k2b` 65001654는 실행 전(PENDING)이라 `scancel`. 재제출 **`k1b` 65045357**, **`k2b` 65045358**.
- yaml: `num_gpus_per_node: 4`, `device_mapping: list(range(0,4))`(actor_train/actor_infer/reference) → TP=4, DP=1 (tictactoe와 동일).
- quickcheck `rollout_batch_size`: 16 → **12** (DP=1이라 `per_device(2)×grad_acc(2)=4`의 배수면 됨, tictactoe quickcheck과 동일).
- sbatch `sbatch_k1_gpu.sh`/`sbatch_k2_gpu.sh`: tictactoe `sbatch_m1_gpu.sh`/`sbatch_m2_gpu.sh`와 job 이름·로그 경로·run 스크립트만 다르고 나머지(`--gres=gpu:l40s:4 --cpus-per-task=32 --mem=256G --exclude=gpu037,gpu038`) 동일.
- run 스크립트 2개를 tictactoe 버전 기반으로 재생성(env 설정 inline, `ray stop` 무조건). 멀티노드용 `ray_multinode.sh`, `env_setup.sh`는 더 이상 안 써서 삭제.
- tictactoe와 남은 차이: env(KuhnPoker), train `env_groups` 64, val(KuhnPoker-first/second, 64+64), `val_batch_size` 128.
- 참고: `m1b`(64803199)는 2026-09-29 09:36 step 199까지 마치고 COMPLETED.
- 2026-09-30 09:5x: k1b가 먼저 노드 받도록 `k2b`(65045358) `scontrol hold` (priority 0, `JobHeldUser`). 풀 때는 `scontrol release 65045358`.
- 2026-09-30 09:56: `k2b`(65045358) `scontrol release`로 hold 해제.

## 2026-10-02: TicTacToe selfplay lr 비교 (Option 1 / Option 2) 추가

m1b 결과(LoRA + lr 1e-6, min_lr 없음)를 보면 val 승률이 거의 안 변함(vs MCTS-100 선공 13.9%→16.4%, 후공 0%→0%), `approxkl` 0.008→0.0005로 정책이 거의 안 움직임, 그리고 **lr이 마지막에 7.6e-9까지 떨어짐**(cosine 최소값이 0이었음). 그래서 lr 두 가지를 같은 조건에서 비교.

| | Option 1 | Option 2 |
|---|---|---|
| yaml | `agentic_val_tictactoe_selfplay_opt1.yaml` | `agentic_val_tictactoe_selfplay_opt2.yaml` |
| `learning_rate` | **5e-6** | **1e-6** |
| cosine 최소 lr | **5e-7** | **1e-7** |
| job | `ttt-lr5e-6` **65165343** | `ttt-lr1e-6` **65165344** |
| 로그 | `slurm_logs/ttt-lr5e-6-<jobid>.out` | `slurm_logs/ttt-lr1e-6-<jobid>.out` |
| 출력 | `runs/tictactoe_selfplay_opt1/<ts>/` | `runs/tictactoe_selfplay_opt2/<ts>/` |

나머지는 `agentic_val_tictactoe_selfplay.yaml`(m1b 설정)과 완전히 동일. sbatch도 `sbatch_m1_gpu.sh`와 job 이름/로그/run 스크립트만 다름(L40S 4장 1노드).

### 버그 주의: `min_lr_rate`/`lr_scheduler_kwargs`를 `training_args` 밑에 쓰면 무시됨
- 제안받은 설정은 `training_args.lr_scheduler_kwargs.min_lr_rate: 0.1`이었는데, 이대로 쓰면 **에러 없이 무시되고 최소 lr = 0**:
  1. ROLL의 `roll/configs/training_args.py` `TrainingArguments`에 `lr_scheduler_kwargs` 필드 자체가 없음 → dacite `from_dict`가 조용히 버림.
  2. Megatron 쪽은 비율(`min_lr_rate`, HF 전용)이 아니라 **절댓값 `min_lr`**만 읽음 (`roll/distributed/strategy/megatron_strategy.py:294`, `mcore_adapter/src/mcore_adapter/trainer/utils.py:94`).
- 그래서 `actor_train.strategy_args.strategy_config.lr_scheduler_kwargs.min_lr`에 넣음. `strategy_config`는 Megatron `TrainingArguments`에 그대로 merge되고(`config_dict.update(strategy_config)`), HF `TrainingArguments`가 `lr_scheduler_kwargs` 필드를 갖고 있어서 정상 전달됨. OmegaConf로 파싱해서 float로 들어가는 것까지 확인.
- 원본 `agentic_val_tictactoe_selfplay.yaml`(사용자 메모 주석 포함)은 수정 안 함.

### 2026-10-06: Option 1 (`ttt-lr5e-6`, 65165343) step 111에서 NaN으로 죽음 — Slurm에는 COMPLETED로 찍힘
- 에러: `ValueError: decoder.final_layernorm.weight weights are not equal diff sum: nan` (step 111 학습 직후 `model_update` → `mcore_adapter_weight_sync_patch.py:52` → `dist_converter.handle_duplicated`). step 111의 grad_norm/approxkl/kl_loss/pg_loss 전부 NaN → 가중치가 NaN이 됨.
- 붕괴 징후는 step ~90부터: `player_1_lose_for_wrong_format`가 0.016 → **1.0**(후공이 매번 형식 오류), `critic/entropy/mean` 0.5 → 3.5~4.6, `actor/kl_loss` 0.15 → 2.0~2.5, step 106~108 approxkl 0.2~0.8로 급등, step 107 grad_norm 8.2. → **lr 5e-6은 이 설정(LoRA r=32, kl_loss_coef 0.2)에서 불안정**.
- Slurm 상태가 COMPLETED(exit 0)인 이유: run 스크립트가 `python ... | tee custom_logs.log`라서 파이프라인 종료코드가 `tee`의 0으로 덮임. 실패를 Slurm 상태로 못 잡음 → 로그의 `pipeline step N finished`로 확인해야 함. (수정 후보: run 스크립트에 `set -o pipefail`.)
- 미확인: LoRA면 `final_layernorm`은 frozen이어야 하는데 NaN이 됨 — Megatron 쪽에서 norm도 학습되고 있는지(=LoRA가 의도대로 적용되는지) 확인 필요.

### 2026-10-06: TensorBoard → PDF 스크립트 `scripts/tb_to_pdf.py`
- `runs/<group>/<timestamp>/tensorboard`를 전부 읽어 PDF 하나로 저장 (기본 `reports/tb_report_<시각>.pdf`). 실행: `python scripts/tb_to_pdf.py` (mg env, 로그인 노드에서 ~1분).
- 구성: 요약표(run별 마지막 step/완료 여부) → 게임별 핵심 지표(val, self-play 승패, 형식/길이 패배, 응답 길이, reward/entropy, PPO/KL/grad/lr, step 시간 분해) → 나머지 모든 scalar 부록. 학습 지표는 EMA 0.6 스무딩(원본은 옅게), val처럼 띄엄띄엄 기록된 지표는 점으로.
- 참고(버그 아님): `warmup_steps: 10`은 **optimizer step** 단위. rollout 128이면 pipeline step 1개 = optimizer step 32개라 warmup이 step 0 안에서 끝남(m1b/opt1/opt2 모두 step 0부터 max lr). m2b(rollout 12)만 몇 step에 걸쳐 warmup이 보임.

### 2026-10-07: lr 스케줄 Option 3 / Option 4 추가 (`ttt-lr2e-6` 65341503, `ttt-lr3e-6` 65341504)
- `agentic_val_tictactoe_selfplay_opt3.yaml`(lr 2e-6, min_lr 2e-7) / `_opt4.yaml`(lr 3e-6, min_lr 3e-7). opt1 yaml 기반에서 lr, min_lr, **`warmup_steps: 10 → 320`**(optimizer step 단위라 10은 step 0 안에서 끝났음; 320 = pipeline ~10 step)만 변경.
- `save_steps`는 10으로 줄이려다 **20 유지**: 체크포인트 1개 = rank당 14G × 4 ≈ 56G, run 하나 ≈ 590G(m1b 실측). ROLL에 오래된 체크포인트 자동 삭제 옵션이 없고 `/scratch` 사용률 88%.
- 새 run 스크립트(opt3/opt4)에만 `set -o pipefail` 추가 — `python ... | tee`의 종료코드가 tee로 덮여서 크래시가 COMPLETED로 찍히던 문제 해결.
