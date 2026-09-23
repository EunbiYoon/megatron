# agentic_val_tictactoe_selfplay.yaml 변경사항 정리

## 지금 쓰는 `salloc` 명령 (m1/m2 공통, `--job-name`만 다르게)

지금까지 알아낸 문제들(CPU 1개만 잡힘, gpu037 노드 GPU 고장, mem 부족)을 전부 반영한 최종 형태:

```bash
salloc --account=pi_dagarwal_umass_edu --job-name=m1 --partition=gpu-preempt --nodes=1 \
  --gres=gpu:l40s:4 --cpus-per-task=32 --mem=256G --time=08:00:00 \
  --exclude=gpu037,gpu038 --mail-user=eunbiyoon@umass.edu --mail-type=BEGIN,END,FAIL,TIME_LIMIT_10
```

- `--gres=gpu:l40s:4`: GPU 4장 (TP=4 필요)
- `--cpus-per-task=32`: L40S 노드는 32 CPU가 맥스 — 원래 이 옵션이 없어서 기본값 1로 잡혀 있었음(느렸던 원인)
- `--mem=256G`: 노드 메모리 부족 OOM 방지
- `--exclude=gpu037,gpu038`: gpu037은 GPU 하드웨어/드라이버 문제(NVML 에러) 있던 노드
- `--time=08:00:00`: 8시간 — 200스텝 다 돌리기엔 부족할 수 있음(늘리려면 이 값만 키우면 됨)

**주의**: `scancel -u $USER`나 `scancel --all`은 이 계정의 **모든 job(m1, m2 전부)**을 취소함 — 하나만 멈추고 싶으면 `scancel <해당 jobid>`로. 또한 이미 떠 있는 파이썬 프로세스만 재시작하고 싶으면(GPU 할당은 유지) `scancel` 자체가 필요 없음 — Ctrl+C로 프로세스만 멈추고 스크립트 다시 실행하면 됨.

## 검증(val) 관련

| 항목 | 원래 | 지금 |
|---|---|---|
| `val_batch_size` | 1500 | **16** |
| `val_env_manager.tags` | 15개 게임 전체(TicTacToe/Connect4/포커/Hanabi) | **TicTacToe 4개만** |
| `val_env_manager.env_groups` | 480 | **128** |
| 0번째(step 0) 검증 실행 여부 | 무조건 실행 (`global_step % eval_steps == 0`) | **건너뜀** (`global_step > 0` 조건 코드에 추가) — `eval_steps: 5`대로 5, 10, 15...에서만 검증 |

## 학습(train) 관련

| 항목 | 원래 | 지금 |
|---|---|---|
| `rollout_batch_size` | 128 | **132** (검증 중엔 4로 잠깐 줄였다가 복원, DP=3 분할 버그로 132로 조정 — 아래 별도 섹션 참고) |
| `train_env_manager.env_groups` | 64 | **256** |
| `actor_train.infer_batch_size` | 2 | **1** (아래 logits OOM 수정) |
| `reference.infer_batch_size` | 2 | **1** (m2에서 발견된 reference OOM 수정, 아래 참고) |

## GPU/모델 설정

| 항목 | 원래 | 지금 |
|---|---|---|
| `num_gpus_per_node` | 8 | 3 |
| `device_mapping` (전체) | `range(0,8)` | `range(0,3)` |
| `tensor_model_parallel_size` | 4 | 1 |
| `sequence_parallel` | true | false |
| `pretrain` 경로 | `/mnt/public/...` (존재 안 함) | `/datasets/ai/qwen3/...` |
| `actor_infer.strategy_config.gpu_memory_utilization` | 0.8 | 0.9 |

참고: `enforce_eager: false`, `enable_prefix_caching: true`, `infer_batch_size: 64`는 한때 넣었었는데, 지금 파일엔 없음 — 누군가(또는 뭔가) 그 사이에 되돌린 것으로 보임.

## 코드(`roll/utils/functionals.py`) 수정

- `log_probs_from_logits`, `entropy_from_logits`: 긴 시퀀스(32768)에서 vocab 전체 분포를 한 번에 만들다 OOM 나던 문제를, 시퀀스를 청크로 나눠 계산하도록 수정
  - `log_probs_from_logits`: `log_softmax` 전체를 만들지 않고 `logits[정답] - logsumexp(logits)`로 수학적으로 동일하게 계산 (메모리 절반 가까이 절약) + 2048 토큰 단위 청크 처리 (피크 메모리 약 16배 감소)

## `actor_train` logits OOM (또 다른 지점)

`log_probs_from_logits`/`entropy_from_logits`를 고친 뒤에도, 그 이전 단계인 **Megatron GPT 모델 자체의 forward pass**에서 또 OOM 발생:

```
File "megatron/core/models/gpt/gpt_model.py", line 712, in _postprocess
    return logits.transpose(0, 1).contiguous()
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 18.55 GiB.
```

**원인**: GPU가 3개뿐이라 `tensor_model_parallel_size`를 4→1로 낮췄는데, TP=1이면 vocab을 GPU들로 안 쪼개서 원본 `logits` 텐서(시퀀스 32768 × vocab 15만 전체)를 GPU 한 장이 통째로 만들어야 함. 게다가 `env_manager.py`가 모든 트라젝토리를 실제 길이와 무관하게 `sequence_length`까지 패딩해서, 매번 최대 길이로 이 텐서가 만들어짐.

**검토했던 옵션**:
1. `sequence_length` 32768→16384로 줄이기 (되돌림, 컨텍스트 길이 줄이고 싶지 않아서)
2. TP를 다시 늘리기 (GPU 3장 제약 + 리스크 커서 보류)
3. **(적용함)** `actor_train.infer_batch_size` 2→1 — 배치 차원을 줄여서 logits 텐서 크기를 절반으로. 컨텍스트 길이는 안 건드림.

## A/B 테스트 결과 — 둘 다 실패, GPU 4장으로 전환

- **m1 (batch=1, safe)**: MLP `glu` OOM은 해결됐지만, 그 다음 `output_layer`(logits, `(seq=32768)×(vocab=151936)` 텐서) 계산에서 다시 OOM (9.27 GiB 부족). `per_device_train_batch_size`가 이미 최솟값(1)이라 더 줄일 수 없어서 여기서 막힘.
- **m2 (PP=3)**: **가설이 틀림** — PP는 레이어만 GPU별로 나누고 vocab은 안 나눔. 마지막 파이프라인 스테이지(actor_train-2, output_layer 담당)가 여전히 전체 logits 텐서(18.55 GiB)를 통째로 만들어야 해서 그대로 OOM. 추가로 `salloc` 8시간 시간제한이 끝나서 job 자체가 SLURM에 의해 강제 종료됨.
- **결론**: GPU 3장 + TP=1(head수 32라 TP=3 불가) 조합에서는 vocab(151936)을 쪼갤 방법이 없음. → **GPU를 4장으로 늘려서 `tensor_model_parallel_size=4`로 복원** (`num_attention_heads=32 ÷ 4 = 8`로 정확히 나눠짐, 원래 리포지토리의 8GPU/TP=4 설정과 같은 방식). vocab도 4-way로 쪼개져서 logits 텐서 문제가 근본적으로 해결될 것으로 예상.

**적용한 변경**:
| 항목 | 3 GPU 시절 | 4 GPU로 변경 |
|---|---|---|
| `num_gpus_per_node` | 3 | **4** |
| `device_mapping`(전체) | `range(0,3)` | **`range(0,4)`** |
| `tensor_model_parallel_size` | 1 | **4** |
| `sequence_parallel` | false | **true** |

`per_device_train_batch_size=1`/`gradient_accumulation_steps=4`(안전 조치)는 일단 유지 — TP=4로 안정 확인되면 원래 값(2/2)으로 복원 검토.

## `train_step`(실제 gradient 업데이트) OOM — 두 가지 해법 A/B 테스트

```
File ".../megatron/core/transformer/mlp.py", line 224, in glu
    return self.config.activation_func(x_glu) * (...)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.19 GiB.
GPU 0 has a total capacity of 44.39 GiB ... this process has 41.47 GiB memory in use.
```
`rollout_batch_size=12`(DP분할 버그 수정 후)로 m1/m2/m3 전부 rollout→adv까지 통과했지만, 마지막 `train_step`(실제 forward+backward)에서 셋 다 동일하게 OOM. batch 크기와 무관하게 재현되는 구조적 메모리 부족 — `tensor_model_parallel_size=1`이라 모델/옵티마이저/activation을 GPU 1장이 전부 부담하는데 `sequence_length=32768`이 커서 44GB GPU 한 장에 안 들어감.

TP를 3으로 늘려서 GPU 3장에 모델을 나누는 방법을 검토했으나, Qwen3-4B는 `num_attention_heads=32, num_key_value_heads=8`이라 3으로 나눠떨어지지 않아 **TP=3은 불가능**(TP는 1/2/4/8/16/32만 가능).

**A/B로 두 가지 해법을 동시에 테스트 중**:
- **m1 (안전한 방법, 메인 yaml)**: `per_device_train_batch_size: 2→1`, `gradient_accumulation_steps: 2→4` — effective batch(4)는 유지하고 forward/backward 1회당 처리하는 시퀀스 수만 절반으로 줄여 activation 메모리 감소. sequence_length는 그대로.
- **m2 (리스크 있는 방법, 별도 파일 `agentic_val_tictactoe_selfplay_pp3.yaml` + `run_agentic_pipeline_tictactoe_selfplay_pp3.sh`)**: `pipeline_model_parallel_size: 1→3` — attention head 개수와 무관하게 36개 레이어를 GPU 3장에 12개씩 나눠 담아 파라미터+옵티마이저+activation을 진짜로 분산. batch 설정은 안 건드림(비교를 위해 변수 하나만 바꿈). 리스크: vLLM 가중치 동기화용 커스텀 패치(`mcore_adapter_weight_sync_patch.py`)가 PP=1 기준으로만 검증됐고 PP=3에서 새 버그가 날 수 있음.

## `reference` 워커 causal mask OOM (m2에서 발견, rollout_batch_size=128 완료 후)

```
File ".../transformers/models/qwen3/modeling_qwen3.py", line 745, in _prepare_4d_causal_attention_mask_with_cache_position
    causal_mask = causal_mask.clone()
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.00 GiB.
```

**원인**: `reference` 워커는 Megatron이 아니라 HF `transformers`를 그대로 씀(`strategy_name: hf_infer`). `sequence_length=32768`이라 causal mask 텐서가 `(batch, 1, 32768, 32768)` 크기인데, `reference.infer_batch_size=2`라 batch=2만 돼도 몇 GB씩 잡아먹어 OOM. `actor_train` logits OOM과 원인이 같음(TP=1이라 별도 샤딩 없이 전체를 GPU 한 장이 만듦).

**수정**: `reference.infer_batch_size` 2→1.

## `rollout_batch_size` DP 분할 나머지 버그 (m3에서 발견)

```
File "roll/distributed/scheduler/protocol.py", line 525, in make_iterator
    assert self.batch.batch_size[0] % mini_batch_size == 0, f"{self.batch.batch_size[0]} % {mini_batch_size} != 0"
AssertionError: 1 % 4 != 0
```

**원인**: `DataProto.chunk()`가 `np.array_split()`을 써서 DP(데이터병렬) 랭크별로 배치를 나누는데, 나머지가 있으면 불균등하게 쪼갬(예: 128을 3으로 나누면 `[43, 43, 42]`). 그리고 `train_step`에서 각 DP 랭크는 자기 몫을 다시 `mini_batch_size = per_device_train_batch_size(2) × gradient_accumulation_steps(2) = 4` 단위로 나눠야 하는데, `43 % 4 = 3`, `42 % 4 = 2`로 0이 아니어서 assert 실패.

GPU를 8장→3장으로 줄이면서(`tensor_model_parallel_size` 4→1) DP가 2→3이 된 부작용. 원래(DP=2)는 `128/2=64`, `64%4=0`으로 문제없었음. `rollout_batch_size=4`로 테스트할 때 가장 먼저 걸렸지만, 고치지 않으면 `rollout_batch_size=128`(진짜 런, m2)에서도 언젠가 똑같이 재발할 버그였음.

**수정**: `rollout_batch_size`를 `DP(3) × mini_batch_size(4) = 12`의 배수인 **132**로 변경 (128 → 132). `132/3=44`, `44%4=0`으로 항상 균등 분할됨.

## 공유 환경(conda `mg`) 버전 호환성 수정

repo 코드가 짜여진 시점의 의존성 버전과 실제 설치된 버전이 안 맞아서 생긴 문제들. 코드 로직 버그는 아니고 전부 "라이브러리 신버전에서 API가 바뀌었다" 유형:

- `click`/`ray` 버전 충돌 (Sentinel 에러) → `click<8.2.0`으로 다운그레이드
- Ray 소켓 경로 107바이트 제한 초과 → `ray start --temp-dir=/tmp/ray` 지정 (`roll/distributed/scheduler/initialize.py`)
- Ray `LogMonitor` API 변경 (`gcs_publisher` → `gcs_client`) → 환경 재설치로 해결됨
- `transformers`에서 `AutoModelForVision2Seq` 못 찾음 → 재설치 중 타이밍 문제였음(일시적)
- `flash_attn`이 다른 torch 버전으로 빌드되어 있던 문제 → 재설치로 해결
- `megatron-core` API 변경들:
  - `legacy_a2a_token_dispatcher` 모듈 삭제됨 → fallback import 추가 (`roll/third_party/megatron/offload_states_patch.py`)
  - `_get_param_groups_and_buffers()`의 `no_weight_decay_cond`/`scale_lr_cond`/`lr_mult` 인자 삭제 → `config_overrides`로 대체 (`roll/third_party/megatron/optimizer.py`)
- `apex` 최신 버전에 레거시 `apex.amp` 서브모듈이 없어서 `transformers.trainer`의 무조건적 import가 깨짐 → 더미 `apex.amp` 모듈 추가
- `mcore_adapter` 업그레이드로 `VirtualModels.all_gather_weights_as_hf_bucket()`(가중치를 vLLM으로 스트리밍하는 제너레이터)이 삭제됨 → 새 API(`_mca_named_params_with_vp_stage`, `gather_tensor_parallel`, `convert_to_hf`, `SendBucketManager`)로 재구현 (`roll/third_party/megatron/mcore_adapter_weight_sync_patch.py` 신규 생성)
- `matplotlib` 3.10에서 `FigureCanvasAgg.tostring_rgb()` 삭제됨 (→ `buffer_rgba()`로 대체) → TicTacToe 렌더링 코드 수정 (`roll/agentic/env/tictactoe/env.py`)
- SLURM `--mem` 부족으로 OOM (실제 사용량은 전체 RAM의 일부지만 cgroup 한도 초과) → `--mem=64G` → `--mem=256G`

## `salloc` CPU 1개만 잡혀있던 문제 (yaml이 아니라 SLURM 제출 옵션)

속도가 너무 느리고 GPU/CPU 사용률은 낮은데 메모리 사용량만 높아 보이는 증상이 있었음. 확인해보니:

```
scontrol show job <jobid> | grep ReqTRES
ReqTRES=cpu=1,mem=256G,node=1,billing=1,gres/gpu=4,gres/gpu:l40s=4
```

**원인**: `salloc` 명령에 `--cpus-per-task`가 없어서 SLURM이 기본값 **1 CPU**로 잡음 (`--mem=256G`, `--gres=gpu:l40s:4`는 넉넉히 요청했지만 CPU는 빠짐). 이전 3-GPU 세션들도 전부 `cpu=1`이었음 — 처음부터 계속 이 상태.

**왜 이게 느려지는가**: rollout 단계(TicTacToe 환경 스텝, 토크나이징, Ray 스케줄링)는 CPU 작업인데, `env_groups=256`(4개 EnvironmentWorker) + actor_train(TP=4, 4프로세스) + actor_infer(vLLM) + reference + driver/scheduler 등 Ray 액터/서브프로세스 20개 넘게가 **CPU 1개를 나눠쓰기**해야 함. GPU는 그 CPU 작업이 끝나야 다음 생성 요청을 받으니 계속 대기 → GPU 사용률도 낮아짐. `mem=256G`는 별개로 넉넉히 잡혀 있어서 메모리만 커 보이는 그림이 됨.

**수정**: `salloc`에 `--cpus-per-task=32` 추가 (L40S 노드는 전부 32 CPU/4 GPU가 맥스라 32가 상한값 — 64는 요청해도 그런 노드가 없어서 영원히 pending됨). 이미 GPU 4장 + RAM 256G로 노드를 통째로 쓰고 있어서 CPU도 32개 다 가져가는 게 안전.

```
salloc --account=pi_dagarwal_umass_edu --job-name=m1 --partition=gpu-preempt --nodes=1 \
  --gres=gpu:l40s:4 --cpus-per-task=32 --mem=256G --time=08:00:00 \
  --exclude=gpu037,gpu038 --mail-user=eunbiyoon@umass.edu --mail-type=BEGIN,END,FAIL,TIME_LIMIT_10
```

## Ray placement group CPU가 노드의 절반만 잡히는 문제 (`resource_manager.py`)

`salloc --cpus-per-task=32`로 고친 뒤 Ray 대시보드에 `0.13/32.0 CPU (... 16.0 reserved in placement groups)`로 보여서 왜 16만 reserved인지 확인함:

```python
# roll/distributed/scheduler/resource_manager.py:45 (수정 전)
bundles.append({"GPU": self.gpu_per_node, "CPU": max(node_cpu / 2, 1)})
```

**원인**: GPU가 필요한 워커(actor_train/actor_infer/reference)용 placement group에 노드 CPU의 **절반만** 예약하도록 코드가 짜여 있었음(`node_cpu / 2` → 32의 절반=16).

**실제 성능에 영향 있는지 확인**: `cluster.py:132`에서 모든 워커(actor_train, actor_infer, env worker 등)는 액터당 `num_cpus=0.01`만 Ray에 요청함. 즉 placement group의 "16" 또는 "32"라는 숫자는 **Ray가 액터를 몇 개까지 동시 배치 허용할지 정하는 상한선**일 뿐(`16÷0.01=1600개`까지 허용) — 실제 액터는 20~30개뿐이라 16이든 32든 이 한도에 걸릴 일이 없음. **진짜 CPU 코어 개수를 결정하는 건 이 Ray 숫자가 아니라 SLURM cgroup(`--cpus-per-task`)** — 그건 이미 32로 고쳐서 실질적인 병목은 해결된 상태였음. 즉 이 수정은 **속도에 실질적 영향은 없을 가능성이 높지만**, Ray 대시보드 숫자를 실제 할당량(32)과 맞추기 위해 적용함.

**수정**: `max(node_cpu / 2, 1)` → `max(node_cpu, 1)` (노드 CPU 전체를 placement group에 반영).

## m1 train_step 안 entropy 계산 OOM (`roll/third_party/megatron/tensor_parallel.py`)

```
File "roll/third_party/megatron/tensor_parallel.py", line 11, in mul_reduce
    return (a * b).sum(dim=-1, keepdim=True)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.64 GiB. (4.39 GiB free)
```

**원인**: `vocab_parallel_entropy`(TP=4 환경에서 entropy 계산, `train_step`의 loss 계산 경로에서 호출)가 `(total_nnz, vocab_size/tp_size)` 크기의 텐서 여러 개(`normalized_exp_logits`, `softmax_logits`, `mul_reduce`의 중간 곱 등)를 한 번에 통째로 만듦. `roll/utils/functionals.py`의 `entropy_from_logits`(추론용)와는 별개의, `train_step` 전용 구현이라 그때는 안 고쳤음.

**수정**: `functionals.py`에서 썼던 것과 같은 시퀀스(row) 청크 처리(`chunk_size=2048`) 적용. 여기는 TP 랭크 간 `dist.all_reduce`(max, sum_exp, sum_softmax_times_logits)가 있어서, 청크마다 이 collective를 그대로 유지 — 각 행(row)의 entropy는 그 행의 vocab만 갖고 계산되므로(다른 행과 독립) 청크 단위로 나눠도 결과는 완전히 동일. 모든 TP 랭크가 같은 `total_nnz`/`chunk_size`를 쓰므로 청크 수가 랭크 간에 항상 일치해 collective가 lockstep 유지됨(이전에 겪은 weight-sync 데드락과 같은 실수를 피하려고 신경 씀).

**위 수정 자체에 버그가 있었음 (바로 발견/수정)**:
```
File "roll/third_party/megatron/tensor_parallel.py", line 43
    sum_softmax_times_logits_full[start:end] = chunk_sum_softmax_times_logits
RuntimeError: The expanded size of the tensor (1) must match the existing size (32768) ...
```
`total_nnz = vocab_parallel_logits.shape[0]`로 2차원(`total_nnz, vocab_shard`)이라고 가정했는데, 실제 호출부(`op_compute_entropy`)에서는 **3차원**(`batch, seq_len, vocab_shard`)으로 들어옴. `batch=1`이라 `shape[0]=1`이 되어 청크가 사실상 안 나뉘고 전체(32768)를 한 번에 처리하면서, 미리 만든 작은 버퍼(`[1,1]`)에 큰 결과(`[32768,1]`)를 넣으려다 shape 에러. **재수정**: 입력을 `reshape(-1, vocab_shard_size)`로 먼저 2차원으로 펴서 처리하고, 끝나면 `view(orig_shape[:-1])`로 원래 shape(batch, seq_len)로 되돌리도록 변경 — 이제 2D/3D 어떤 모양으로 들어와도 올바르게 청크 처리됨.

## 체크포인트 저장 실패: `ShardedTensor.flattened_range is not supported` (m2에서 발견, save_steps=20 첫 저장 시)

```
File ".../megatron/core/dist_checkpointing/mapping.py", line 134, in validate_metadata_integrity
    raise CheckpointingException("ShardedTensor.flattened_range is not supported.")
```

**원인**: `use_distributed_optimizer: true`일 때 Megatron이 옵티마이저 상태를 파라미터 버킷 단위로 쪼개서 저장하는데, 그 버킷 슬라이스(`flattened_range`)를 체크포인트에 기록하는 기능이 지금 설치된 megatron-core 버전에서 무조건 `raise`하도록 막혀 있음(버전 제약/미구현으로 보임). 메모리 문제 아니라 순수 체크포인트 저장 경로 버그.

`use_distributed_optimizer`는 DP(데이터병렬) 랭크들 사이에 옵티마이저 상태를 나눠 담아 메모리를 아끼는 기능인데, 지금은 `TP=4, DP=1`이라 나눠 담을 DP 랭크가 1개뿐 — 켜봤자 메모리 절감 효과가 전혀 없으면서 이 버그만 유발함.

**수정**: `actor_train.strategy_args.strategy_config.use_distributed_optimizer: true → false`. DP=1이라 메모리 손해 없음.

## 체크포인트 저장 실패 (2차): `mpu.get_data_modulo_expert_parallel_rank` 이름 바뀜

`use_distributed_optimizer=false`로 고친 뒤 `flattened_range` 에러는 사라졌지만, 체크포인트 저장 코드의 그 다음 줄에서 또 다른 버전 호환성 에러 발생:

```
File "roll/distributed/strategy/megatron_strategy.py", line 510
    elif not dist.is_initialized() or mpu.get_data_modulo_expert_parallel_rank() == 0:
AttributeError: module 'megatron.core.parallel_state' has no attribute 'get_data_modulo_expert_parallel_rank'
```

**원인**: 지금 설치된 megatron-core 버전에서 이 함수가 `get_expert_data_parallel_rank`로 이름이 바뀜(다른 곳, 예: mcore_adapter의 `save_model_as_hf_inflight`에서도 이미 새 이름을 쓰고 있음 — 리포지토리의 이 파일만 옛 이름을 참조하고 있었음). 로직/의미는 동일(DP replica 중 하나만 optimizer state 저장), 순수 API 이름 변경.

**수정**: `mpu.get_data_modulo_expert_parallel_rank()` → `mpu.get_expert_data_parallel_rank()`.

## 빠른 확인용 설정/스크립트 분리 (`_quickcheck`)

m1(진짜 설정, `rollout_batch_size=128`)과 m2(버그/체크포인트 등 빨리 확인용, `rollout_batch_size=12`)를 같은 yaml 파일로 계속 왔다갔다 수정하면서 돌리다가, 실수로 m2가 128로 시작돼버리는 사고가 생김(파일 수정 타이밍을 놓침). **PP=3 테스트할 때처럼 별도 파일로 완전히 분리**:

- `examples/tictactoe/agentic_val_tictactoe_selfplay_quickcheck.yaml`: 메인 yaml 복사본, `rollout_batch_size: 12`만 다름
- `examples/tictactoe/run_agentic_pipeline_tictactoe_selfplay_quickcheck.sh`: `--config_name agentic_val_tictactoe_selfplay_quickcheck` 사용, 출력 디렉토리도 `runs/tictactoe_selfplay_quickcheck/`로 분리(진짜 학습 run들과 안 섞이게)

앞으로 m2에서 빠른 버그/수정 확인이 필요하면 이 스크립트를 쓰고, 메인 yaml(`agentic_val_tictactoe_selfplay.yaml`)은 항상 "진짜 설정"으로 유지.

## TP=4에서 가중치 동기화 NCCL 데드락 (30분 타임아웃)

GPU 4장/TP=4로 전환한 뒤 rollout이 시작되기도 전, 모델 초기화 직후 actor_train→actor_infer(vLLM) 가중치 동기화 단계에서 30분간 멈췄다가 죽음:

```
[Rank 3] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=12, OpType=GATHER, ...) 
ran for 1800074 milliseconds before timing out.
[PG GUID model_update_actor_train_2_to_actor_infer_(3,0) Rank 1] ... OpType=BROADCAST ... timing out.
Fatal Python error: Aborted
```

**원인**: `roll/distributed/strategy/megatron_strategy.py:411` `model_update()`를 보면, `tensor_model_parallel_size > 1`일 때 **TP 랭크마다 각자 자기 몫의 actor_infer 타겟에 독립적으로 `collective.broadcast()`를 호출**하도록 설계되어 있음(`tp_rank==0` 조건 없이 매 랭크가 수행). 그런데 제가 만든 `roll/third_party/megatron/mcore_adapter_weight_sync_patch.py`가 `gather_tensor_parallel`(`dist.gather`, **결과를 tp_rank==0에만 줌**)을 써서, tp_rank 1~3은 매 파라미터마다 `weights is None`으로 `continue` — 즉 **자기 몫의 브로드캐스트를 한 번도 호출 안 함**. 반대편 actor_infer는 그 브로드캐스트를 계속 기다리다 NCCL 기본 타임아웃(30분)으로 죽음.

(LoRA는 원인이 아니었음 — TP=1에서는 `dist.gather`가 랭크 1개짜리라 사실상 no-op이라 문제가 안 드러났을 뿐, TP=4가 되면서 처음으로 진짜 collective가 되어 구조적 설계 불일치가 드러난 것.)

mcore_adapter 자체의 `save_model_as_hf_inflight`(디스크 저장용, 이 패치가 참고했던 원본)는 `gather_tensor_parallel`을 쓰는 게 맞음 — 파일 하나만 쓰면 되니 tp_rank==0만 있어도 충분. 하지만 이 패치의 용도(브로드캐스트 fan-out)는 **모든 랭크가 병합된 가중치를 들고 있어야** 각자 자기 브로드캐스트를 수행할 수 있음.

**수정**: `gather_tensor_parallel` → `all_gather_tensors`(`mcore_adapter.models.converter.convert_utils`에 이미 존재, `torch.distributed.all_gather` 기반 — 모든 랭크가 결과를 받음)로 교체. `weights is None` 스킵 로직도 제거(이제 모든 랭크가 유효한 데이터를 받으므로 불필요).

## 전체 파이프라인 단계 순서 (`agentic_pipeline.py`)

```
for global_step in 0..max_steps(200):
    rollout                      — train_env로 트라젝토리 수집
    cal_ref_log_probs            — reference 모델로 KL penalty용 log_probs 계산 (OOM 났던 지점, 수정함)
    cal_old_log_probs_values     — actor_train 자신의 (업데이트 전) log_probs + critic values 계산
    adv                          — 리워드 정규화, advantage 계산
    (gradient 업데이트)           — 실제 PPO loss 계산 + actor_train 파라미터 업데이트
                                    ← 여기까지 끝나야 "스텝 1 완료"
```
이 사이클을 최대 200번 반복. `val`(검증)은 이 루프 안에서 `eval_steps`마다 한 번씩 끼어들 뿐, 학습(gradient 업데이트) 경로와는 완전히 분리되어 있음 (`eval_batch`는 로깅 후 바로 `del`).
