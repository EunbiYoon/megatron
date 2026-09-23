import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu


class _VocabParallelEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor, chunk_size: int = 2048) -> torch.Tensor:
        @torch.compile(dynamic=True)
        def mul_reduce(a, b):
            return (a * b).sum(dim=-1, keepdim=True)

        group = mpu.get_tensor_model_parallel_group()
        # Caller may pass either a 2D (total_nnz, vocab_shard) tensor or an unflattened
        # (..., vocab_shard) tensor (e.g. (batch, seq_len, vocab_shard)). Flatten to 2D so
        # chunking below always splits over the true row count, not just dim 0.
        orig_shape = vocab_parallel_logits.shape
        vocab_shard_size = orig_shape[-1]
        flat_logits = vocab_parallel_logits.reshape(-1, vocab_shard_size)
        total_nnz = flat_logits.shape[0]

        # Process rows in chunks to bound peak memory: each row's entropy only depends on
        # that row's own vocab slice (reduced across TP ranks), so chunking over dim 0 is
        # exact (not an approximation) as long as every TP rank uses the same chunk boundaries
        # (guaranteed here since total_nnz is identical on all ranks and chunk_size is fixed),
        # which keeps the per-chunk all_reduce calls in lockstep across ranks.
        softmax_logits = torch.empty_like(flat_logits)
        sum_softmax_times_logits_full = torch.empty(
            total_nnz, 1, dtype=flat_logits.dtype, device=flat_logits.device
        )
        entropy_chunks = []
        for start in range(0, total_nnz, chunk_size):
            end = min(start + chunk_size, total_nnz)
            chunk_logits = flat_logits[start:end]

            logits_max = chunk_logits.max(dim=-1, keepdim=True).values
            dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=group)
            normalized_chunk_logits = chunk_logits - logits_max
            normalized_exp_logits = normalized_chunk_logits.exp_()
            normalized_sum_exp_logits = normalized_exp_logits.sum(dim=-1, keepdim=True)
            dist.all_reduce(normalized_sum_exp_logits, group=group)
            chunk_softmax_logits = normalized_exp_logits.div_(normalized_sum_exp_logits)
            chunk_sum_softmax_times_logits = mul_reduce(chunk_softmax_logits, chunk_logits)
            dist.all_reduce(chunk_sum_softmax_times_logits, group=group)
            chunk_entropy = logits_max + normalized_sum_exp_logits.log() - chunk_sum_softmax_times_logits

            entropy_chunks.append(chunk_entropy.squeeze(dim=-1))
            softmax_logits[start:end] = chunk_softmax_logits
            sum_softmax_times_logits_full[start:end] = chunk_sum_softmax_times_logits

        # Save the flattened views (same underlying storage as the original input) so
        # backward's in-place ops mutate the same memory the original code relied on.
        ctx.save_for_backward(flat_logits, softmax_logits, sum_softmax_times_logits_full)
        ctx.orig_shape = orig_shape
        entropy_flat = torch.cat(entropy_chunks, dim=0)
        return entropy_flat.view(orig_shape[:-1])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        vocab_parallel_logits, softmax_logits, sum_softmax_times_logits = ctx.saved_tensors
        grad_output_flat = grad_output.reshape(-1)
        # reuse softmax_logits as grad
        vocab_parallel_logits.sub_(sum_softmax_times_logits)
        softmax_logits.mul_(vocab_parallel_logits)
        softmax_logits.mul_(grad_output_flat.unsqueeze(dim=-1))
        # recover vocab_parallel_logits
        vocab_parallel_logits.add_(sum_softmax_times_logits)
        softmax_logits.mul_(-1)
        return softmax_logits.view(ctx.orig_shape)


def vocab_parallel_entropy(vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
    """
    ref: https://github.com/volcengine/verl/blob/78532923368aeb058f62201489546d013df47710/verl/utils/megatron/tensor_parallel.py#L109
    Compute entropy when the logits are sharded in tp ranks

    Args:
        vocab_parallel_logits: (..., vocab_size // tp_size), e.g. (total_nnz, vocab_size // tp_size)
            or (batch, seq_len, vocab_size // tp_size)

    Returns: (...,) i.e. the input shape with the last (vocab) dimension removed

    """
    return _VocabParallelEntropy.apply(vocab_parallel_logits)
