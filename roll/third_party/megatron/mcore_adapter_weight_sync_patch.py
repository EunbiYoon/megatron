"""
Newer mcore_adapter dropped `VirtualModels.all_gather_weights_as_hf_bucket` /
`ModelConverter.all_gather_weights_as_hf_bucket` (the streaming bucket generator used to
broadcast trained weights to the vLLM/SGLang inference workers) when it refactored weight
export around `save_model_as_hf_inflight`'s per-parameter iteration. This restores an
equivalent generator on top of the current API (`_mca_named_params_with_vp_stage`,
`all_gather_tensors`, `convert_to_hf`) instead of writing shards to disk.

Note: `save_model_as_hf_inflight` (mcore_adapter's own disk-saving path, which this was
modeled after) uses `gather_tensor_parallel` (dist.gather -> only tp_rank==0 gets the
merged weight), which is correct for its use case since only one rank needs to write the
shared checkpoint file. This generator's caller (`megatron_strategy.py: model_update`) is
different: with tensor_model_parallel_size > 1, EVERY tp_rank independently broadcasts its
bucket to its own paired actor_infer target via `collective.broadcast(..., group_name=comm_plan[...])`
(unconditional on tp_rank, see `p2p_tgt_devices`/`broadcast_tgt_devices` handling). Using
`gather_tensor_parallel` here made tp_rank>0 skip every parameter (weights=None -> continue),
so those ranks never yielded a bucket and thus never called their own `collective.broadcast`,
leaving their paired actor_infer targets waiting on a NCCL broadcast that would never arrive
(30-minute NCCL watchdog timeout). `all_gather_tensors` gives every rank the merged result so
every rank yields buckets and performs its own broadcast.
"""
import torch.distributed as dist
from megatron.core import mpu

from mcore_adapter.models.converter.convert_utils import SendBucketManager, all_gather_tensors
from mcore_adapter.models.converter.model_converter import ModelConverter
from mcore_adapter.models.model_factory import VirtualModels


def _all_gather_weights_as_hf_bucket(self: "VirtualModels", models=None, bucket_size: int = None):
    models = models or self.models
    converter = ModelConverter(self.config, to_hf=True)
    bucket_manager = SendBucketManager(bucket_size or converter._auto_bucket_size())

    expert_parallel = converter.mca_config.expert_model_parallel_size > 1
    only_need_expert = expert_parallel and mpu.get_expert_model_parallel_rank() > 0

    for _adapter_name, vp_stage, mca_name, weight in converter._mca_named_params_with_vp_stage(models):
        ep_group = mpu.get_expert_model_parallel_group()
        ep_world_size = dist.get_world_size(ep_group)
        if converter._needs_moe_allgather(mca_name, ep_world_size):
            converted_state_dict = converter._convert_moe_weight_with_allgather(
                mca_name, weight, vp_stage, ep_group, ep_world_size,
                move_to_cpu=False, only_need_expert=only_need_expert,
            )
            if converted_state_dict is None:
                continue
        else:
            if only_need_expert and not converter.dist_converter.is_expert_parallel_weight(mca_name):
                continue
            weights = all_gather_tensors(weight, group=mpu.get_tensor_model_parallel_group(), async_op=False)
            converted_state_dict = converter.convert_to_hf(mca_state_dict={mca_name: weights}, vp_stage=vp_stage)

        for hf_name, hf_weight in (converted_state_dict or {}).items():
            yield from bucket_manager.push_tensor(hf_weight, name=hf_name)

    last_meta, last_buffer = bucket_manager.pop_last_bucket()
    if last_meta is not None:
        yield last_meta, last_buffer


if not hasattr(VirtualModels, "all_gather_weights_as_hf_bucket"):
    VirtualModels.all_gather_weights_as_hf_bucket = _all_gather_weights_as_hf_bucket
