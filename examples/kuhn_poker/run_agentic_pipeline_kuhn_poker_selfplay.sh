#!/bin/bash
set +x
export RAY_TMPDIR=/tmp
ray stop

# torch is built against CUDA 12.4 (torch.version.cuda), so pin nvcc/CUDA_HOME to
# the matching module instead of whatever "cuda" module happens to be loaded
# (mismatched nvcc breaks building CUDA extensions, e.g. apex).
module unload cuda >/dev/null 2>&1
module load cuda/12.4.1
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

# transformer_engine dlopen's libcudnn/libcublas at import time; without this,
# actor_train workers crash with "libcudnn_adv.so.9: cannot open shared object file"
export LD_LIBRARY_PATH=$(python -c "import nvidia.cudnn, os; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), 'lib'))"):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(python -c "import nvidia.cublas, os; print(os.path.join(os.path.dirname(nvidia.cublas.__file__), 'lib'))"):$LD_LIBRARY_PATH

# vLLM/torch.compile's Triton autotune cache defaults to $HOME/.cache/triton, which
# is on NFS here and makes first-time kernel compilation much slower. Use local
# node disk instead so compiled kernels are cached fast across runs on the same node.
export TRITON_CACHE_DIR=/tmp/triton_cache_$(whoami)
mkdir -p "$TRITON_CACHE_DIR"

CONFIG_PATH=$(basename $(dirname $0))

ROLL_PATH=${PWD}
export PYTHONPATH="$ROLL_PATH:$PYTHONPATH"

ROLL_OUTPUT_DIR="./runs/kuhn_poker_selfplay/$(date +%Y%m%d-%H%M%S)"
ROLL_LOG_DIR=$ROLL_OUTPUT_DIR/logs
ROLL_RENDER_DIR=$ROLL_OUTPUT_DIR/render
export ROLL_OUTPUT_DIR=$ROLL_OUTPUT_DIR
export ROLL_LOG_DIR=$ROLL_LOG_DIR
export ROLL_RENDER_DIR=$ROLL_RENDER_DIR
mkdir -p $ROLL_LOG_DIR $ROLL_RENDER_DIR

python examples/start_agentic_pipeline.py --config_path $CONFIG_PATH  --config_name agentic_val_kuhn_poker_selfplay | tee $ROLL_LOG_DIR/custom_logs.log
