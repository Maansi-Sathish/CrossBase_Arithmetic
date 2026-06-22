#!/bin/bash
# ==============================================================================
# Architecture Sweep: Grokking Search
# ==============================================================================

# TODO: Replace with your actual Hugging Face username or organization
HF_USERNAME="CrossBaseArithmetic"
WANDB_PROJECT_NAME="CrossBaseArithmetic"

# Force WandB to route all these runs into the exact same project dashboard
export WANDB_PROJECT=$WANDB_PROJECT_NAME

# The grid: 3x4x4 =  48 configs
DIMS=(32 64 128 256)
LAYERS=(1 2 3)
HEADS=(2 4 8 16)

echo "Starting Architecture Sweep. Total possible combinations: 64."
echo "Monitoring via WandB project: $WANDB_PROJECT_NAME"
echo "=============================================================================="

for dim in "${DIMS[@]}"; do
  for layers in "${LAYERS[@]}"; do
    for heads in "${HEADS[@]}"; do

      # Transformer requires hidden_size to be divisible by num_attention_heads.
      # Skip configs that violate this.
      if [ $((dim % heads)) -ne 0 ]; then
        echo "Skipping invalid config: dim=$dim is not divisible by heads=$heads."
        continue
      fi

      RUN_NAME="L${layers}_H${heads}_D${dim}"
      HUB_REPO="${HF_USERNAME}/arithmetic-model-${RUN_NAME}"
      OUT_DIR="./saved_models/${RUN_NAME}"

      echo ""
      echo "------------------------------------------------------------------------"
      echo "Launching Run: $RUN_NAME"
      echo "   Hidden Size : $dim"
      echo "   Layers      : $layers"
      echo "   Attn Heads  : $heads"
      echo "   HF Hub Repo : $HUB_REPO"
      echo "------------------------------------------------------------------------"

      python train_model.py \
        --hidden-size "$dim" \
        --num-hidden-layers "$layers" \
        --num-attention-heads "$heads" \
        --run-name "$RUN_NAME" \
        --hub-name "$HUB_REPO" \
        --output-dir "$OUT_DIR" \
        --report-to wandb

      EXIT_CODE=$?
      if [ $EXIT_CODE -ne 0 ]; then
        echo "Error during $RUN_NAME. Exit code: $EXIT_CODE. Continuing to next run."
      else
        echo "Completed $RUN_NAME successfully."
      fi

    done
  done
done

echo ""
echo "Sweep complete. Check WandB to analyze train/val accuracy curves."