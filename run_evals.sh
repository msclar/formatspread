#!/bin/bash
#SBATCH --job-name=format
#SBATCH --account=a100-sage
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:8
#SBATCH --time=100:00:00
#SBATCH --output=slurm_logs/generate_plans_%j.log

# Load any necessary modules (e.g., Python, CUDA)
#module load python/3.8  # Adjust the version as needed
#module load cuda/11.2   # Adjust the version as needed

# Activate your virtual environment if needed
source ~/earlyenv/bin/activate

# Get the model name and command variation from the command-line arguments
MODEL_NAME=$1
COMMAND_VARIATION=$2

# Check if the model name and command variation are provided
if [ -z "$MODEL_NAME" ] || [ -z "$COMMAND_VARIATION" ]; then
    echo "Error: Model name and command variation must be provided."
    exit 1
fi

echo "Running model: $MODEL_NAME"

# Start the vllm server
vllm serve $MODEL_NAME --gpu-memory-utilization 0.9 --tensor-parallel-size 8 --download-dir /data/home/melaniesclar/lotsofdata/vllm_cache --disable-log-requests &

# Wait until the vllm server is available
while true; do
    STATUS_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://0.0.0.0:8000)
   echo $STATUS_CODE
    if [ "$STATUS_CODE" -eq 404 ]; then
        echo "vllm server is available."
        break
    else
        echo "Waiting for vllm server to become available..."
        sleep 5
    fi
done


# Define task sets for each command variation
case $COMMAND_VARIATION in
    1)
        TASKS=("task050_" "task065_" "task069_" "task070_")
        ;;
    2)
        TASKS=("task114_" "task133_" "task155_" "task158_")
        ;;
    3)
        TASKS=("task161_" "task162_" "task163_")
        ;;
    4)
        TASKS=("task190_" "task213_" "task214_" "task220_")
        ;;
    5)
        TASKS=("task279_" "task280_" "task286_")
        ;;
    6)
        TASKS=("task296_" "task297_" "task316_" "task317_")
        ;;
    7)
        TASKS=("task319_" "task320_" "task322_")
        ;;
    8)
        TASKS=("task323_" "task325_" "task326_" "task327_")
        ;;
    9)
        TASKS=("task328_" "task335_" "task337_")
        ;;
    10)
        TASKS=("task385_" "task580_" "task607_" "task608_")
        ;;
    11)
        TASKS=("task609_" "task904_" "task905_")
        ;;
    12)
        TASKS=("task1186_" "task1283_" "task1284_")
        ;;
    13)
        TASKS=("task1297_" "task1347_" "task1387_")
        ;;
    14)
        TASKS=("task1419_" "task1420_" "task1421_")
        ;;
    15)
        TASKS=("task1423_" "task1502_" "task1612_")
        ;;
    16)
        TASKS=("task1678_" "task1724_")
        ;;
    *)
        echo "Error: Invalid command variation."
        exit 1
        ;;
esac

# Run each task in parallel
for TASK in "${TASKS[@]}"; do
    python main.py \
        --task_filename $TASK \
        --dataset_name natural-instructions \
        --num_formats_to_analyze 499 \
        --num_samples 1000 \
        --model_name $MODEL_NAME \
        --model_access_method vllm-api \
        --n_shot 5 \
        --evaluation_metric exact_prefix_matching \
        --evaluation_type format_spread \
        --num_formats_format_spread 320 \
        --batch_size_format_spread 20 \
        --budget_format_spread 40000 &
done

# Wait for all background processes to finish
wait