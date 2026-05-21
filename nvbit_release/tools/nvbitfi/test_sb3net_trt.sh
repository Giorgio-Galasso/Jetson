#!/bin/bash

# STEP 0: Environment Configuration (Jetson Nano & TensorRT)
export CUDA_HOME=/usr/local/cuda
export NVBITFI_HOME=$(pwd)
export PATH=$CUDA_HOME/bin:/usr/src/tensorrt/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
export CPATH=$CUDA_HOME/include:$CPATH
export PYTHONPATH=/usr/lib/python3.6/dist-packages:$PYTHONPATH

# NVBitFI internal variables
export NOBANNER=1
export TOOL_VERBOSE=0
export VERBOSE=0

# Application name registered in params.py
APP_NAME="sb3net_trt"

echo "=== Starting NVBitFI Pipeline (TensorRT) ==="
echo "Target application: $APP_NAME"

# STEP 1: Injection Site Generation (Profiling)
echo ""
echo "--- Step 1: Profiling and generating injection list ---"
rm -rf logs/profiling_results/
rm -rf logs/${APP_NAME}/

# Run the profiler first to generate nvbitfi-igprofile.txt
python3 scripts/run_profiler.py $APP_NAME

if [ $? -ne 0 ]; then
    echo "Error during profiling execution. Aborting."
    exit 1
fi

# Generate the injection list based on the profile
python3 scripts/generate_injection_list.py standalone $APP_NAME

if [ $? -ne 0 ]; then
    echo "Error during Step 1 (Generating list). Aborting."
    exit 1
fi

# STEP 2: Error Injection Campaign Execution
echo ""
echo "--- Step 2: Running error injection campaign ---"

python3 scripts/run_injections.py standalone $APP_NAME

if [ $? -ne 0 ]; then
    echo "Error during Step 2 (Injection). Aborting."
    exit 1
fi

# STEP 3: Parsing and Data Collection
echo ""
echo "--- Step 3: Result parsing completed ---"
echo "Detailed SDC logs are available in: logs/results/"
echo "=== Campaign completed successfully ==="
