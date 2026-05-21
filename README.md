# Jetson Model Inference & Fault Injection Workspace

This repository contains scripts and tools for converting PyTorch models to TensorRT, running inference benchmarks on Jetson hardware, and conducting fault injection campaigns.

## Repository Structure

### 1. `model_conversion/`
Contains scripts used to convert PyTorch models into optimized TensorRT engines.
* Includes a test configuration evaluating a model with `batch_size = 10`.
* Default models are located under the `standard_models/` directory.

### 2. `model_inference/`
Contains scripts dedicated to running standard model inference pipelines and checking baseline performance.

### 3. `trt_inference/`
Contains optimized scripts specifically tailored for executing TensorRT (`.plan` / `.engine`) model inference and verifying execution correctness.

### 4. `nvbit_release/tools/nvbitfi/`
The main environment for the hardware fault injection campaign.
* Currently running fault injection tests via the script: `test_sb3net_trt.sh`
* *Note:* The active serialized TensorRT engine (`.plan` file) might currently be configured with `batch_size = 4`. However, the core setup, pipeline behavior, and encountered errors remain identical.
