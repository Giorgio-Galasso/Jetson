#!/bin/bash
# Ferma lo script in caso di errore
#set -e 

CWD=`pwd`
echo "Current working directory: $CWD"

###############################################################################
# Step 0: Setup Ambiente e Strumenti
###############################################################################

# 0(1): Permessi di esecuzione agli script
find . -name "*.sh" | xargs chmod +x 

# 0(2): Variabili d'ambiente (CUDA 11.6)
printf "\nStep 0 (2): Setting environment variables for CUDA 11.6\n"
export NOBANNER=1
export TOOL_VERBOSE=0
export VERBOSE=0

export CUDA_BASE_DIR=/usr/local/cuda
export NVBITFI_HOME=$(pwd)
export PATH=$CUDA_BASE_DIR/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_BASE_DIR/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDA_BASE_DIR/include:$CPATH
export NVCC_PREPEND_FLAGS="-I $CUDA_BASE_DIR/include"

echo "Environment ready. Using NVCC: $(which nvcc)"

# 0(3): Compilazione dei tool NVBit (Injector e Profiler)
printf "\nStep 0 (3): Build the nvbitfi tools\n"
cd injector && make && cd ..
cd profiler && make && cd ..

###############################################################################
# Step 0 (4): Golden Run per PyTorch (MODIFICATO PER SB3NET)
###############################################################################
printf "\nStep 0 (4): Generating Golden Output for PyTorch Model\n"
# Entriamo nella cartella della tua tesi
APP_DIR="test-apps/sb3net_pytorch"

if [ -d "$APP_DIR" ]; then
    cd $APP_DIR
    # Non usiamo 'make' perché PyTorch è interpretato.
    # Generiamo il file golden_stdout.txt eseguendo l'inferenza pulita.
    python3 sb3net_pytorch.py > golden_stdout.txt 2> golden_stderr.txt
    cd $CWD
else
    echo "ERRORE: Cartella $APP_DIR non trovata!"
    exit 1
fi

###############################################################################
# Step 1: Profiling e Injection List
###############################################################################
cd scripts/

printf "\nStep 1 (1): Profile the application\n"
# Ricorda: run_profiler legge 'apps' da params.py
python3 run_profiler.py
#rm -f stdout.txt stderr.txt

printf "\nStep 1 (2): Generate injection list\n"
python3 generate_injection_list.py 

###############################################################################
# Step 2 & 3: Campagna e Risultati
###############################################################################
printf "\nStep 2: Run the error injection campaign\n"
# 'standalone' significa che usi una sola GPU (la tua Jetson/3060ti)
python3 run_injections.py standalone 

printf "\nStep 3: Parse results\n"
python3 parse_results.py

cd $CWD
echo "--- FINE CAMPAGNA ---"
