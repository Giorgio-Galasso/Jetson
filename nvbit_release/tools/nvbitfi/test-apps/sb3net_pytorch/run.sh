#!/bin/bash
# NVBitFI inietterà il profiler/injector tramite la variabile PRELOAD_FLAG
# Definiamo il percorso assoluto della cartella dell'applicazione
APP_DIR="/home/g.galasso/g.galasso/nvbit_release/tools/nvbitfi/test-apps/sb3net_pytorch"

# Eseguiamo Python con il percorso completo del file .py
# Usiamo -u per evitare il buffering dell'output (fondamentale per i log)
eval ${PRELOAD_FLAG} python3 -u ${APP_DIR}/sb3net_pytorch.py > stdout.txt 2> stderr.txt
