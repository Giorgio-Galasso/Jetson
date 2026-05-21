#!/bin/bash
# NVBitFI inietterà il profiler/injector tramite la variabile PRELOAD_FLAG

# Ottiene in automatico il percorso della cartella dove si trova questo script.
# Così non devi preoccuparti dei doppi "g.galasso" o del nome della cartella!
APP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Eseguiamo Python usando il percorso dinamico.
#CAMBIA 'trt_inference.py' con il nome del file della tua nuova applicazione
eval ${PRELOAD_FLAG} python3 -u "${APP_DIR}/trt_inference.py" > stdout.txt 2> stderr.txt
