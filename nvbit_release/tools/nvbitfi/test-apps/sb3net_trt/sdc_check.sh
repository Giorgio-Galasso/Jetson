#!/bin/bash

# Configurazione del percorso dinamico per la cartella TensorRT
APP_DIR_DYNAMIC="$NVBITFI_HOME/test-apps/sb3net_trt"

# (1) Creazione del file diff.log richiesto dal framework
touch diff.log

# (2) Confronto dello stdout con il golden file di TensorRT
diff stdout.txt "${APP_DIR_DYNAMIC}/golden_stdout.txt" > stdout_diff.log

# (3) Confronto dello stderr
diff stderr.txt "${APP_DIR_DYNAMIC}/golden_stderr.txt" > stderr_diff.log

# (4) Special check basato sull'output di trt_inference.py
# Cerchiamo la stringa "Azione predetta" invece di "FINAL_ACTIONS"
grep "Azione predetta" stdout.txt > selected_output.txt
grep "Azione predetta" "${APP_DIR_DYNAMIC}/golden_stdout.txt" > selected_golden_output.txt

# Se l'azione del modello TensorRT cambia a causa del guasto, special_check.log non sara vuoto
diff selected_output.txt selected_golden_output.txt > special_check.log
