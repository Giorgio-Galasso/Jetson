#!/bin/bash

# # Definiamo il percorso base della tua app per sicurezza
# BASE_DIR="/home/g.galasso/nvbit_release/tools/nvbitfi/test-apps/sb3net_pytorch"

# # (1) Creiamo il file diff.log vuoto
# touch diff.log 

# # (2) Confronto stdout usando il percorso assoluto per il file Golden
# diff stdout.txt ${BASE_DIR}/golden_stdout.txt > stdout_diff.log

# # (3) Confronto stderr
# diff stderr.txt ${BASE_DIR}/golden_stderr.txt > stderr_diff.log

# # (4) Special check per le azioni del drone
# grep "FINAL_ACTIONS" stdout.txt > current_actions.txt 
# grep "FINAL_ACTIONS" ${BASE_DIR}/golden_stdout.txt > golden_actions.txt 

# # Se le azioni cambiano, special_check.log conterrà la differenza
# diff current_actions.txt golden_actions.txt > special_check.log

#!/bin/bash

# Invece di hardcoded path, ricaviamo la cartella dell'app dinamicamente
# Questo funziona se NVBITFI_HOME è esportato correttamente
APP_DIR_DYNAMIC="$NVBITFI_HOME/test-apps/sb3net_pytorch"

# (1) Se non ci sono file extra, creiamo un diff.log vuoto come da istruzioni NVIDIA
touch diff.log 

# (2) Confronto dello stdout
# Usiamo la variabile dinamica per trovare il file golden
diff stdout.txt "${APP_DIR_DYNAMIC}/golden_stdout.txt" > stdout_diff.log

# (3) Confronto dello stderr
diff stderr.txt "${APP_DIR_DYNAMIC}/golden_stderr.txt" > stderr_diff.log

# (4) Special check per il tuo modello di droni
# MODIFICA CRUCIALE: 'sum' diventa 'FINAL_ACTIONS'
grep "FINAL_ACTIONS" stdout.txt > selected_output.txt 
grep "FINAL_ACTIONS" "${APP_DIR_DYNAMIC}/golden_stdout.txt" > selected_golden_output.txt 

# Se le azioni differiscono, special_check.log NON sarà vuoto
diff selected_output.txt selected_golden_output.txt > special_check.log

# Uscita pulita per evitare esiti "Uncategorized"
exit 0