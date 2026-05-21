import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import time
import os
from pathlib import Path

# ==========================================
# CONFIGURAZIONE E UTILITY
# ==========================================
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
BATCH_SIZE = 4  # Definiamo esplicitamente la dimensione del batch

def load_engine(plan_path):
    if not os.path.exists(plan_path):
        raise FileNotFoundError(f"File non trovato: {plan_path}")
    else:
        print("File trovato!")

    with open(plan_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        print("Engine caricato!")
        return engine

def np_dtype_from_trt(dtype: trt.DataType):
    """Converte i tipi TensorRT in tipi NumPy."""
    if dtype == trt.DataType.FLOAT:   return np.float32
    if dtype == trt.DataType.HALF:    return np.float16
    if dtype == trt.DataType.INT8:    return np.int8
    if dtype == trt.DataType.INT32:   return np.int32
    if dtype == trt.DataType.BOOL:    return np.bool_
    if dtype == trt.DataType.UINT8:   return np.uint8
    raise NotImplementedError(f"Tipo {dtype} non supportato.")

# ==========================================
# ALLOCAZIONE MEMORIA (Aggiornata per contesti dinamici)
# ==========================================
def allocate_bindings(engine, context):
    """Versione ottimizzata per shape dinamiche (legge dal contesto)"""
    host_inout = {}
    device_inout = {}
    bindings_ptrs = []
    stream = cuda.Stream()

    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        is_input = engine.binding_is_input(i)
        dtype = engine.get_binding_dtype(i)
        np_dtype = np_dtype_from_trt(dtype)

        # NOTA: Interroghiamo il context invece dell'engine per avere le dimensioni
        # reali calcolate dopo l'impostazione del Batch Size dinamico
        shape = tuple(context.get_binding_shape(i))
        vol = int(np.prod(shape))

        # Buffer CPU e GPU
        host_mem = cuda.pagelocked_empty(vol, dtype=np_dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings_ptrs.append(int(device_mem))

        host_inout[name] = {"is_input": is_input, "buffer": host_mem, "shape": shape}
        device_inout[name] = device_mem

    return bindings_ptrs, host_inout, device_inout, stream

# ==========================================
# LOGICA DI TEST (MAIN)
# ==========================================
def main():
    # Inizializzazione manuale sicura per NVBitFI (NIENTE autoinit)
    cuda.init()
    device = cuda.Device(0)
    ctx = device.make_context()

    try:
        np.random.seed(42)
        plan_path = "/home/g.galasso/g.galasso/nvbit_release/tools/nvbitfi/test-apps/sb3net_trt/sb3net.plan"

        print(f"Caricamento engine: {plan_path}...")
        engine = load_engine(plan_path)
        context = engine.create_execution_context()

        # Configurazione delle dimensioni dinamiche del contesto prima dell'allocazione
        print(f"Configurazione del Context per Batch Size = {BATCH_SIZE}...")
        for i in range(engine.num_bindings):
            if engine.binding_is_input(i):
                shape = engine.get_binding_shape(i)
                # Se il primo elemento (batch) è dinamico (-1), lo forziamo a BATCH_SIZE
                if shape[0] == -1:
                    new_shape = (BATCH_SIZE,) + tuple(shape[1:])
                    context.set_binding_shape(i, new_shape)

        # Alloca i buffer basandosi sulle nuove shape da 10 elementi
        bindings_ptrs, host_inout, device_inout, stream = allocate_bindings(engine, context)

        print(f"\n1. Generazione di {BATCH_SIZE} input paralleli...")
        for name, meta in host_inout.items():
            if meta["is_input"]:
                # Genera un unico blocco contenente i 10 campioni accodati
                dummy_data = np.random.random(meta["shape"]).astype(np.float32)
                meta["buffer"][:] = dummy_data.ravel()
                print(f"    -> Input '{name}' pronto. Shape allocata su host: {meta['shape']}")

        print("\n2. Spostamento dati e calcolo inferenza parallela sulla GPU...")
        start_time = time.time()

        # H2D: Unica copia per tutti i 10 ingressi
        for name, meta in host_inout.items():
            if meta["is_input"]:
                cuda.memcpy_htod_async(device_inout[name], meta["buffer"], stream)

        # CALCOLO: La GPU calcola i 10 output simultaneamente
        ok = context.execute_async_v2(bindings_ptrs, stream.handle, None)
        if not ok:
            raise RuntimeError("Errore durante l'inferenza TensorRT!")

        # D2H: Unica copia dei 10 output verso la CPU
        for name, meta in host_inout.items():
            if not meta["is_input"]:
                cuda.memcpy_dtoh_async(meta["buffer"], device_inout[name], stream)

        # Sincronizzazione dell'intero flusso
        stream.synchronize()

        end_time = time.time()
        latenza = (end_time - start_time) * 1000

        # Rilascio esplicito immediato della memoria GPU prima dell'analisi
        for name, mem in device_inout.items():
            mem.free()
        device_inout.clear()

        # ==========================================
        # ANALISI RISULTATI (Matrice 10 x 20)
        # ==========================================
        print("\n3. Analisi dei risultati del batch...")
        output_key = "output"
        output_buffer = host_inout[output_key]["buffer"]

        # Modifichiamo la shape del buffer piatto per isolare le 10 righe di output
        output_logits = output_buffer.reshape((BATCH_SIZE, -1))

        # Calcoliamo l'ArgMax lungo l'asse orizzontale (axis=1) per ottenere le 10 azioni
        azioni_predette = np.argmax(output_logits, axis=1)

        print(f"Tempo di esecuzione batch parallelo: {latenza:.2f} ms")
        print(f"TEST COMPLETATO CON SUCCESSO!")
        print(f"Elenco delle 10 azioni predette: {azioni_predette}")
        print(f"Logits grezzi dell'input #1:\n{output_logits[0]}")

    except Exception as e:
        print(f"Errore durante l'esecuzione: {e}")

    finally:
        # Pulisce il contesto CUDA per evitare blocchi nei job successivi di NVBitFI
        try:
            ctx.pop()
        except Exception:
            pass

if __name__ == "__main__":
    main()
