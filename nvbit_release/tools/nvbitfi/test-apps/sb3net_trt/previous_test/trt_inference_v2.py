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
# ALLOCAZIONE MEMORIA
# ==========================================
def allocate_bindings(engine, context):
    """Versione compatibile con TensorRT 7.x / 8.2 (Jetson Nano)"""
    host_inout = {}
    device_inout = {}
    bindings_ptrs = []
    stream = cuda.Stream()

    # In TensorRT < 8.5 si usa num_bindings invece di num_io_tensors
    for i in range(engine.num_bindings):
        # Si accede tramite indice 'i' invece che per nome
        name = engine.get_binding_name(i)
        is_input = engine.binding_is_input(i)
        dtype = engine.get_binding_dtype(i)
        np_dtype = np_dtype_from_trt(dtype)

        # Ottieni la shape direttamente dall'engine
        shape = tuple(engine.get_binding_shape(i))
        vol = int(np.prod(shape))

        # Buffer CPU e GPU
        host_mem = cuda.pagelocked_empty(vol, dtype=np_dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        # Inserisci il puntatore nella lista dei bindings nell'ordine corretto
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
        bindings_ptrs, host_inout, device_inout, stream = allocate_bindings(engine, context)

        print("Preparazione dati fasulli (Dummy Data)...")
        for name, meta in host_inout.items():
            if meta["is_input"]:
                # Generiamo dati casuali float32 tra 0 e 1
                dummy_data = np.random.random(meta["shape"]).astype(np.float32)
                # Copiamo nel buffer bloccato
                meta["buffer"][:] = dummy_data.ravel()
                print(f"    -> Input '{name}' pronto. Shape: {meta['shape']}")

        # --- INFERENZA ---
        print("Esecuzione inferenza sulla GPU...")
        start_time = time.time()

        # 1. H2D: Da CPU a GPU
        for name, meta in host_inout.items():
            if meta["is_input"]:
                cuda.memcpy_htod_async(device_inout[name], meta["buffer"], stream)
        
        print("Calcolo inferenza check")
        
        # 2. CALCOLO (Asincrono)
        ok = context.execute_async_v2(bindings_ptrs, stream.handle, None)
        if not ok:
            raise RuntimeError("Errore durante l'inferenza TensorRT!")

        # 3. D2H: Da GPU a CPU (Asincrono)
        for name, meta in host_inout.items():
            if not meta["is_input"]:
                cuda.memcpy_dtoh_async(meta["buffer"], device_inout[name], stream)

        # 4. SINCRONIZZA (Blocca la CPU finche la GPU non ha finito tutto)
        stream.synchronize()

        # >>> SOLO ORA E' SICURO LIBERARE LA MEMORIA GPU <<<
        for name, mem in device_inout.items():
            mem.free()
        device_inout.clear()

        end_time = time.time()
        latenza = (end_time - start_time) * 1000

        # ==========================================
        # ANALISI RISULTATI
        # ==========================================
        output_key = "output" 
        output_logits = host_inout[output_key]["buffer"]

        # Applichiamo ArgMax per trovare l'azione
        azione = np.argmax(output_logits)

        print(f"TEST COMPLETATO CON SUCCESSO!")
        print(f"Azione predetta: {azione}")
        print(f"Logits grezzi: {output_logits}")

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
