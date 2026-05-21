import socket
import pickle
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import os

# --- CONFIGURAZIONE ---
IP_JETSON = '0.0.0.0'  # Ascolta su tutte le interfacce
PORTA = 5005
PLAN_PATH = "/home/g.galasso/thesis/model_conversion/sb3net.plan"
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# --- UTILS TENSORRT (Versione Legacy) ---
def load_engine(path):
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine, context):
    host_inout = {}
    device_inout = {}
    bindings = []
    stream = cuda.Stream()
    
    print("num_bindings: ", engine.num_bindings)
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        print("name: ", name)
        
        dtype = engine.get_binding_dtype(i)
        print("dtype: ", dtype)
        
        shape = engine.get_binding_shape(i)
        print("shape: ", shape)
        
        is_input = engine.binding_is_input(i)
        print("is_input: ", is_input)
        
        size = trt.volume(shape)
        print("size: ", size)
        
        np_dtype = np.float32 #if dtype == trt.DataType.FLOAT else np.float16
        print("np_dtype: ", np_dtype)
        
        host_mem = cuda.pagelocked_empty(size, np_dtype)
        print("host_mem: ", host_mem)
        
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        print("device_mem: ", device_mem)
        
        bindings.append(int(device_mem))
        host_inout[name] = {"is_input": is_input, "buffer": host_mem}
        device_inout[name] = device_mem
        
    return bindings, host_inout, device_inout, stream

# --- PREPARAZIONE ---
print("🧠 Caricamento Modello TensorRT...")
engine = load_engine(PLAN_PATH)
context = engine.create_execution_context()
bindings, host_inout, device_inout, stream = allocate_buffers(engine, context)

print("🔍 Nomi dei binding rilevati nell'engine:")
for name in host_inout.keys():
    print(f"   -> '{name}' (Input: {host_inout[name]['is_input']})")

# --- SERVER SOCKET ---
server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_sock.bind((IP_JETSON, PORTA))
server_sock.listen(1)

print(f"📡 Server in ascolto sulla porta {PORTA}...")

try:
    while True:
        conn, addr = server_sock.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"🔗 Connesso con: {addr}")

        try:
            while True:
                # 1. Leggi l'header di 4 byte (lunghezza del pacchetto)
                raw_size = conn.recv(4)
                if not raw_size: 
                    break
                size = int.from_bytes(raw_size, 'big')

                # 2. Leggi il payload (il dizionario pickle)
                data = b""
                while len(data) < size:
                    packet = conn.recv(size - len(data))
                    if not packet: 
                        break
                    data += packet

                # 3. Deserializzazione (Protocollo 2 come richiesto)
                obs = pickle.loads(data)
                img = obs['img'].astype(np.float32) / 255.0  # Normalizzazione
                vec = obs['vec'].astype(np.float32)

                # 4. Inferenza TensorRT con i nomi REALI dei binding
                host_inout['input.1']['buffer'][:] = img.ravel()
                host_inout['27']['buffer'][:] = vec.ravel()

                # Trasferimento dati da CPU a GPU
                for name, meta in host_inout.items():
                    if meta['is_input']:
                        #print("Input layer: ", name)
                        cuda.memcpy_htod_async(device_inout[name], meta['buffer'], stream)

                # Esecuzione sulla GPU
                context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

                # Recupero risultati da GPU a CPU
                for name, meta in host_inout.items():
                    if not meta['is_input']:
                        #print("Output layer: ", name)
                        cuda.memcpy_dtoh_async(meta['buffer'], device_inout[name], stream)

                stream.synchronize()

                # 5. Estrazione Azione
                output_data = host_inout['output']['buffer']
                action = int(np.argmax(output_data))
                print("Action: ", action)

                # 6. Risposta al PC (4 byte, big endian)
                conn.sendall(action.to_bytes(4, 'big'))

        except Exception as e:
            print(f"⚠️ Connessione interrotta: {e}")
        finally:
            conn.close()
            print("🔌 Socket client chiuso. In attesa di nuova connessione...")

except KeyboardInterrupt:
    print("\n🛑 Server spento manualmente.")
finally:
    server_sock.close()
