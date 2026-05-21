import socket
import pickle
import torch
import os
import io
import time
import onnx
import tensorrt as trt

# 1. Setup GPU e Classe Modello
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Avvio Jetson Inference Server su: {}".format(device))

class SB3Net(torch.nn.Module):
    def __init__(self, cnn_extractor, linear_extractor, vec_extractor, q_net):
        super(SB3Net, self).__init__()
        self.cnn_extractor = cnn_extractor
        self.linear_extractor = linear_extractor
        self.vec_extractor = vec_extractor
        self.q_net = q_net

    def forward(self, img, vec):
        img_f = self.linear_extractor(self.cnn_extractor(img))
        vec_f = self.vec_extractor(vec)
        cat = torch.cat([img_f, vec_f], dim=1)
        pred = self.q_net(cat)
        return pred

# 2. Caricamento Modello
MODEL_PATH = '/home/g.galasso/g.galasso/model_conversion/sb3net_out_pred.p'
print(MODEL_PATH)
with open(MODEL_PATH, 'rb') as f:
    arch = pickle.load(f)

model = SB3Net(arch.cnn_extractor, arch.linear_extractor, arch.vec_extractor, arch.q_net)
model.to(device).eval()
print("Modello caricato in VRAM. Pronto per l'inferenza.")

print("\n" + "="*30)
print("ISPEZIONE PESI MODELLO JETSON")
print("="*30)
print(model)

onnx_path = "/home/g.galasso/g.galasso/model_conversion/sb3net.onnx"

# Utilizziamo ancora batch=1 per esportare la topologia di base
dummy_img = torch.randn(4, 3, 144, 256).to(device)
dummy_vec = torch.randn(4, 12).to(device)
inputs = (dummy_img, dummy_vec)

# MODIFICA 1: Aggiungiamo i nomi agli input e diciamo a ONNX che l'asse 0 (il batch) è dinamico
torch.onnx.export(
        model,
        inputs,
        onnx_path,
        input_names=["img_input", "vec_input"], # Diamo nomi chiari ai due layer di ingresso
        output_names=["output"],
        dynamic_axes={
            "img_input": {0: "batch_size"}, # L'asse 0 di img_input può variare
            "vec_input": {0: "batch_size"}, # L'asse 0 di vec_input può variare
            "output": {0: "batch_size"}     # Anche l'output cambierà di conseguenza
        },
        verbose=False # Messo su False per mantenere i log più puliti
)

onnx_model = onnx.load(onnx_path)
print("[OK] Modello ONNX esportato con assi dinamici.")

# ==========================================
# TRT Engine
# ==========================================
plan_path = "/home/g.galasso/g.galasso/model_conversion/sb3net.plan"
logger = trt.Logger(trt.Logger.WARNING)
explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

with trt.Builder(logger) as builder, \
        builder.create_network(explicit_batch) as network, \
        trt.OnnxParser(network, logger) as parser, \
        builder.create_builder_config() as config:

        with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                        for i in range(parser.num_errors):
                                print(f"[TRT][Parser] {parser.get_error(i)}")
                        raise RuntimeError("Parsing ONNX failed.")

        # Creazione del Profilo di Ottimizzazione
        profile = builder.create_optimization_profile()
        
        # NOVITÀ: Estraiamo i nomi dinamicamente dalla rete appena parsata
        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            name = tensor.name
            shape = tensor.shape
            
            # Se la shape ha 4 dimensioni (es. Batch, Canali, Altezza, Larghezza) è l'immagine
            if len(shape) == 4:
                profile.set_shape(
                    name, 
                    min=(1, 3, 144, 256), 
                    opt=(4, 3, 144, 256), 
                    max=(4, 3, 144, 256)
                )
                print(f"[TRT] Profilo immagine assegnato correttamente al layer: '{name}'")
                
            # Se la shape ha 2 dimensioni (es. Batch, Features) è il vettore telemetrico
            elif len(shape) == 2:
                profile.set_shape(
                    name, 
                    min=(1, 12), 
                    opt=(4, 12), 
                    max=(4, 12)
                )
                print(f"[TRT] Profilo vettore assegnato correttamente al layer: '{name}'")

        config.add_optimization_profile(profile)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

        # Limite di memoria per la compilazione (Workspace)
        try:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1 GB
        except AttributeError:
            config.max_workspace_size = 1 << 30

        print("\nInizio compilazione TensorRT... (Potrebbe richiedere alcuni minuti sulla Jetson)")
        engine_bytes = builder.build_serialized_network(network, config)
        
        if engine_bytes is None:
                raise RuntimeError("Build engine failed. Controlla gli errori precedenti.")

        if plan_path:
                with open(plan_path, "wb") as f:
                        f.write(engine_bytes)
        print(f"[OK] Engine salvato in: {plan_path} con supporto per Batch = 10")
