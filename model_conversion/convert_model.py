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
        #img = img.float() / 255.0
        img_f = self.linear_extractor(self.cnn_extractor(img))
        vec_f = self.vec_extractor(vec)
        cat = torch.cat([img_f, vec_f], dim=1)
        pred = self.q_net(cat)
	#return torch.argmax(pred, axis=1).int()
        return pred

# 2. Caricamento Modello
MODEL_PATH = '/home/g.galasso/thesis/Model_Inference/sb3net_out_pred.p'
print(MODEL_PATH) 
with open(MODEL_PATH, 'rb') as f:
    arch = pickle.load(f)

model = SB3Net(arch.cnn_extractor, arch.linear_extractor, arch.vec_extractor, arch.q_net)

model.to(device).eval()
print("Modello caricato in VRAM. Pronto per l'inferenza.")

# --- DEBUG PESI SULLA JETSON ---
print("\n" + "="*30)

#print(arch)
#print(model)

print("ISPEZIONE PESI MODELLO JETSON")
print("="*30)

print(model)

onnx_path = "/home/g.galasso/thesis/model_conversion/sb3net.onnx"

dummy_img = torch.randn(1, 3, 144, 256).to(device)
dummy_vec = torch.randn(1, 12).to(device)

inputs = (dummy_img, dummy_vec)

'''
def make_inputs(input_shapes):
    leaves = list(iter_shape_leaves(input_shapes))
    # Cambia dtype in torch.float16
    return tuple(torch.randn(*sh, dtype=torch.float16, device='cuda') for sh in leaves)

inputs = make_inputs(input_shapes)
'''

torch.onnx.export(
	model,
	inputs,
	onnx_path,
	output_names = ["output"],
	verbose = True
)

onnx_model = onnx.load(onnx_path)
#onnx.checker.check_model(onnx_model)

#TRT Engine

plan_path = "/home/g.galasso/thesis/model_conversion/sb3net.plan"


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

	profile = builder.create_optimization_profile()
	input_tensor = network.get_input(0)

	config.add_optimization_profile(profile)
	config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

	engine_bytes = builder.build_serialized_network(network, config)
	if engine_bytes is None:
		raise RuntimeError("Build engine failed.")

	if plan_path:
		with open(plan_path, "wb") as f:
			f.write(engine_bytes)
	print(f"[OK] Engine saved in: {plan_path}")
