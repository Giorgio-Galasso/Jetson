import torch
import numpy as np
import pickle
import os
import sys

# --- DETERMINISMO ---
# Fondamentale per NVBitFI: il risultato deve essere sempre lo stesso
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DEFINIZIONE MODELLO ---
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
        return torch.argmax(pred, axis=1).int()

# --- CARICAMENTO ---

# 1. Ricava il percorso assoluto della cartella dello script
# __file__ contiene il path completo di sb3net_pytorch.py
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# 2. Costruisci il percorso del modello unendo la cartella al nome del file
MODEL_PATH = os.path.join(BASE_PATH, 'sb3net.p')

# Ora os.path.exists(MODEL_PATH) sarà SEMPRE True

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            arch = pickle.load(f)
        model = SB3Net(arch.cnn_extractor, arch.linear_extractor, arch.vec_extractor, arch.q_net)
    
    model.to(device).eval()
    print("Model:", model)
    return model

# --- ESECUZIONE 10 INFERENZE ---
def main():
    model = load_model()
    
    # Generazione di 10 input diversi ma deterministici
    # Immagini 3x224x224 e vettori di stato da 18 elementi
    test_imgs = [torch.randn(1, 3, 144, 256).to(device) for _ in range(10)]
    test_vecs = [torch.randn(1, 12).to(device) for _ in range(10)]
    
    print("Inference Start")
    results = []
    
    with torch.no_grad():
        for i in range(10):
            # Normalizzazione come nel tuo script originale
            img = test_imgs[i] / 255.0
            vec = test_vecs[i]
            
            azione = model(img, vec).item()
            results.append(azione)
            
    # Questo output sarà catturato nel golden_stdout.txt
    print("FINAL_ACTIONS: {}".format(results))

if __name__ == "__main__":
    main()