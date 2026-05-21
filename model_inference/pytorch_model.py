import torch
import pickle
import os

# Definiamo solo la struttura necessaria per ricostruire l'oggetto dal pickle
class SB3Net(torch.nn.Module):
    def __init__(self, cnn_extractor, linear_extractor, vec_extractor, q_net):
        super(SB3Net, self).__init__()
        self.cnn_extractor = cnn_extractor
        self.linear_extractor = linear_extractor
        self.vec_extractor = vec_extractor
        self.q_net = q_net
    
    def forward(self, img, vec):
        # Flusso: Immagine -> CNN -> Linear
        img_features = self.linear_extractor(self.cnn_extractor(img))
        # Flusso: Vettore -> Extractor
        vec_features = self.vec_extractor(vec)
        # Concatenazione e calcolo azione
        cat = torch.cat([img_features, vec_features], dim=1)
        pred = self.q_net(cat)
        return torch.argmax(pred, axis=1).int()

# Configurazione rapida
MODEL_PATH = './sb3net.p'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists(MODEL_PATH):
    # 1. Carica l'architettura dal file pickle
    with open(MODEL_PATH, 'rb') as f:
        arch = pickle.load(f)
    
    # 2. Inizializza il modello SB3Net "pulito"
    model = SB3Net(arch.cnn_extractor, arch.linear_extractor, arch.vec_extractor, arch.q_net)
    model.to(device).eval()
    
    print(f"Modello caricato correttamente su: {device}")

    # 3. Test veloce (esempio con shape per 'NH')
    # Usa (1, 4, 36, 64) se stai usando 'blocks'
    test_img = torch.randn(4, 3, 144, 256).to(device)
    test_vec = torch.randn(4, 12).to(device)

    with torch.no_grad():
        azione = model(test_img, test_vec)
    
    print(f"Azione predetta: {azione}")
else:
    print("File non trovato. Controlla il percorso!")
