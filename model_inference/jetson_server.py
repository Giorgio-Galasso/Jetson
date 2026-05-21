import socket
import pickle
import torch
import os
import io
import time

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
        return torch.argmax(pred, axis=1).int()

# 2. Caricamento Modello
MODEL_PATH = '/home/g.galasso/thesis/Model_Inference/sb3net.p'
print(MODEL_PATH) 
with open(MODEL_PATH, 'rb') as f:
    arch = pickle.load(f)

model = SB3Net(arch.cnn_extractor, arch.linear_extractor, arch.vec_extractor, arch.q_net)
'''
cnn_extractor = torch.nn.Sequential(
    torch.nn.Conv2d(3, 32, 8, stride=4),
    torch.nn.ReLU(),
    torch.nn.Conv2d(32, 64, 4, stride=2),
    torch.nn.ReLU(),
    torch.nn.Conv2d(64, 64, 3, stride=1),
    torch.nn.ReLU(),
    torch.nn.Flatten()
)

linear_extractor = torch.nn.Sequential(
    torch.nn.Linear(25088, 256),
    torch.nn.ReLU()
)

vec_extractor = torch.nn.Flatten()

q_net = torch.nn.Sequential(
    torch.nn.Linear(268, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 20)
)

#########
checkpoint = torch.load("sb3_model.pth")

cnn_extractor.load_state_dict(checkpoint["cnn"])
linear_extractor.load_state_dict(checkpoint["linear"])
vec_extractor.load_state_dict(checkpoint["vec"])
q_net.load_state_dict(checkpoint["q_net"])
#######

model = SB3Net(
    cnn_extractor,
    linear_extractor,
    vec_extractor,
    q_net
)
'''
model.to(device).eval()
print("Modello caricato in VRAM. Pronto per l'inferenza.")

# --- DEBUG PESI SULLA JETSON ---
print("\n" + "="*30)

#print(arch)
#print(model)

print("ISPEZIONE PESI MODELLO JETSON")
print("="*30)

print(model)

# Se cnn_extractor è già il blocco Sequential:
#pesi_last = model.q_net[-1].weight.data
#pesi_last = model.q_net[-1]

#print(f"Shape primo layer CNN: {pesi_last.shape}")
#print("Primi valori pesi CNN:")
#print(pesi_last[-1].shape) 

# Per la Q-Net (che è anch'essa un Sequential):
#pesi_qnet = model.q_net[0].weight.data
#print(f"\nShape primo layer Q-Net: {pesi_qnet.shape}")
#print(pesi_qnet[0][:10])

# 3. Setup Socket Server
HOST = '0.0.0.0'  # In ascolto su tutte le interfacce di rete della Jetson
PORT = 5005      # La porta da esporre

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.bind((HOST, PORT))
    s.listen(1)
    print("In attesa di connessione dal PC sulla porta {}...".format(PORT))

    while True:
        conn, addr = s.accept()
        with conn:

            print("Connesso al PC: {}".format(addr))
            while True:
                try:
                    # Ricevi prima i 4 byte che indicano la lunghezza del pacchetto in arrivo
                    raw_len = conn.recv(4)
                    if not raw_len:
                        break # Il PC ha chiuso la connessione
                    msglen = int.from_bytes(raw_len, 'big')
                    
                    # Ricevi i dati effettivi (immagine + vettore)
                    data = b''
                    while len(data) < msglen:
                        packet = conn.recv(min(msglen - len(data), 65536))
                        if not packet: break
                        data += packet
                    
                    if not data: break
                    
                    # Deserializza l'osservazione inviata dal PC
                    obs = pickle.loads(data)
                    
                    # Converte in tensori, aggiunge la dimensione Batch (unsqueeze) e sposta su GPU
                    img_t = torch.from_numpy(obs['img']).unsqueeze(0).to(device).int() / 255.0
                    vec_t = torch.from_numpy(obs['vec']).unsqueeze(0).to(device).float()
                    #print(img_t)
                    #print(vec_t)
                    # Inferenza
                    inizio = time.time()
                    with torch.no_grad():
                        azione = model(img_t, vec_t).item()
                        #print("Azione", azione)
                    fine = time.time()
                    #print("Step ms", (fine-inizio)*1000)

                    # Invia indietro l'azione come un intero compresso a 4 byte
                    conn.sendall(azione.to_bytes(4, 'big'))
                
                except Exception as e:
                    print("Errore durante il ciclo: {}".format(e))
                    break
            print("Connessione col PC terminata. In attesa di una nuova simulazione...")
