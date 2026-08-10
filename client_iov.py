"""P1 - VAN-FED-IDS: Flower client (CNN1D, CICIoV).

Bai bao: Chen et al., "Fast and practical intrusion detection system based on
federated learning for VANET", Computers & Security 142 (2024) 103881.

Thay doi so voi bai bao: classifier dung CNN1D (thay Bi-LSTM + LightGBM + DST)
de dong bo voi AFSIC-IoV / FedLiTeCAN.

Client chi lam viec cua RSU: huan luyen cuc bo roi gui trong so len cloud.
Danh gia tap trung tai server (khong danh gia cuc bo).

Chay:
  python client_iov.py --client-id 0
  python client_iov.py --client-id 0 --task 0        # che do class-incremental
"""
import argparse
import logging

import flwr as fl
import numpy as np
import torch
import torch.optim as optim

import common as C
from model_cnn1d import CNN1D_IDS, FocalLoss, INPUT_LEN, NUM_GLOBAL_CLASSES

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = r"C:\FederatedLearning\AFSIC-IOV\data\100client"


class VanFedClient(fl.client.NumPyClient):
    def __init__(self, client_id, data_dir, device, max_samples, batch_size,
                 task, lr, dropout):
        self.cid = client_id
        self.device = device
        self.lr = lr

        x, y = C.load_client_data(data_dir, client_id, task, max_samples)
        self.loader = C.make_loader(x, y, batch_size, shuffle=True)
        self.n_samples = len(y)

        self.model = CNN1D_IDS(INPUT_LEN, NUM_GLOBAL_CLASSES, dropout).to(device)
        self.criterion = FocalLoss(
            alpha=C.make_focal_alpha(y).to(device), gamma=2.0)

    # ---- Flower API -------------------------------------------------------
    def get_parameters(self, config):
        return C.get_model_parameters(self.model)

    def set_parameters(self, parameters):
        self.model.load_state_dict(C.ndarrays_to_state_dict(self.model, parameters))

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        epochs = int(config.get("local_epochs", 1))
        rnd = int(config.get("server_round", 0))
        lr = float(config.get("lr", self.lr))

        self.model.train()
        opt = optim.Adam(self.model.parameters(), lr=lr)
        total_loss, n_batches = 0.0, 0
        for ep in range(epochs):
            for xb, yb in self.loader:
                xb, yb = xb.to(self.device).float(), yb.to(self.device)
                opt.zero_grad()
                loss = self.criterion(self.model(xb), yb)
                loss.backward()
                opt.step()
                total_loss += loss.item()
                n_batches += 1
        avg = total_loss / max(n_batches, 1)
        logger.info(f"[Client {self.cid}][Round {rnd}] {epochs} epoch, "
                    f"n={self.n_samples}, train_loss={avg:.4f}")
        return C.get_model_parameters(self.model), self.n_samples, {"train_loss": avg}

    def evaluate(self, parameters, config):
        # Danh gia tap trung o server -> client tra ve gia tri rong.
        return 0.0, self.n_samples, {}


def main():
    p = argparse.ArgumentParser(description="P1 VAN-FED-IDS Flower client")
    p.add_argument("--client-id", type=int, required=True)
    p.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--server", type=str, default="127.0.0.1:8081")
    p.add_argument("--max-samples", type=int, default=500_000,
                   help="0 = dung toan bo du lieu client")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--task", type=int, default=None, choices=range(C.NUM_TASKS),
                   help="Class-incremental: chi hoc du lieu cua task nay")
    args = p.parse_args()

    C.setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client = VanFedClient(args.client_id, args.data_dir, device, args.max_samples,
                          args.batch_size, args.task, args.lr, args.dropout)
    fl.client.start_client(server_address=args.server, client=client.to_client())


if __name__ == "__main__":
    main()
