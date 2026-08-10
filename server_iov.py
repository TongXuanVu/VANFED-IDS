"""P1 - VAN-FED-IDS: Flower server (cloud aggregator).

Bai bao: Chen et al., "Fast and practical intrusion detection system based on
federated learning for VANET", Computers & Security 142 (2024) 103881.

Y tuong giu lai tu bai bao:
  - Kien truc 2 tang: RSU = client, cloud = server.
  - "Fast": moi round chi chon mot PHAN client (fraction_fit < 1.0) de giam
    do tre giao tiep -> day la bien so chinh cua bai.
  - Tong hop bang FedAvg co trong so theo so mau.
  - Danh gia TAP TRUNG tai cloud tren global_test_data.pt.

Thay doi: classifier la CNN1D (thay Bi-LSTM + LightGBM + Dempster-Shafer)
de 4 baseline dung chung mot backbone -> so sanh cong bang.

Chay:
  python server_iov.py --rounds 30 --num-clients 10
  python server_iov.py --rounds 30 --num-clients 10 --task 0     # class-incremental
  python server_iov.py --mode resume --rounds 50                 # chay tiep
  python server_iov.py --mode test --ckpt out/checkpoints/latest.pth
"""
import argparse
import logging
import os
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from flwr.common import (FitRes, Parameters, Scalar, ndarrays_to_parameters,
                         parameters_to_ndarrays)
from flwr.server.client_proxy import ClientProxy

import common as C
from model_cnn1d import CNN1D_IDS, INPUT_LEN, NUM_GLOBAL_CLASSES

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = r"C:\FederatedLearning\AFSIC-IOV\data\100client"
DEFAULT_OUT_DIR = r"C:\FederatedLearning\Rebuild-IOV\P1-VANFED-IDS\out"


# ----------------------------------------------------------------------------
# Strategy: FedAvg + checkpoint moi round
# ----------------------------------------------------------------------------
class VanFedStrategy(fl.server.strategy.FedAvg):
    """FedAvg chuan + luu checkpoint sau moi round tong hop."""

    def __init__(self, model, ckpt_dir: str, start_round: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.ckpt_dir = ckpt_dir
        self.start_round = start_round

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        params, metrics = super().aggregate_fit(server_round, results, failures)

        if results:
            losses = [(r.num_examples, r.metrics.get("train_loss", 0.0))
                      for _, r in results]
            n_tot = sum(n for n, _ in losses) or 1
            metrics["train_loss"] = sum(n * l for n, l in losses) / n_tot
            metrics["num_clients"] = len(results)
            logger.info(f"[Round {server_round}] tong hop {len(results)} client "
                        f"({n_tot} mau), train_loss={metrics['train_loss']:.4f}")
        if failures:
            logger.warning(f"[Round {server_round}] {len(failures)} client loi")

        if params is not None:
            abs_round = self.start_round + server_round
            sd = C.ndarrays_to_state_dict(self.model, parameters_to_ndarrays(params))
            C.save_checkpoint(self.ckpt_dir, abs_round, sd,
                              extra={"train_loss": metrics.get("train_loss")})
        return params, metrics


# ----------------------------------------------------------------------------
# Danh gia tap trung
# ----------------------------------------------------------------------------
def make_evaluate_fn(model, loader, criterion, device, csv_file, out_dir,
                     class_names, total_rounds, start_round, task):
    """Tra ve evaluate_fn cho Flower. Round cuoi -> xuat confusion matrix."""

    def evaluate_fn(server_round: int, parameters, config):
        if server_round == 0:                      # danh gia model khoi tao
            return None
        abs_round = start_round + server_round
        model.load_state_dict(C.ndarrays_to_state_dict(model, parameters))
        model.to(device)
        m, y_true, y_pred = C.evaluate(model, loader, criterion, device)
        C.log_and_save_metrics(abs_round, m, csv_file)

        if server_round == total_rounds:
            tag = f"task{task}" if task is not None else "final"
            C.save_confusion_matrix(y_true, y_pred, out_dir, tag, class_names)
        return m["loss"], {k: v for k, v in m.items() if k != "loss"}

    return evaluate_fn


def fit_config_fn(local_epochs: int, lr: float):
    def fn(server_round: int) -> Dict[str, Scalar]:
        return {"server_round": server_round, "local_epochs": local_epochs, "lr": lr}
    return fn


# ----------------------------------------------------------------------------
# Che do test doc lap
# ----------------------------------------------------------------------------
def run_test(args, model, device):
    ckpt = args.ckpt or os.path.join(args.out_dir, "checkpoints", "latest.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Khong tim thay checkpoint: {ckpt}")
    rnd, _ = C.load_checkpoint(ckpt, model)
    model.to(device)
    logger.info(f"Nap checkpoint {ckpt} (round {rnd})")

    loader, _ = C.load_global_test(args.data_dir, args.test_samples, args.task)
    m, y_true, y_pred = C.evaluate(model, loader, nn.CrossEntropyLoss(), device)
    logger.info(C.format_metrics(rnd, m))
    C.append_csv_row(os.path.join(args.out_dir, "test_metrics.csv"),
                     [rnd] + [round(m[k], 6) for k in C.METRIC_KEYS])
    tag = f"test_task{args.task}" if args.task is not None else "test"
    C.save_confusion_matrix(y_true, y_pred, args.out_dir, tag,
                            C.load_class_names(args.data_dir))


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="P1 VAN-FED-IDS Flower server")
    p.add_argument("--mode", choices=["train", "resume", "test"], default="train")
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--num-clients", type=int, default=10,
                   help="So client toi thieu phai ket noi truoc khi bat dau")
    p.add_argument("--fraction-fit", type=float, default=1.0,
                   help="Ty le RSU tham gia moi round ('fast' trong bai bao)")
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--address", type=str, default="0.0.0.0:8081")
    p.add_argument("--test-samples", type=int, default=1_000_000,
                   help="0 = dung toan bo global test")
    p.add_argument("--task", type=int, default=None, choices=range(C.NUM_TASKS),
                   help="Class-incremental: chi danh gia cac lop da hoc")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Checkpoint de resume/test (mac dinh: <out>/checkpoints/latest.pth)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    C.setup_logging(os.path.join(args.out_dir, "server.log"))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Thiet bi: {device} | che do: {args.mode} | task: {args.task}")

    model = CNN1D_IDS(INPUT_LEN, NUM_GLOBAL_CLASSES, args.dropout).to(device)

    if args.mode == "test":
        run_test(args, model, device)
        return

    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    start_round = 0
    if args.mode == "resume":
        ckpt = args.ckpt or os.path.join(ckpt_dir, "latest.pth")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Khong co checkpoint de resume: {ckpt}")
        start_round, _ = C.load_checkpoint(ckpt, model)
        model.to(device)
        logger.info(f"Resume tu round {start_round} ({ckpt})")

    loader, y_test = C.load_global_test(args.data_dir, args.test_samples, args.task)
    criterion = nn.CrossEntropyLoss()
    class_names = C.load_class_names(args.data_dir)
    suffix = f"_task{args.task}" if args.task is not None else ""
    csv_file = os.path.join(args.out_dir, f"metrics{suffix}.csv")

    strategy = VanFedStrategy(
        model=model,
        ckpt_dir=ckpt_dir,
        start_round=start_round,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=0.0,                       # danh gia tap trung
        min_fit_clients=max(1, int(args.num_clients * args.fraction_fit)),
        min_evaluate_clients=0,
        min_available_clients=args.num_clients,
        initial_parameters=ndarrays_to_parameters(C.get_model_parameters(model)),
        on_fit_config_fn=fit_config_fn(args.local_epochs, args.lr),
        evaluate_fn=make_evaluate_fn(model, loader, criterion, device, csv_file,
                                     args.out_dir, class_names, args.rounds,
                                     start_round, args.task),
    )

    logger.info(f"Server lang nghe {args.address} | {args.rounds} round | "
                f"fraction_fit={args.fraction_fit} | CSV -> {csv_file}")
    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
    logger.info(f"Xong. Ket qua trong {args.out_dir}")


if __name__ == "__main__":
    main()
