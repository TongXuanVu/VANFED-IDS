"""Nhanh physics-based IDS cua VAN-FED-IDS: train, luu, nap, va suy luan.

Chen et al., C&S 142 (2024) 103881. Kien truc hai nhanh:

    packet-based IDS   -> p_b      (CNN1D, xem model_cnn1d.py)
    physics-based IDS  -> p_w      (cay tang cuong gradient lien ket, fed_gbdt.py)
                DST fusion  ->  du doan cuoi cung

Nhanh cay KHONG nam trong vong FedAvg. Bai bao noi ro hai nhanh duoc huan luyen
tach nhau ("the FL model is bifurcated into two segments: neural network FL and
tree model FL"), va cay it phai cap nhat hon mang neural. O day cay duoc dung
MOT LAN cho moi task, truoc khi vao vong FL cua CNN1D — nho vay ba che do
train/resume/test cua server khong phai sua gi.

Tinh lien ket cua nhanh cay nam o cho: client chi gui histogram (G, H) theo
tung o (dac trung, bin), server cong don theo Eq. 9 roi tim diem chia. Du lieu
tho khong roi client. Xem fed_gbdt.py.

CHO LECH SO VOI BAI:
  - Bai tach 18 dac trung BSM cho nhanh packet, phan con lai cho nhanh physics.
    CICIoV khong kem ten cot. Mac dinh o day: CNN1D van doc DU 31 dac trung
    (giu backbone chung voi P2/P3/P4), nhanh cay doc 13 cot cuoi. Doi bang
    --n-packet-features.
  - `c` (ty le duong tinh gia) trong Eq. 2 do tren chinh tap danh gia sau khi
    train xong, dung nhu bai mo ta.
"""
import json
import logging
import os
import pickle

import numpy as np

from fed_gbdt import FederatedGBDT
from model_vanids import bpa, dempster_combine, false_positive_rate

logger = logging.getLogger(__name__)


def physics_slice(x, n_packet):
    """Cot danh cho nhanh physics. n_packet=0 -> dung het."""
    return x if n_packet <= 0 else x[:, n_packet:]


def train_physics_branch(data_dir, client_ids, task, n_packet, load_client_data,
                         n_classes=13, n_bins=64, max_depth=6, n_rounds=20,
                         lr=0.3, max_samples=200_000,
                         gbdt_max_per_client=20_000):
    """Dung cay lien ket tu du lieu cac client (chi gop histogram)."""
    cx, cy = [], []
    for cid in client_ids:
        try:
            x, y = load_client_data(data_dir, cid, task, max_samples)
        except FileNotFoundError:
            continue
        if len(y) == 0:
            continue
        cx.append(physics_slice(x, n_packet))
        cy.append(y)
    if not cx:
        raise RuntimeError("Khong client nao co du lieu cho nhanh physics")
    logger.info(f"Nhanh physics: {len(cx)} client, {sum(len(y) for y in cy)} mau "
                f"(lay toi da {gbdt_max_per_client}/client), {cx[0].shape[1]} dac trung, "
                f"{n_rounds} vong boosting")
    gbdt = FederatedGBDT(n_classes=n_classes, n_bins=n_bins, max_depth=max_depth,
                         n_rounds=n_rounds, lr=lr)
    gbdt.fit(cx, cy, verbose=False, max_per_client=gbdt_max_per_client)
    return gbdt


def save_physics(gbdt, path, meta=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"gbdt": gbdt, "meta": meta or {}}, f)
    logger.info(f"Luu nhanh physics -> {path}")


def load_physics(path):
    with open(path, "rb") as f:
        blob = pickle.load(f)
    return blob["gbdt"], blob.get("meta", {})


# ----------------------------------------------------------------------------
class DSTFuser:
    """Giu nhanh cay + hai he so c, thuc hien hop nhat DST luc danh gia."""

    def __init__(self, gbdt, n_packet, c_packet=0.05, c_physics=0.05, device=None):
        self.gbdt = gbdt
        self.n_packet = n_packet
        self.device = device
        self.c_b = c_packet
        self.c_w = c_physics

    def physics_probs(self, x):
        return self.gbdt.predict_proba(physics_slice(x, self.n_packet),
                                       device=self.device)

    def fuse(self, p_b, x_raw):
        """p_b: (N, K+1) tu CNN1D. Tra ve (nhan sau hop nhat, mass)."""
        p_w = self.physics_probs(x_raw)
        m = dempster_combine(bpa(p_b, self.c_b), bpa(p_w, self.c_w))
        return m[:, :-1].argmax(1), m

    def calibrate(self, p_b, x_raw, y_true, normal_class=0):
        """Do `c` cho tung nhanh (Eq. 2: "After training is completed, c is
        confirmed"), roi cap nhat vao chinh doi tuong nay."""
        p_w = self.physics_probs(x_raw)
        self.c_b = false_positive_rate(y_true, p_b.argmax(1), normal_class)
        self.c_w = false_positive_rate(y_true, p_w.argmax(1), normal_class)
        logger.info(f"DST: c(packet)={self.c_b:.4f}  c(physics)={self.c_w:.4f}")
        return self.c_b, self.c_w


def evaluate_with_dst(model, loader, criterion, device, fuser, common):
    """Danh gia HAI NHANH + hop nhat DST.

    Tra ve dung dinh dang cua common.evaluate() de phan ghi CSV / confusion
    matrix / checkpoint o server khong phai sua gi:
        (metrics, y_true, y_pred_fused, y_pred_packet, y_pred_physics)
    """
    import torch
    model.eval()
    loss_sum, nb = 0.0, 0
    P_b, X, Y = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb_d = xb.to(device).float()
            out = model(xb_d)
            loss_sum += criterion(out, yb.to(device)).item()
            nb += 1
            P_b.append(torch.softmax(out, 1).cpu().numpy())
            X.append(xb.numpy().astype(np.float32))
            Y.append(yb.numpy())
    p_b = np.concatenate(P_b)
    x_raw = np.concatenate(X)
    y_true = np.concatenate(Y)

    y_fused, _ = fuser.fuse(p_b, x_raw)
    y_packet = p_b.argmax(1)
    y_physics = fuser.physics_probs(x_raw).argmax(1)

    m = common.compute_metrics(y_true, y_fused, loss_sum / max(nb, 1))
    return m, y_true, y_fused, y_packet, y_physics


def fusion_report(y_true, y_packet, y_physics, y_fused, path=None):
    """So sanh tung nhanh voi ket qua hop nhat — bang can co trong bao cao."""
    r = {
        "acc_packet_only": float((y_true == y_packet).mean()),
        "acc_physics_only": float((y_true == y_physics).mean()),
        "acc_dst_fused": float((y_true == y_fused).mean()),
    }
    r["gain_vs_packet"] = r["acc_dst_fused"] - r["acc_packet_only"]
    r["gain_vs_physics"] = r["acc_dst_fused"] - r["acc_physics_only"]
    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2)
    return r
