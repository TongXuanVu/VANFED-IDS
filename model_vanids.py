"""VAN-FED-IDS: hai nhanh IDS + hop nhat Dempster-Shafer, dung theo bai bao.

Chen et al., "Fast and practical intrusion detection system based on federated
learning for VANET", Computers & Security 142 (2024) 103881.

Thong so lay TRUC TIEP tu bai (muc 3.2.3 va 3.2.4), khong phai suy doan:

  Packet-based IDS (Bi-LSTM), Fig. 3:
    - "A two-layer bidirectional LSTM layer is utilized for the extraction of
       sequential features. The intermediate layer has an output dimensionality
       of 32."
    - "the sequential features are aggregated via a fully connected layer, with
       a reduction in dimensionality from 32 to 16, utilizing a ReLU activation"
    - "A dropout layer is incorporated to mitigate the risk of overfitting."
    - "a softmax activation function is applied to execute the classification
       and yield the final prediction p_b"

  Physics-based IDS: LightGBM tren dac trung vat ly (toc do, gia toc, goc, vi
  tri). Xem fed_gbdt.py.

  Hop nhat DST (Eq. 2 va Eq. 3):
    BPA:  m(S_k) = (1 - c) * p_k   voi k = 0..K
          m(Omega) = c              voi c = ty le duong tinh gia cua model
    Luat Dempster, xung dot  Phi = sum_{k1 != k2, k1,k2 != Omega} p_b[k1]*p_w[k2]

  Hop nhat 2 tang (Fig. 4):
    Tang 1 (IDS level)     : gop Bi-LSTM + LightGBM cho MOT ban tin
    Tang 2 (vehicle level) : gop nhieu ban tin cua CUNG MOT xe -> ket luan ve xe

CHO LECH SO VOI BAI, phai ghi ro khi bao cao:
  - Bai dung 18 dac trung packet tu ban tin BSM (GPS, toc do, do tin cay...).
    CICIoV la du lieu CAN-bus 31 dac trung, khong co ranh gioi packet/physics
    tu nhien. Tach theo chi so cot qua tham so `n_packet_features`, mac dinh 18
    dung nhu bai. Day la LUA CHON CAI DAT, khong phai theo bai.
  - Bai train tren du lieu Veins/F2MD tu sinh, khong cong bo.
"""
import numpy as np
import torch
import torch.nn as nn

NUM_GLOBAL_CLASSES = 13
INPUT_LEN = 31
N_PACKET_FEATURES = 18          # bai bao: "18 attributes" cho nhanh packet


class BiLSTM_IDS(nn.Module):
    """Packet-based IDS. Kien truc dung Fig. 3 cua bai bao.

    (B, n_packet) -> (B, n_packet, 1) -> BiLSTM 2 lop -> 32 -> FC 16 -> softmax
    """

    def __init__(self, n_features=N_PACKET_FEATURES,
                 num_classes=NUM_GLOBAL_CLASSES, hidden=16, dropout=0.15):
        super().__init__()
        self.n_features = n_features
        self.num_classes = num_classes
        # hidden=16 moi chieu, hai chieu -> 32 = "intermediate output
        # dimensionality of 32" trong bai
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=2,
                            batch_first=True, bidirectional=True,
                            dropout=dropout)
        self.fc = nn.Linear(2 * hidden, 16)      # 32 -> 16, dung nhu bai
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(16, num_classes)

    def embed(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)                  # (B, n_features, 1)
        seq, _ = self.lstm(x)
        return seq[:, -1, :]                     # (B, 32)

    def forward(self, x):
        """Tra ve LOGIT. Softmax nam trong loss (cross-entropy) khi train,
        va trong `probs()` khi suy luan — tranh softmax hai lan."""
        return self.out(self.drop(self.relu(self.fc(self.embed(x)))))

    @torch.no_grad()
    def probs(self, x):
        """p_b trong bai: phan phoi xac suat tren K+1 lop."""
        return torch.softmax(self.forward(x), dim=1)


# ----------------------------------------------------------------------------
# Tach dac trung: nhanh packet / nhanh physics
# ----------------------------------------------------------------------------
def split_features(x, n_packet=N_PACKET_FEATURES):
    """(N, 31) -> (packet (N, n_packet), physics (N, 31-n_packet)).

    Bai tach theo Y NGHIA dac trung (ban tin BSM vs dai luong vat ly). CICIoV
    khong kem ten cot nen o day tach theo CHI SO. Doi `n_packet` neu ban biet
    ranh gioi that cua bo du lieu.
    """
    return x[:, :n_packet], x[:, n_packet:]


# ----------------------------------------------------------------------------
# Dempster-Shafer
# ----------------------------------------------------------------------------
def bpa(p, c):
    """Eq. (2): tu xac suat model -> khoi tin (mass) tren {S_0..S_K, Omega}.

    p : (N, K+1) xac suat model
    c : ty le duong tinh gia cua model do, xac dinh SAU khi train xong
    Tra ve (N, K+2), cot cuoi la m(Omega) = c.
    """
    p = np.asarray(p, dtype=np.float64)
    m = np.empty((p.shape[0], p.shape[1] + 1), dtype=np.float64)
    m[:, :-1] = (1.0 - c) * p
    m[:, -1] = c
    return m


def dempster_combine(m1, m2):
    """Luat Dempster cho hai nguon (Eq. 3).

    Voi moi lop k:
        m(S_k) ∝ m1[k]*m2[k] + m1[k]*m2[Omega] + m1[Omega]*m2[k]
        m(Omega) ∝ m1[Omega]*m2[Omega]
    Chuan hoa bang (1 - Phi), Phi = xung dot = sum_{k1 != k2} m1[k1]*m2[k2].

    m1, m2 : (N, K+2), cot cuoi la Omega. Tra ve (N, K+2).
    """
    m1 = np.asarray(m1, dtype=np.float64)
    m2 = np.asarray(m2, dtype=np.float64)
    a, b = m1[:, :-1], m2[:, :-1]                 # phan tren cac lop
    o1, o2 = m1[:, -1:], m2[:, -1:]               # phan Omega

    # xung dot: moi cap lop KHAC nhau
    phi = a.sum(1, keepdims=True) * b.sum(1, keepdims=True) - (a * b).sum(1, keepdims=True)
    denom = np.clip(1.0 - phi, 1e-12, None)

    out = np.empty_like(m1)
    out[:, :-1] = (a * b + a * o2 + o1 * b) / denom
    out[:, -1:] = (o1 * o2) / denom
    return out


def fuse_ids_level(p_b, p_w, c_b, c_w):
    """Tang 1 (Fig. 4): gop packet-based IDS va physics-based IDS.

    p_b : (N, K+1) xac suat tu Bi-LSTM   | c_b : FPR cua Bi-LSTM
    p_w : (N, K+1) xac suat tu LightGBM  | c_w : FPR cua LightGBM
    Tra ve (mass (N, K+2), nhan du doan D = argmax S_k).
    """
    m = dempster_combine(bpa(p_b, c_b), bpa(p_w, c_w))
    return m, m[:, :-1].argmax(1)


def fuse_vehicle_level(mass, vehicle_ids):
    """Tang 2 (Fig. 4): gop nhieu ban tin cua CUNG MOT xe thanh mot ket luan.

    mass        : (N, K+2) khoi tin sau tang 1
    vehicle_ids : (N,) xe gui ban tin do
    Tra ve (danh sach xe, mass moi xe, nhan moi xe).
    """
    mass = np.asarray(mass, dtype=np.float64)
    vehicle_ids = np.asarray(vehicle_ids)
    xe = np.unique(vehicle_ids)
    gop = np.empty((len(xe), mass.shape[1]), dtype=np.float64)
    for i, v in enumerate(xe):
        rows = mass[vehicle_ids == v]
        acc = rows[0:1]
        for j in range(1, len(rows)):             # gop lan luot tung ban tin
            acc = dempster_combine(acc, rows[j:j + 1])
        gop[i] = acc[0]
    return xe, gop, gop[:, :-1].argmax(1)


def false_positive_rate(y_true, y_pred, normal_class=0):
    """c trong Eq. (2): ty le mau BINH THUONG bi bao nham la tan cong.

    Bai noi "the false positive rate of machine learning models. After the
    training is completed, the value of c is confirmed." — nen do tren tap
    validation SAU khi train xong, roi truyen vao bpa().
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    binh_thuong = y_true == normal_class
    if binh_thuong.sum() == 0:
        return 0.0
    return float((y_pred[binh_thuong] != normal_class).mean())


if __name__ == "__main__":
    m = BiLSTM_IDS()
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    x = torch.randn(4, N_PACKET_FEATURES)
    print(f"Bi-LSTM params: {n:,} | out {tuple(m(x).shape)} | "
          f"embed {tuple(m.embed(x).shape)} (phai la 32)")

    # kiem tra DST: hai nguon dong thuan -> tin tang; mau thuan -> Omega tang
    K = 3
    dong_thuan = fuse_ids_level(np.array([[0.1, 0.8, 0.05, 0.05]]),
                                np.array([[0.1, 0.7, 0.1, 0.1]]), 0.05, 0.10)[0]
    mau_thuan = fuse_ids_level(np.array([[0.8, 0.1, 0.05, 0.05]]),
                               np.array([[0.1, 0.8, 0.05, 0.05]]), 0.05, 0.10)[0]
    print(f"dong thuan -> m(S1)={dong_thuan[0, 1]:.4f} m(Omega)={dong_thuan[0, -1]:.4f}")
    print(f"mau thuan  -> m(S0)={mau_thuan[0, 0]:.4f} m(S1)={mau_thuan[0, 1]:.4f} "
          f"m(Omega)={mau_thuan[0, -1]:.4f}")
    print(f"tong mass = {dong_thuan.sum():.6f} (phai la 1)")
