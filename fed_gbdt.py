"""Physics-based IDS: cay tang cuong gradient HOC LIEN KET (FedTree, Eq. 9).

Chen et al., Computers & Security 142 (2024) 103881, muc 3.3.1:
"the federated learning (FL) model is bifurcated into two segments: neural
network FL and tree model FL". Nhanh cay dung FedTree (Li et al., 2022) voi

    G_mk = sum_n sum_{i in W_nmk} g_i        (Eq. 9)
    H_mk = sum_n sum_{i in W_nmk} h_i

nghia la: moi client dung du lieu CUA MINH tinh histogram gradient/hessian
theo tung o (feature m, bin k); server chi CONG DON cac histogram do lai roi
tim diem chia. Du lieu tho khong roi khoi client — chi co (G, H) duoc gui di.

Day KHAC HAN voi "moi client train mot LightGBM roi ghep cay lai": cay o day
duoc dung CHUNG tu thong ke gop cua tat ca client, dung nghia lien ket.

Cai dat theo kieu XGBoost/LightGBM:
  - da lop bang softmax: moi vong boosting dung K+1 cay, moi cay mot lop
  - g_i = p_i - y_i,  h_i = p_i (1 - p_i)
  - gain khi chia:  0.5 * (GL^2/(HL+lam) + GR^2/(HR+lam) - G^2/(H+lam)) - gamma
  - la:  w = -G / (H + lam)
  - histogram theo bin (mac dinh 64 bin, chia theo phan vi tren du lieu gop)

Bin edge phai giong nhau o moi client thi histogram moi cong duoc. Bai khong
noi ro cach thong nhat bin; o day server gom phan vi tu cac client (chi gui
thong ke phan vi, khong gui mau) — ghi ro day la lua chon cai dat.
"""
import numpy as np


# ----------------------------------------------------------------------------
def build_bin_edges(client_x, n_bins=64, n_features=None, seed=42):
    """Thong nhat bien bin giua cac client tu phan vi cua tung client.

    Moi client gui len n_bins phan vi cua tung dac trung (khong gui mau tho),
    server lay trung binh -> bien bin dung chung.
    """
    if n_features is None:
        n_features = client_x[0].shape[1]
    qs = np.linspace(0, 100, n_bins + 1)
    per_client = []
    for x in client_x:
        if len(x) == 0:
            continue
        per_client.append(np.percentile(x, qs, axis=0))       # (n_bins+1, F)
    edges = np.mean(per_client, axis=0)                       # gop
    # dam bao tang nghiem ngat de np.digitize chay dung
    for f in range(n_features):
        e = edges[:, f]
        for i in range(1, len(e)):
            if e[i] <= e[i - 1]:
                e[i] = e[i - 1] + 1e-6
        edges[:, f] = e
    return edges


CHUNK = 2_000_000          # so dong xu ly moi lan, de chan bo nho tam


def to_bins(x, edges):
    """(N, F) gia tri thuc -> (N, F) chi so bin. int8 du cho <=127 bin."""
    n_bins = edges.shape[0] - 1
    out = np.empty(x.shape, dtype=np.int8 if n_bins <= 127 else np.int16)
    for f in range(x.shape[1]):
        out[:, f] = np.clip(np.digitize(x[:, f], edges[1:-1, f]), 0, n_bins - 1)
    return out


# ----------------------------------------------------------------------------
class ClientData:
    """Phan nam o client. Chi tra ve histogram (G, H) — khong bao gio tra mau."""

    def __init__(self, xb, y, n_classes, n_bins):
        self.xb = xb                       # (N, F) da binning
        self.y = y
        self.n_bins = n_bins
        self.n_classes = n_classes
        self.n, self.F = xb.shape
        self.score = np.zeros((self.n, n_classes), dtype=np.float32)
        # Offset de doi (dac trung f, bin b) -> f*n_bins + b, gop ca ma tran
        # (F, n_bins) vao MOT np.bincount.
        #
        # Truoc day toi tinh SAN ca mang chi so nay (N, F) kieu int64 — voi 97.7
        # trieu mau x 13 dac trung do la 9.5 GB, du mot minh no cung lam tran RAM
        # Kaggle. Gio chi giu vector offset (F,) va dung chi so theo TUNG KHOI.
        self.offset = (np.arange(self.F, dtype=np.int64) * n_bins)

    def grad_hess(self, k):
        """g, h cua lop k theo softmax (nhu XGBoost multi:softprob).

        Tinh theo khoi: ma tran softmax trung gian (N, K) voi 97.7 trieu mau la
        4.7 GB neu lam mot lan.
        """
        g = np.empty(self.n, dtype=np.float32)
        h = np.empty(self.n, dtype=np.float32)
        for s in range(0, self.n, CHUNK):
            e = min(s + CHUNK, self.n)
            sc = self.score[s:e]
            p = np.exp(sc - sc.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            pk = p[:, k]
            g[s:e] = pk - (self.y[s:e] == k)
            h[s:e] = np.maximum(pk * (1 - pk), 1e-6)
        return g, h

    def histogram(self, gh, mask):
        """Eq. 9 phia client: cong don g, h vao tung o (feature, bin).

        `gh` la (g, h) da tinh SAN cho ca cay. Truoc day ham nay goi grad_hess()
        o TUNG node -> mot cay ~127 node phai tinh softmax tren toan bo mau 127
        lan. Voi 4.4 trieu mau x 13 lop, do la vai gio thay vi vai phut.

        Tra ve (G (F, n_bins), H (F, n_bins), so mau) — day la TAT CA nhung gi
        roi khoi client.
        """
        size = self.F * self.n_bins
        G = np.zeros(size)
        H = np.zeros(size)
        g, h = gh
        n_sel = 0
        for s in range(0, self.n, CHUNK):                 # cong don theo khoi
            e = min(s + CHUNK, self.n)
            m = mask[s:e]
            c = int(m.sum())
            if c == 0:
                continue
            n_sel += c
            flat = (self.offset + self.xb[s:e][m]).ravel()
            G += np.bincount(flat, weights=np.repeat(g[s:e][m], self.F),
                             minlength=size)
            H += np.bincount(flat, weights=np.repeat(h[s:e][m], self.F),
                             minlength=size)
        return (G.reshape(self.F, self.n_bins),
                H.reshape(self.F, self.n_bins), n_sel)


# ----------------------------------------------------------------------------
class FederatedGBDT:
    """Server: dieu phoi dung cay tu histogram gop cua cac client."""

    def __init__(self, n_classes=13, n_bins=64, max_depth=6, n_rounds=20,
                 lr=0.3, lam=1.0, gamma=0.0, min_child_weight=1e-3):
        self.n_classes = n_classes
        self.n_bins = n_bins
        self.max_depth = max_depth
        self.n_rounds = n_rounds
        self.lr = lr
        self.lam = lam
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.edges = None
        self.trees = []                    # [(round, class, cay)]

    # ---- tim diem chia tu histogram DA GOP ---------------------------------
    def _best_split(self, G, H):
        Gt, Ht = G.sum(1)[0], H.sum(1)[0]      # tong cua node (moi feature nhu nhau)
        goc = Gt * Gt / (Ht + self.lam)
        tot, tot_f, tot_b = -np.inf, -1, -1
        GL_c = np.cumsum(G, axis=1)
        HL_c = np.cumsum(H, axis=1)
        for f in range(G.shape[0]):
            GL, HL = GL_c[f, :-1], HL_c[f, :-1]
            GR, HR = Gt - GL, Ht - HL
            hop_le = (HL > self.min_child_weight) & (HR > self.min_child_weight)
            if not hop_le.any():
                continue
            gain = 0.5 * (GL ** 2 / (HL + self.lam) + GR ** 2 / (HR + self.lam)
                          - goc) - self.gamma
            gain = np.where(hop_le, gain, -np.inf)
            b = int(np.argmax(gain))
            if gain[b] > tot:
                tot, tot_f, tot_b = float(gain[b]), f, b
        return tot, tot_f, tot_b

    def _grow(self, clients, masks, k, depth, gh=None):
        """Dung mot node. masks[i] = mau nao cua client i thuoc node nay.

        gh: (g, h) cua tung client, tinh MOT LAN cho ca cay roi truyen xuong.
        """
        if gh is None:
            gh = [c.grad_hess(k) for c in clients]
        # --- gop histogram tu tat ca client (Eq. 9) ---
        G = H = None
        tong_mau = 0
        for c, m, e in zip(clients, masks, gh):
            g, h, n = c.histogram(e, m)
            G = g if G is None else G + g
            H = h if H is None else H + h
            tong_mau += n
        if tong_mau == 0:
            return {"leaf": 0.0}

        Gt, Ht = G.sum(1)[0], H.sum(1)[0]
        if depth >= self.max_depth:
            return {"leaf": float(-Gt / (Ht + self.lam))}

        gain, f, b = self._best_split(G, H)
        if gain <= 0 or f < 0:
            return {"leaf": float(-Gt / (Ht + self.lam))}

        trai = [m & (c.xb[:, f] <= b) for c, m in zip(clients, masks)]
        phai = [m & (c.xb[:, f] > b) for c, m in zip(clients, masks)]
        if sum(t.sum() for t in trai) == 0 or sum(p.sum() for p in phai) == 0:
            return {"leaf": float(-Gt / (Ht + self.lam))}
        return {"f": f, "bin": b,
                "L": self._grow(clients, trai, k, depth + 1, gh),
                "R": self._grow(clients, phai, k, depth + 1, gh)}

    # ---- suy luan nhanh: nen cay thanh mang, duyet theo TANG tren GPU -------
    @staticmethod
    def _flatten(tree):
        """dict long nhau -> (feat, thr, left, right, val). La co feat = -1."""
        feat, thr, left, right, val = [], [], [], [], []

        def rec(node):
            i = len(feat)
            feat.append(-1); thr.append(0); left.append(-1); right.append(-1)
            val.append(0.0)
            if "leaf" in node:
                val[i] = float(node["leaf"])
            else:
                feat[i] = int(node["f"])
                thr[i] = int(node["bin"])
                left[i] = rec(node["L"])
                right[i] = rec(node["R"])
            return i

        rec(tree)
        return (np.array(feat, np.int64), np.array(thr, np.int64),
                np.array(left, np.int64), np.array(right, np.int64),
                np.array(val, np.float32))

    def _compile(self):
        """Nen tat ca cay mot lan, dung lai cho moi lan predict."""
        if getattr(self, "_flat", None) is None:
            self._flat = [(k, self._flatten(t)) for _, k, t in self.trees]
        return self._flat

    @staticmethod
    def _predict_tree(tree, xb):
        out = np.zeros(len(xb))
        pile = [(tree, np.ones(len(xb), dtype=bool))]
        while pile:
            node, m = pile.pop()
            if not m.any():
                continue
            if "leaf" in node:
                out[m] = node["leaf"]
                continue
            di_trai = m & (xb[:, node["f"]] <= node["bin"])
            pile.append((node["L"], di_trai))
            pile.append((node["R"], m & ~di_trai))
        return out

    # ---- API ---------------------------------------------------------------
    def fit(self, client_x, client_y, verbose=True, max_per_client=0, seed=42):
        """client_x/client_y: danh sach mang cua TUNG client (khong gop lai).

        max_per_client > 0: lay mau bot o client qua lon. Cay chi can THONG KE
        phan bo chu khong can tung mau, nen vai chuc nghin mau moi client la du.
        """
        if max_per_client > 0:
            rng = np.random.default_rng(seed)
            cx, cy = [], []
            for x, y in zip(client_x, client_y):
                if len(y) > max_per_client:
                    sel = rng.choice(len(y), max_per_client, replace=False)
                    x, y = x[sel], y[sel]
                cx.append(x)
                cy.append(y)
            client_x, client_y = cx, cy
        self.edges = build_bin_edges(client_x, self.n_bins)
        clients = [ClientData(to_bins(x, self.edges), y, self.n_classes, self.n_bins)
                   for x, y in zip(client_x, client_y)]
        for rnd in range(self.n_rounds):
            for k in range(self.n_classes):
                masks = [np.ones(c.n, dtype=bool) for c in clients]
                cay = self._grow(clients, masks, k, 0)
                self.trees.append((rnd, k, cay))
                for c in clients:            # cap nhat score cuc bo
                    c.score[:, k] += self.lr * self._predict_tree(cay, c.xb)
            if verbose:
                acc = np.mean([(c.score.argmax(1) == c.y).mean() for c in clients])
                print(f"  vong {rnd + 1}/{self.n_rounds}: acc cuc bo tb = {acc:.4f}")
        return self

    def predict_proba(self, x, device=None, batch=2_000_000):
        """Duyet cay theo TANG, vector hoa. Dung GPU neu co — nhanh hon numpy
        hang chuc lan, du de danh gia tren toan bo tap test hang chuc trieu mau.
        """
        try:
            import torch
        except ImportError:
            torch = None
        if torch is None or device is None or str(device) == "cpu":
            return self._predict_proba_numpy(x)

        flat = self._compile()
        n = len(x)
        out = np.empty((n, self.n_classes), dtype=np.float32)
        sau = self.max_depth + 2                       # so tang toi da phai duyet
        for i0 in range(0, n, batch):
            xb = to_bins(x[i0:i0 + batch], self.edges)
            xt = torch.as_tensor(xb, device=device, dtype=torch.int64)
            m = xt.shape[0]
            ar = torch.arange(m, device=device)
            score = torch.zeros((m, self.n_classes), device=device)
            for k, (feat, thr, left, right, val) in flat:
                F_ = torch.as_tensor(feat, device=device)
                T_ = torch.as_tensor(thr, device=device)
                L_ = torch.as_tensor(left, device=device)
                R_ = torch.as_tensor(right, device=device)
                V_ = torch.as_tensor(val, device=device)
                idx = torch.zeros(m, dtype=torch.int64, device=device)
                for _ in range(sau):
                    f = F_[idx]
                    la = f < 0
                    if bool(la.all()):
                        break
                    xv = xt[ar, f.clamp(min=0)]
                    di_trai = xv <= T_[idx]
                    tiep = torch.where(di_trai, L_[idx], R_[idx])
                    idx = torch.where(la, idx, tiep)
                score[:, k] += self.lr * V_[idx]
            p = torch.softmax(score, dim=1)
            out[i0:i0 + batch] = p.cpu().numpy()
            del xt, score
        return out

    def _predict_proba_numpy(self, x):
        xb = to_bins(x, self.edges)
        score = np.zeros((len(x), self.n_classes))
        for _, k, cay in self.trees:
            score[:, k] += self.lr * self._predict_tree(cay, xb)
        p = np.exp(score - score.max(1, keepdims=True))
        return p / p.sum(1, keepdims=True)

    def predict(self, x, device=None):
        return self.predict_proba(x, device).argmax(1)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    K, F = 4, 6
    tam = rng.normal(0, 3, (K, F))
    cx, cy = [], []
    for _ in range(5):                                   # 5 client
        y = rng.integers(0, K, 400)
        cx.append((tam[y] + rng.normal(0, 1, (400, F))).astype(np.float32))
        cy.append(y)
    m = FederatedGBDT(n_classes=K, n_bins=32, max_depth=4, n_rounds=5, lr=0.3)
    m.fit(cx, cy)
    yt = rng.integers(0, K, 500)
    xt = (tam[yt] + rng.normal(0, 1, (500, F))).astype(np.float32)
    print(f"acc tren tap test doc lap: {(m.predict(xt) == yt).mean():.4f}")
