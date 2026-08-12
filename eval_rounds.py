"""Danh gia NHIEU checkpoint tren tap test that, xuat confusion matrix SO LUONG THO.

Dung de soi doan chuyen giao task — vi du round 30 (het task 0) sang round 31,
32, 33 (dau task 1), noi model bat dau quen lop cu.

Khac `server_iov.py --mode test` o hai diem:
  1. Chay duoc NHIEU round trong mot lan, tap test chi nap MOT LAN (tap test
     that nang ~2.9 GB, nap lai 3 lan la phi 3 lan thoi gian nap).
  2. Confusion matrix ghi va ve bang SO LUONG THO, khong chuan hoa theo hang.
     `common.save_confusion_matrix` ve PNG tu ma tran da chuan hoa nen khong
     doc duoc so mau that trong tung o.

Ngoai ma tran cua he hop nhat DST, script con xuat rieng ma tran cua TUNG
NHANH (chi CNN1D, chi cay GBDT) — can de biet nhanh nao dang keo nhanh nao.

Chay:
  python eval_rounds.py --data-dir <DATA> --rounds 31 32 33 --task 1
  python eval_rounds.py --data-dir <DATA> --rounds 30 --task 0 --no-dst
"""
import argparse
import csv
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import common as C                                              # noqa: E402
from model_cnn1d import CNN1D_IDS, INPUT_LEN, NUM_GLOBAL_CLASSES  # noqa: E402

logger = logging.getLogger(__name__)


def save_counts(cm, names, path):
    """Ghi CSV so luong THO + them cot tong hang de doi chieu voi support."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + names + ["TONG (support)"])
        for i, row in enumerate(cm):
            w.writerow([names[i]] + [int(v) for v in row] + [int(row.sum())])
        w.writerow(["TONG (du doan)"] + [int(v) for v in cm.sum(0)]
                   + [int(cm.sum())])


def bang_text(cm, names):
    """Ma tran duoi dang bang chu — de doc thang trong log, khoi phai tai file."""
    ng = [n[:11] for n in names]
    w = max(11, max(len(f"{int(v):,}") for v in cm.flatten()))
    head = " " * 14 + "".join(f"{n:>{w + 2}}" for n in ng) + f"{'TONG':>{w + 2}}"
    dong = [head, " " * 14 + "-" * (len(head) - 14)]
    for i, row in enumerate(cm):
        dong.append(f"{ng[i]:>12} |"
                    + "".join(f"{int(v):>{w + 2},}" for v in row)
                    + f"{int(row.sum()):>{w + 2},}")
    dong.append(f"{'TONG du doan':>12} |"
                + "".join(f"{int(v):>{w + 2},}" for v in cm.sum(0))
                + f"{int(cm.sum()):>{w + 2},}")
    return "\n".join(dong)


def plot_counts(cm, names, path, tieu_de):
    """PNG voi SO LUONG THO trong tung o.

    Mau to theo thang LOGARIT: so mau trai dai tu 0 den 41.7 trieu, neu to
    tuyen tinh thi moi o tru lop da so deu trang xoa, nhin nhu ma tran rong.
    Chu trong o van la so dem that.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    n = len(names)
    fig, ax = plt.subplots(figsize=(1.6 + 1.15 * n, 1.3 + 0.95 * n))
    hien = np.where(cm > 0, cm, np.nan)          # o = 0 de trang
    im = ax.imshow(hien, cmap="Blues",
                   norm=LogNorm(vmin=max(1, np.nanmin(hien)), vmax=cm.max()))
    ax.set_xticks(range(n), names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n), names, fontsize=9)
    ax.set_xlabel("Du doan (Predicted)")
    ax.set_ylabel("That (True)")
    ax.set_title(tieu_de, fontsize=11)

    nguong = cm.max() * 0.02 if cm.max() else 0
    for i in range(n):
        for j in range(n):
            v = int(cm[i, j])
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=8,
                    color="white" if v > nguong else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="so mau (thang log)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description="Danh gia nhieu checkpoint, confusion matrix so luong tho")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--rounds", type=int, nargs="+", required=True)
    p.add_argument("--task", type=int, default=1, choices=range(C.NUM_TASKS))
    p.add_argument("--ckpt-dir", default=os.path.join(HERE, "logs", "ckpt_eval"))
    p.add_argument("--physics", default=None,
                   help="Mac dinh <ckpt-dir>/physics_branch_task<TASK>.pkl")
    p.add_argument("--out-dir", default=os.path.join(HERE, "logs", "eval_out"))
    p.add_argument("--test-samples", type=int, default=0,
                   help="0 = dung HET tap test")
    p.add_argument("--no-dst", action="store_true",
                   help="Chi danh gia CNN1D, bo nhanh cay va hop nhat DST")
    p.add_argument("--n-packet-features", type=int, default=18)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--batch-rows", type=int, default=500_000)
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    C.setup_logging(os.path.join(a.out_dir, "eval_rounds.log"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_cls = C.learned_classes(a.task)
    ten = C.load_class_names(a.data_dir)[:NUM_GLOBAL_CLASSES]
    if len(ten) < NUM_GLOBAL_CLASSES:
        ten += [f"class_{i}" for i in range(len(ten), NUM_GLOBAL_CLASSES)]
    ten_hien = ten[:n_cls]

    logger.info(f"Thiet bi {dev} | task {a.task} -> lop 0..{n_cls - 1} | "
                f"round {a.rounds}")

    # --- nap tap test MOT LAN cho ca ba round -------------------------------
    loader, y_all = C.load_global_test(a.data_dir, a.test_samples, a.task)
    logger.info(f"Tap test: {len(y_all):,} mau")

    fuser = None
    if not a.no_dst:
        from physics_branch import DSTFuser, load_physics, evaluate_with_dst
        gpath = a.physics or os.path.join(a.ckpt_dir,
                                          f"physics_branch_task{a.task}.pkl")
        gbdt, _ = load_physics(gpath)
        fuser = DSTFuser(gbdt, a.n_packet_features, device=dev)
        logger.info(f"Nhanh physics: {gpath} ({len(gbdt.trees)} cay)")

    model = CNN1D_IDS(INPUT_LEN, NUM_GLOBAL_CLASSES, a.dropout).to(dev)
    tom_tat = []

    for r in a.rounds:
        ck = os.path.join(a.ckpt_dir, f"round_{r:03d}.pth")
        if not os.path.exists(ck):
            logger.error(f"Khong co {ck} — bo qua round {r}")
            continue
        rnd, extra = C.load_checkpoint(ck, model)
        model.to(dev)
        logger.info(f"===== round {rnd} (train_loss luc luu = "
                    f"{extra.get('train_loss')}) =====")

        if fuser is not None:
            m, y_true, y_f, y_p, y_w = evaluate_with_dst(
                model, loader, nn.CrossEntropyLoss(), dev, fuser, C,
                batch_rows=a.batch_rows)
            nhanh = {"DSTfused": y_f, "packetCNN1D": y_p, "physicsGBDT": y_w}
        else:
            m, y_true, y_f = C.evaluate(model, loader, nn.CrossEntropyLoss(), dev)
            nhanh = {"CNN1D": y_f}

        logger.info(C.format_metrics(rnd, m))
        for ten_nhanh, y_pred in nhanh.items():
            cm = np.zeros((n_cls, n_cls), dtype=np.int64)
            hop_le = (y_true < n_cls) & (y_pred < n_cls)
            np.add.at(cm, (y_true[hop_le].astype(int),
                           y_pred[hop_le].astype(int)), 1)
            ngoai = int((~hop_le).sum())
            if ngoai:
                logger.warning(f"  {ten_nhanh}: {ngoai:,} du doan roi ra ngoai "
                               f"lop 0..{n_cls - 1} (khong nam trong ma tran)")
            nen = f"round{rnd:03d}_{ten_nhanh}"
            save_counts(cm, ten_hien,
                        os.path.join(a.out_dir, f"cm_counts_{nen}.csv"))
            plot_counts(cm, ten_hien,
                        os.path.join(a.out_dir, f"cm_counts_{nen}.png"),
                        f"Round {rnd} — {ten_nhanh} — so mau tho "
                        f"(task {a.task}, {cm.sum():,} mau)")
            dung = int(np.trace(cm))
            logger.info(f"  {ten_nhanh}: dung {dung:,}/{cm.sum():,} "
                        f"= {dung / max(cm.sum(), 1):.6f}\n"
                        f"  (hang = nhan that, cot = du doan, SO MAU THO)\n"
                        + bang_text(cm, ten_hien))

        if len(nhanh) > 1:
            dd = {k: int((v == y_true).sum()) for k, v in nhanh.items()}
            tot = max(dd, key=dd.get)
            n_t = len(y_true)
            logger.info("  --- so sanh nhanh (so mau dung) ---")
            for k, v in sorted(dd.items(), key=lambda kv: -kv[1]):
                logger.info(f"    {k:<14} {v:>12,}  ({v / n_t:.6f})"
                            + ("   <- tot nhat" if k == tot else ""))
            if tot != "DSTfused":
                thiet = dd[tot] - dd["DSTfused"]
                logger.info(f"    ** Hop nhat DST KEM hon '{tot}' "
                            f"{thiet:,} mau ({thiet / n_t:.6f}) **")

        from sklearn.metrics import classification_report
        rp = classification_report(y_true, y_f, labels=list(range(n_cls)),
                                   target_names=ten_hien, digits=4,
                                   zero_division=0)
        with open(os.path.join(a.out_dir, f"report_round{rnd:03d}.txt"), "w",
                  encoding="utf-8") as f:
            f.write(rp)
        logger.info("\n" + rp)
        tom_tat.append({"round": rnd, **{k: round(float(v), 6)
                                         for k, v in m.items()}})

    if tom_tat:
        with open(os.path.join(a.out_dir, "summary.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(tom_tat[0].keys()))
            w.writeheader()
            w.writerows(tom_tat)
        with open(os.path.join(a.out_dir, "summary.json"), "w",
                  encoding="utf-8") as f:
            json.dump(tom_tat, f, indent=2)
        logger.info(f"Xong. Ket qua trong {a.out_dir}")


if __name__ == "__main__":
    main()
