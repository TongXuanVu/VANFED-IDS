"""Dong goi / khoi phuc trang thai chay do — de resume qua GitHub thay vi Kaggle dataset.

VAN DE: /kaggle/working bi xoa sach giua cac session. Cach chinh thong la
"Save Version" roi attach output cu lam input dataset — nhieu buoc tay, de
quen, va moi lan lai phai sua duong dan trong notebook.

CACH O DAY: nhet trang thai vao chinh repo (logs/resume/), day len GitHub.
Session sau chi can `git clone` la co du de chay tiep.

Chi luu nhung gi CAN de resume, khong luu het:
  checkpoints/latest.pth       trong so model
  metrics_task*.csv            dem so round da xong (rounds_done() doc file nay)
  physics_branch_task*.pkl     cay lien ket — co san thi khong phai dung lai
  classification_report/cm/dst  ket qua da co, khong muon mat
KHONG luu: checkpoint tung round (.pth moi round ~170KB x 150 round = 26MB,
phinh lich su git), va file .png (ve lai duoc tu CSV).

Chay:
  python resume_state.py --status              # dang o round nao
  python resume_state.py --save                # out/ -> logs/resume/  (tren may, sau khi tai ket qua ve)
  python resume_state.py --load                # logs/resume/ -> out/  (tren Kaggle, truoc khi train)
"""
import argparse
import glob
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESUME = os.path.join(HERE, "logs", "resume")

# (mau glob, co giu khong) — theo thu tu uu tien khi in
MAU = [
    "metrics*.csv",
    "physics_branch_task*.pkl",
    "classification_report_*.txt",
    "confusion_matrix_*.csv",
    "dst_fusion_*.json",
    "sim.log",
]


def dem_round(thu_muc):
    """Doc so round da xong tung task — dung dung logic cua run_sim.rounds_done()."""
    ket = {}
    for p in sorted(glob.glob(os.path.join(thu_muc, "metrics*.csv"))):
        ten = os.path.basename(p)
        if ten.startswith("test_"):
            continue
        try:
            with open(p, encoding="utf-8") as f:
                n = sum(1 for _ in f) - 1
        except OSError:
            n = 0
        ket[ten] = max(n, 0)
    return ket


def round_checkpoint(thu_muc):
    p = os.path.join(thu_muc, "checkpoints", "latest.pth")
    if not os.path.exists(p):
        return None
    try:
        import torch
        return torch.load(p, map_location="cpu", weights_only=False).get("round")
    except Exception as e:
        print(f"  (khong doc duoc latest.pth: {e})")
        return None


def in_trang_thai(ten, thu_muc):
    print(f"\n{ten}: {thu_muc}")
    if not os.path.isdir(thu_muc):
        print("  (chua co)")
        return
    r = round_checkpoint(thu_muc)
    print(f"  latest.pth       : round {r}" if r is not None else
          "  latest.pth       : KHONG CO")
    d = dem_round(thu_muc)
    if d:
        for k, v in d.items():
            print(f"  {k:<22}: {v} round")
    else:
        print("  (chua co metrics CSV)")
    for pkl in sorted(glob.glob(os.path.join(thu_muc, "physics_branch_task*.pkl"))):
        print(f"  cay co san        : {os.path.basename(pkl)}")


def chep(nguon, dich, ten_file):
    os.makedirs(os.path.dirname(dich), exist_ok=True)
    shutil.copy2(nguon, dich)
    return ten_file


def save(out_dir, force):
    if not os.path.isdir(out_dir):
        sys.exit(f"Khong thay {out_dir} — chua chay lan nao?")
    cu, moi = dem_round(RESUME), dem_round(out_dir)
    tong_cu, tong_moi = sum(cu.values()), sum(moi.values())
    if tong_moi < tong_cu and not force:
        sys.exit(f"DUNG LAI: out/ dang o {tong_moi} round, con logs/resume/ da co "
                 f"{tong_cu} round. Luu de se LUI tien do.\n"
                 f"Chac chan thi them --force.")

    os.makedirs(RESUME, exist_ok=True)
    da = []
    latest = os.path.join(out_dir, "checkpoints", "latest.pth")
    if os.path.exists(latest):
        da.append(chep(latest, os.path.join(RESUME, "checkpoints", "latest.pth"),
                       "checkpoints/latest.pth"))
    for mau in MAU:
        for p in sorted(glob.glob(os.path.join(out_dir, mau))):
            da.append(chep(p, os.path.join(RESUME, os.path.basename(p)),
                           os.path.basename(p)))
    print(f"Da dong goi {len(da)} file -> {RESUME}")
    for f in da:
        print(f"   {f}")
    print("\nBuoc tiep: commit + push, roi session sau chi can git clone.")


def load(out_dir, force):
    if not os.path.isdir(RESUME):
        sys.exit(f"Khong thay {RESUME} — repo chua co trang thai nao de khoi phuc.")
    cu, moi = dem_round(RESUME), dem_round(out_dir)
    tong_cu, tong_moi = sum(cu.values()), sum(moi.values())
    if tong_moi > tong_cu and not force:
        sys.exit(f"DUNG LAI: out/ dang o {tong_moi} round, moi hon logs/resume/ "
                 f"({tong_cu} round). Khoi phuc se GHI DE ket qua moi hon.\n"
                 f"Chac chan thi them --force.")

    os.makedirs(out_dir, exist_ok=True)
    da = []
    latest = os.path.join(RESUME, "checkpoints", "latest.pth")
    if os.path.exists(latest):
        da.append(chep(latest, os.path.join(out_dir, "checkpoints", "latest.pth"),
                       "checkpoints/latest.pth"))
    for p in sorted(glob.glob(os.path.join(RESUME, "*"))):
        if os.path.isfile(p):
            da.append(chep(p, os.path.join(out_dir, os.path.basename(p)),
                           os.path.basename(p)))
    print(f"Da khoi phuc {len(da)} file -> {out_dir}")
    in_trang_thai("Sau khi khoi phuc", out_dir)
    print("\nChay tiep binh thuong. TUYET DOI KHONG them --restart.")


def main():
    p = argparse.ArgumentParser(description="Dong goi/khoi phuc trang thai resume")
    p.add_argument("--out-dir", default=os.path.join(HERE, "out"))
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--save", action="store_true", help="out/ -> logs/resume/")
    g.add_argument("--load", action="store_true", help="logs/resume/ -> out/")
    g.add_argument("--status", action="store_true", help="Chi xem, khong dong gi")
    p.add_argument("--force", action="store_true",
                   help="Bo qua canh bao lui tien do / ghi de")
    a = p.parse_args()

    if a.status:
        in_trang_thai("Trong repo (logs/resume)", RESUME)
        in_trang_thai("Dang chay (out)", a.out_dir)
    elif a.save:
        save(a.out_dir, a.force)
    else:
        load(a.out_dir, a.force)


if __name__ == "__main__":
    main()
