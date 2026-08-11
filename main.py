"""VAN-FED-IDS — mot lenh chay het.

Goi lai run_sim.py (Flower simulation, chay duoc du 100 client) nhung dat ten
tham so theo kieu quen thuoc: --data_dir, --num_users, --com_round...

Chay:
  python main.py --data_dir /kaggle/input/iov-100client --num_users 100

Mac dinh da la cau hinh day du: 100 client, 5 task noi tiep, 30 round moi task
(tong 150 round), kien truc hai nhanh + hop nhat Dempster-Shafer, dung het du
lieu, ghi metric va checkpoint moi round.

Ket qua nam trong --out_dir (mac dinh ./out):
  metrics_task0..4.csv        12 cot, round danh so lien tuc 1..150
  checkpoints/round_NNN.pth   moi round mot file, resume duoc
  confusion_matrix_task*.csv/.png, classification_report_task*.txt
  dst_fusion_task*.json       accuracy tung nhanh va muc tang nho hop nhat
  physics_branch_task*.pkl    nhanh cay lien ket da train

Kaggle het gio thi CHAY LAI DUNG LENH CU — no tu chay tiep tu cho dung.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    p = argparse.ArgumentParser(description="VAN-FED-IDS: mot lenh chay het")
    p.add_argument("--data_dir", required=True,
                   help="Thu muc chua federated_data/ va global_test_data.pt")
    p.add_argument("--out_dir", default=os.path.join(HERE, "out"))
    p.add_argument("--num_users", type=int, default=100, help="So client")
    p.add_argument("--tasks", type=int, default=5, help="So task noi tiep")
    p.add_argument("--com_round", type=int, default=30, help="Round MOI task")
    p.add_argument("--local_ep", type=int, default=1, help="Epoch cuc bo moi round")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_samples", type=int, default=0, help="0 = dung het du lieu")
    p.add_argument("--test_samples", type=int, default=0,
                   help="0 = DUNG HET tap test moi round (mac dinh). Dat so duong "
                        "de lay mau THEO TI LE neu muon chay nhanh hon")
    p.add_argument("--no_dst", action="store_true",
                   help="Tat kien truc hai nhanh, chi chay CNN1D (dung lam doi chung)")
    p.add_argument("--n_packet_features", type=int, default=18)
    p.add_argument("--gbdt_rounds", type=int, default=20)
    p.add_argument("--gbdt_depth", type=int, default=6)
    p.add_argument("--gbdt_max_per_client", type=int, default=0,
                   help="0 = DUNG HET du lieu client de dung cay (mac dinh). Dat "
                        "so duong de lay mau bot neu muon nhanh hon")
    p.add_argument("--cm_every", type=int, default=5)
    p.add_argument("--no_full_test", action="store_true",
                   help="Bo qua buoc danh gia tren toan bo tap test cuoi moi task")
    p.add_argument("--flat", action="store_true",
                   help="Gop ca 5 task lam mot (khong class-incremental)")
    p.add_argument("--restart", action="store_true", help="Bo ket qua cu, chay lai tu dau")
    p.add_argument("--fed_subdir", default="federated_data",
                   choices=["federated_data", "federated_data_fewshot",
                            "federated_data_10shot"])
    p.add_argument("--actor_gpus", type=float, default=-1.0,
                   help="Ty le GPU moi client song song. -1 = tu tinh. 0 = ep CPU")
    p.add_argument("--actor_cpus", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    argv = [
        "run_sim.py",
        "--data-dir", a.data_dir,
        "--out-dir", a.out_dir,
        "--clients", str(a.num_users),
        "--rounds", str(a.com_round),
        "--tasks", "none" if a.flat else ",".join(str(t) for t in range(a.tasks)),
        "--local-epochs", str(a.local_ep),
        "--batch-size", str(a.batch_size),
        "--lr", str(a.lr),
        "--max-samples", str(a.max_samples),
        "--test-samples", str(a.test_samples),
        "--cm-every", str(a.cm_every),
        "--seed", str(a.seed),
        "--fed-subdir", a.fed_subdir,
        "--actor-gpus", str(a.actor_gpus),
        "--actor-cpus", str(a.actor_cpus),
    ]
    if not a.no_full_test:
        argv.append("--final-full-test")
    if not a.no_dst:
        argv += ["--dst",
                 "--n-packet-features", str(a.n_packet_features),
                 "--gbdt-rounds", str(a.gbdt_rounds),
                 "--gbdt-depth", str(a.gbdt_depth),
                 "--gbdt-max-per-client", str(a.gbdt_max_per_client)]
    if a.restart:
        argv.append("--restart")

    print("=" * 70)
    print("VAN-FED-IDS | Chen et al., Computers & Security 142 (2024) 103881")
    print(f"  du lieu   : {a.data_dir}")
    print(f"  ket qua   : {a.out_dir}")
    print(f"  cau hinh  : {a.num_users} client | {a.tasks} task x {a.com_round} round "
          f"= {a.tasks * a.com_round} round")
    print(f"  kien truc : {'CNN1D don nhanh' if a.no_dst else 'CNN1D + cay lien ket, hop nhat Dempster-Shafer'}")
    print("=" * 70, flush=True)

    sys.argv = argv
    import run_sim
    run_sim.main()


if __name__ == "__main__":
    main()
