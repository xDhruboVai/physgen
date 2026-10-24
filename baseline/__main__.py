import argparse

from .train import FEATURES, evaluate, train

p = argparse.ArgumentParser(prog="baseline", description="Frozen-backbone temporal head (doc section 10)")
sub = p.add_subparsers(dest="cmd", required=True)
t = sub.add_parser("train")
t.add_argument("--root", required=True)
t.add_argument("--out", required=True)
t.add_argument("--features", default=FEATURES)
t.add_argument("--pairwise", action="store_true", help="add the c0-vs-violating ranking loss")
t.add_argument("--epochs", type=int, default=60)
t.add_argument("--seed", type=int, default=0)
e = sub.add_parser("evaluate")
e.add_argument("--root", required=True)
e.add_argument("--run", required=True)
a = p.parse_args()
if a.cmd == "train":
    train(a.root, a.out, a.features, a.pairwise, a.epochs, seed=a.seed)
else:
    evaluate(a.root, a.run)
