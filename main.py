import argparse
import subprocess
import shlex

# from app.n_7mer import train_7mer, test_7mer
# from app.n_4mer import train_4mer, test_4mer
# from app.n_123mer import train_123mer, test_123mer
# from app.n_dotplot import dotplot_train, dotplot_test
# from app.n_hyenadna import hyena_train, hyena_test
# from app.n_ensemble import ensemble_train, ensemble_test

def run_in_env(env_name, code):
    cmd = [
        "conda", "run", "--no-capture-output",
        "-n", env_name,
        "python", "-c", code
    ]

    print("RUN:", " ".join(shlex.quote(x) for x in cmd))

    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed in env {env_name}")


def train(fasta_file, metadata_file, checkpoint_dir):
    print("TRAIN")
    print("fasta_file:", fasta_file)
    print("metadata_file:", metadata_file)
    print("checkpoint_dir:", checkpoint_dir)

    # env: train_cnn
    run_in_env(
        "train_cnn",
        f"""
from app.n_7mer import train_7mer
train_7mer({fasta_file!r}, {metadata_file!r}, {checkpoint_dir!r})
"""
    )

    run_in_env(
        "train_cnn",
        f"""
from app.n_123mer import train_123mer
train_123mer({fasta_file!r}, {metadata_file!r}, {checkpoint_dir!r})
"""
    )

    run_in_env(
        "train_cnn",
        f"""
from app.n_4mer import train_4mer
train_4mer({fasta_file!r}, {metadata_file!r}, {checkpoint_dir!r}, length=20)
"""
    )

    run_in_env(
        "train_cnn",
        f"""
from app.n_4mer import train_4mer
train_4mer({fasta_file!r}, {metadata_file!r}, {checkpoint_dir!r}, length=30)
"""
    )

    # env: dgl_env
    run_in_env(
        "dgl_env",
        f"""
from app.n_dotplot import dotplot_train
dotplot_train({fasta_file!r}, {metadata_file!r}, {checkpoint_dir!r})
"""
    )

    # env: mobiraph
    run_in_env(
        "mobiraph",
        f"""
from app.n_hyenadna import hyena_train
hyena_train({fasta_file!r}, {metadata_file!r}, {checkpoint_dir!r})
"""
    )

    run_in_env(
        "mobiraph",
        f"""
from app.n_ensemble import ensemble_train
ensemble_train({fasta_file!r}, {metadata_file!r}, {checkpoint_dir!r})
"""
    )

    print("TRAIN: DONE")


def test(fasta_file, models_path, output_file, checkpoint_dir):
    print("TEST")
    print("fasta_file:", fasta_file)
    print("models_path:", models_path)
    print("output_file:", output_file)
    print("checkpoint_dir:", checkpoint_dir)

    # env: train_cnn
    run_in_env(
        "train_cnn",
        f"""
from app.n_7mer import test_7mer
test_7mer({fasta_file!r}, {checkpoint_dir!r}, {models_path!r})
"""
    )

    run_in_env(
        "train_cnn",
        f"""
from app.n_123mer import test_123mer
test_123mer({fasta_file!r}, {checkpoint_dir!r}, {models_path!r})
"""
    )

    run_in_env(
        "train_cnn",
        f"""
from app.n_4mer import test_4mer
test_4mer({fasta_file!r}, {checkpoint_dir!r}, {models_path!r}, length=20)
"""
    )

    run_in_env(
        "train_cnn",
        f"""
from app.n_4mer import test_4mer
test_4mer({fasta_file!r}, {checkpoint_dir!r}, {models_path!r}, length=30)
"""
    )

    # env: dgl_env
    run_in_env(
        "dgl_env",
        f"""
from app.n_dotplot import dotplot_test
dotplot_test({fasta_file!r}, {checkpoint_dir!r}, {models_path!r})
"""
    )

    # env: mobiraph
    run_in_env(
        "mobiraph",
        f"""
from app.n_hyenadna import hyena_test
hyena_test({fasta_file!r}, {checkpoint_dir!r}, {models_path!r})
"""
    )

    run_in_env(
        "mobiraph",
        f"""
from app.n_ensemble import ensemble_test
ensemble_test({fasta_file!r}, {checkpoint_dir!r}, {models_path!r})
"""
    )

    print("TEST: DONE")


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--fasta_file", required=True)
    train_parser.add_argument("--metadata_file", required=True)
    train_parser.add_argument("--checkpoint_dir", required=True)

    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--fasta_file", required=True)
    test_parser.add_argument("--models_path", required=True)
    test_parser.add_argument("--output_file", required=True)
    test_parser.add_argument("--checkpoint_dir", required=True)

    args = parser.parse_args()

    if args.command == "train":
        train(
            fasta_file=args.fasta_file,
            metadata_file=args.metadata_file,
            checkpoint_dir=args.checkpoint_dir
        )

    elif args.command == "test":
        test(
            fasta_file=args.fasta_file,
            models_path=args.models_path,
            output_file=args.output_file,
            checkpoint_dir=args.checkpoint_dir,
        )


if __name__ == "__main__":
    main()