"""Build the paper's version of the 0.0.5 backend with a TpT-only fix."""

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from urllib.request import urlopen


NAME = "scikit_lexicographical_trees-0.0.5"
URL = (
    "https://files.pythonhosted.org/packages/53/ad/"
    f"406ad5663d663a44fcc101f9825104cf0cbc10491a1f385ac00ebccae265/{NAME}.tar.gz"
)
SHA256 = "67d6d705e3e4b17b1c24151d67d9c056840edadcc5e88c2739a46f6d789dce83"


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"Unexpected source for TpT patch: {old!r}")
    return text.replace(old, new, 1)


def patch_splitter(path):
    source = path.read_text(encoding="utf-8")
    start = source.index("cdef inline int node_TpT_split(")
    stop = source.index("# Sort n-element arrays", start)
    split = source[start:stop]
    split = replace_once(
        split,
        """        if feature_values[end - 1] <= feature_values[start] + FEATURE_THRESHOLD:
            features[f_j], features[n_total_constants] = features[n_total_constants], features[f_j]
            n_found_constants += 1
            n_total_constants += 1
            continue
""",
        """        if feature_values[end - partitioner.n_missing - 1] <= feature_values[start] + FEATURE_THRESHOLD:
            if partitioner.n_missing == 0:
                features[f_j], features[n_total_constants] = features[n_total_constants], features[f_j]
                n_found_constants += 1
                n_total_constants += 1
            continue
""",
    )
    # Duration rows are at the end, not on either of the ordinary branches.
    scan = split.index("        for search_idx in range(")
    end_scan = split.index("    if best_penalized_gain == -INFINITY", scan)
    body = split[scan:end_scan].splitlines(keepends=True)[1:]
    body = "".join(line[4:] if line.strip() else line for line in body)
    body = replace_once(
        body,
        "        criterion.missing_go_to_left = (search_idx == 1 and partitioner.n_missing != 0)\n",
        "        criterion.missing_go_to_left = False\n",
    )
    body = replace_once(
        body,
        """            right_span = end - current_split.pos
            if partitioner.n_missing != 0:
                if criterion.missing_go_to_left:
                    left_span -= partitioner.n_missing
                else:
                    right_span -= partitioner.n_missing
""",
        "            right_span = end - current_split.pos - partitioner.n_missing\n",
    )
    split = split[:scan] + body + split[end_scan:]
    # Recreate the candidate's sample order and duration totals before storing it.
    split = replace_once(
        split,
        """    partitioner.partition_samples_final(
        best_split.pos,
        best_split.threshold,
        best_split.feature,
        best_split.n_missing
    )

""",
        """    partitioner.sort_samples_and_feature_values(best_split.feature)
    criterion.missing_go_to_left = False
    criterion.init_missing(best_split.n_missing)
""",
    )
    path.write_text(source[:start] + split + source[stop:], encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-dir", type=Path, help="Build a wheel instead of installing")
    options = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="tpt-backend-") as directory:
        build = Path(directory)
        archive = build / f"{NAME}.tar.gz"
        cached = os.environ.get("TPT_BACKEND_SDIST")
        if cached:
            data = Path(cached).read_bytes()
        else:
            with urlopen(URL, timeout=60) as response:
                data = response.read()
        if hashlib.sha256(data).hexdigest() != SHA256:
            raise RuntimeError("Unexpected SHA-256 for the TpT backend source archive")
        archive.write_bytes(data)
        with tarfile.open(archive) as tar:
            tar.extractall(build, filter="data")
        package = build / NAME
        patch_splitter(package / "sklearn/tree/_splitter.pyx")
        pip = [sys.executable, "-m", "pip"]
        if options.wheel_dir is not None:
            action = ["wheel", "--wheel-dir", str(options.wheel_dir.resolve())]
        else:
            action = ["install", "--force-reinstall"]
        subprocess.run([*pip, *action, "--no-build-isolation", "--no-deps", str(package)], check=True)


if __name__ == "__main__":
    main()
