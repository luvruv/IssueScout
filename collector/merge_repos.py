"""Append a single repos .txt file or an entire directory of .txt files to repos.txt without duplicates."""
import os, glob


def load_repos(src):
    """Returns a list of repo strings from a file or directory of .txt files."""
    if os.path.isdir(src):
        files = sorted(glob.glob(os.path.join(src, "*.txt")))
    elif os.path.isfile(src):
        files = [src]
    else:
        files = []

    repos = []
    for f in files:
        with open(f, "r", encoding="utf-8", errors="ignore") as fp:
            for l in fp:
                l = l.strip()
                if l and not l.startswith("#"):
                    repos.append(l)
    return repos


def main(src="repo_lists", dst="repos.txt"):
    if not os.path.exists(dst):
        open(dst, "w", encoding="utf-8").close()
    have = {l.strip() for l in open(dst, encoding="utf-8") if l.strip() and not l.startswith("#")}
    all_incoming = load_repos(src)
    new = []
    for r in all_incoming:
        if r not in have:
            new.append(r)
            have.add(r)
    if new:
        with open(dst, "a", encoding="utf-8", newline="\n") as f:
            f.write(f"\n# added from {src}\n" + "\n".join(new) + "\n")
    print(f"Added {len(new)} new repos from '{src}' into '{dst}' (Total now: {len(have)})")
