"""Append a single repos .txt file or an entire directory of .txt files to repos.txt without duplicates.
Supports creating new modular repo lists and dynamic directory ingestion.
"""
import os
import glob
from typing import List, Set, Union


def load_repos(src: Union[str, List[str]]) -> List[str]:
    """Returns a deduplicated list of repo strings from a file, list of files, or directory of .txt files."""
    if isinstance(src, list):
        files = src
    elif os.path.isdir(src):
        files = sorted(glob.glob(os.path.join(src, "*.txt")))
    elif os.path.isfile(src):
        files = [src]
    else:
        files = []

    repos: List[str] = []
    seen: Set[str] = set()

    for f in files:
        if not os.path.isfile(f):
            continue
        with open(f, "r", encoding="utf-8", errors="ignore") as fp:
            for line in fp:
                l = line.strip()
                if l and not l.startswith("#"):
                    # Normalize repo format (e.g. remove trailing slashes or github.com prefix)
                    if "github.com/" in l:
                        l = l.split("github.com/")[-1].strip("/")
                    if l not in seen:
                        repos.append(l)
                        seen.add(l)
    return repos


def create_repo_list(name: str, repos: List[str], directory: str = "repo_lists") -> str:
    """Creates a new modular repo list document (.txt) in the specified directory."""
    os.makedirs(directory, exist_ok=True)
    clean_name = name.strip()
    if not clean_name.endswith(".txt"):
        clean_name += ".txt"
    
    file_path = os.path.join(directory, clean_name)
    clean_repos = []
    for r in repos:
        r_str = r.strip()
        if "github.com/" in r_str:
            r_str = r_str.split("github.com/")[-1].strip("/")
        if r_str and not r_str.startswith("#"):
            clean_repos.append(r_str)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"# Modular Repository List: {clean_name}\n")
        f.write("# Generated via IssueScout Repo Management\n\n")
        f.write("\n".join(sorted(set(clean_repos))) + "\n")

    return file_path


def main(src: str = "repo_lists", dst: str = "repos.txt"):
    """Merges all unique repositories from `src` (file or directory) into `dst`."""
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
            f.write(f"\n# Merged from {src}\n" + "\n".join(sorted(new)) + "\n")
        print(f"Added {len(new)} new unique repos from '{src}' into '{dst}' (Total active: {len(have)})")
    else:
        print(f"No new repos to add from '{src}' (All {len(all_incoming)} repos already exist in '{dst}')")
