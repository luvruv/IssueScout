"""Append repos_extra.txt to repos.txt without duplicates:  python cli.py add-repos repos_extra.txt"""
def main(src="repos_extra.txt", dst="repos.txt"):
    have = {l.strip() for l in open(dst) if l.strip() and not l.startswith("#")}
    new = []
    for l in open(src):
        l = l.strip()
        if l and not l.startswith("#") and l not in have:
            new.append(l); have.add(l)
    with open(dst, "a", newline="\n") as f:
        f.write("\n# added from " + src + "\n" + "\n".join(new) + "\n")
    print(f"added {len(new)} repos to {dst}")
