# build index path for every .mp4 file in the data_root
import click
import argparse
from pathlib import Path

def build_index_recursive(path: Path):
    """
    Recursively find all folders containing frames.mp4 files.
    """
    index = []
    for item in sorted(path.iterdir()):
        if item.is_dir():
            # check if frames.mp4 exists
            if (item / "frames.mp4").exists():
                index.append(item)
            else:
                # recursively check subdirectories
                index.extend(build_index_recursive(item))
    return index

@click.command()
@click.argument(
    "data_root",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
)
def main(data_root: str):
    """
    Get relative path folder for all scene under the data_root.
    The folder must contain frames.mp4 files.
    save to .index.txt under data_root.
    notice that folder can be recursive.
    """
    data_root = Path(data_root)
    index = build_index_recursive(data_root)
    index = [Path(scene).relative_to(data_root).as_posix() for scene in index]
    # recursively 
    index_path = data_root / ".index.txt"
    with open(index_path, "w") as f:
        for scene in index:
            f.write(f"{scene}\n")
    print(f"Index saved to {data_root}/.index.txt")
    
if __name__ == "__main__":
    main()