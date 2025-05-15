import os
from pathlib import Path

def generate_index(root_dir, output_file='.index.txt', target_file='start_image.png'):
    """
    Traverse the root directory, find all subdirectories containing the target_file,
    sort their relative paths, and write them into output_file, one per line.

    :param root_dir: The root directory to search.
    :param output_file: The name of the output index file (will be created in root_dir).
    :param target_file: The filename to look for in subdirectories.
    """
    root_path = Path(root_dir)
    if not root_path.is_dir():
        raise ValueError(f"Provided root_dir '{root_dir}' is not a directory.")

    # Collect relative paths containing the target_file
    relative_paths = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        if target_file in filenames:
            # Compute relative path from root_dir
            rel_path = Path(dirpath).relative_to(root_path)
            # Convert to posix style (forward slashes), or use str(rel_path)
            relative_paths.append(str(rel_path))

    # Sort the paths
    relative_paths.sort()

    # Write to output file
    output_path = root_path / output_file
    with open(output_path, 'w', encoding='utf-8') as f:
        for path in relative_paths:
            f.write(f"{path}\n")

    print(f"Index written to {output_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate index of directories containing start_image.png')
    parser.add_argument('root_dir', help='The root directory to search')
    parser.add_argument('--output', '-o', default='.index.txt', help='Output index filename')
    parser.add_argument('--file', '-f', default='start_image.png', help='Target filename to search for')
    args = parser.parse_args()

    generate_index(args.root_dir, args.output, args.file)
