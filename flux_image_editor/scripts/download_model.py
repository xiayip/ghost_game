#!/usr/bin/env python3
"""Pre-download FLUX.2 [klein] 4B into the mounted Hugging Face cache."""

from huggingface_hub import snapshot_download


def main():
    path = snapshot_download('black-forest-labs/FLUX.2-klein-4B')
    print(path)


if __name__ == '__main__':
    main()

