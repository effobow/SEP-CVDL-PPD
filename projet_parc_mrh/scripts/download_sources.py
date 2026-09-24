from pathlib import Path

import yaml

from parc_mrh.pipeline import download, extract


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "configs/default.yml").read_text(encoding="utf-8")
    )
    raw = root / config["paths"]["raw_dir"]
    archive = raw / "downloads/logement_2023.zip"

    download(config["sources"]["logement_2023"]["url"], archive)
    extract(
        archive,
        raw / "logement_2023",
        (".csv",),
    )
    print("Source INSEE 2023 téléchargée et extraite.")


if __name__ == "__main__":
    main()
