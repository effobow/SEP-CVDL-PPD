from pathlib import Path

import yaml

from parc_mrh.pipeline import download, extract


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs/default.yml").read_text(encoding="utf-8"))
    raw = root / cfg["paths"]["raw_dir"]

    z2017 = raw / "downloads/logement_2017.zip"
    z2023 = raw / "downloads/logement_2023.zip"
    zgeo = raw / "downloads/table_passage_geo2003_geo2026.zip"

    download(cfg["sources"]["logement_2017"]["url"], z2017)
    download(cfg["sources"]["logement_2023"]["url"], z2023)
    download(cfg["geography"]["source_url"], zgeo)

    extract(z2017, raw / "logement_2017", (".csv",))
    extract(z2023, raw / "logement_2023", (".csv",))
    extract(zgeo, raw / "geographie", (".csv", ".xlsx", ".xls"))
    print("Sources INSEE téléchargées et extraites.")


if __name__ == "__main__":
    main()
