import numpy as np
import pandas as pd

from parc_mrh.pipeline import clean_code, logit_share, share_from_logit, source_long


def test_insee_code_keeps_leading_zeroes():
    assert clean_code(1) == "00001"
    assert clean_code("01001") == "01001"


def test_logit_share_is_bounded():
    for value in [0.01, 0.5, 0.99]:
        transformed = logit_share(value, 0.0005)
        assert np.isfinite(transformed)
        assert 0 <= share_from_logit(transformed) <= 1


def test_historical_share_formula():
    df = pd.DataFrame({
        "nb_logements": [100, 250],
        "nb_residences_principales": [80, 200],
    })
    share = df["nb_residences_principales"] / df["nb_logements"]
    assert share.tolist() == [0.8, 0.8]


def test_source_long_extracts_all_observed_years(tmp_path):
    rows = []
    dimensions = {
        "GEO_OBJECT": "COM",
        "RP_MEASURE": "DWELLINGS",
        "OBS_STATUS": "A",
        "L_STAY": "_T",
        "TDW": "_T",
        "CARS": "_T",
        "CARPARK": "_T",
        "NOR": "_T",
        "TSH": "_T",
        "BUILD_END": "_T",
        "NRG_SRC": "_T",
    }
    for year, total, main in [(2012, 100, 80), (2017, 110, 88), (2023, 120, 96)]:
        for ocs, value in [("_T", total), ("DW_MAIN", main)]:
            rows.append({
                "GEO": "01001",
                "TIME_PERIOD": year,
                "OCS": ocs,
                "OBS_VALUE": value,
                **dimensions,
            })

    path = tmp_path / "source.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    result = source_long(path, [2012, 2017, 2023])

    assert result["annee"].tolist() == [2012, 2017, 2023]
    assert result["part_residences_principales"].tolist() == [0.8, 0.8, 0.8]
