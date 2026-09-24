import numpy as np
import pandas as pd

from parc_mrh.pipeline import clean_code, logit_share, share_from_logit


def test_insee_code_keeps_leading_zeroes():
    assert clean_code(1) == "00001"
    assert clean_code("01001") == "01001"


def test_logit_share_is_bounded():
    values = [0.01, 0.5, 0.99]
    for value in values:
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
