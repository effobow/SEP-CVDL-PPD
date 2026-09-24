from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
import yaml
from scipy.special import expit, logit
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan


NON_DATA = {"", "-", "--", "nd", "s", "ns", "na", "n/a", "nan"}


def clean_col(value: object) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(value).strip().upper()).strip("_")


def clean_code(value: object, width: int = 5) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(width) if text.isdigit() and len(text) < width else text


def number(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower().replace(" ", " ")
    if text in NON_DATA:
        return np.nan
    text = re.sub(r"[^0-9eE+\-.,]", "", text.replace(" ", "")).replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return np.nan


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [clean_col(c) for c in df.columns]
    return df


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    path.write_bytes(response.content)


def extract(zip_path: Path, out_dir: Path, suffixes: tuple[str, ...]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        members = [
            item for item in archive.infolist()
            if not item.is_dir() and item.filename.lower().endswith(suffixes)
        ]
        if not members:
            raise FileNotFoundError(f"Aucun fichier {suffixes} dans {zip_path}")
        member = max(members, key=lambda item: item.file_size)
        target = (out_dir / Path(member.filename).name).resolve()
        if out_dir.resolve() not in target.parents:
            raise ValueError("Archive non sûre")
        with archive.open(member) as source:
            target.write_bytes(source.read())
    return target


def source_long(path: Path, years: list[int]) -> pd.DataFrame:
    df = read_csv(path)
    required = {
        "GEO", "GEO_OBJECT", "RP_MEASURE", "TIME_PERIOD", "OBS_STATUS",
        "OCS", "OBS_VALUE", "L_STAY", "TDW", "CARS", "CARPARK", "NOR",
        "TSH", "BUILD_END", "NRG_SRC",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes: {sorted(missing)}")

    mask = (
        (df["GEO_OBJECT"].astype(str).str.upper() == "COM")
        & (df["RP_MEASURE"].astype(str).str.upper() == "DWELLINGS")
        & df["TIME_PERIOD"].isin(years)
        & (df["OBS_STATUS"].astype(str).str.upper() == "A")
    )
    for column in [
        "L_STAY", "TDW", "CARS", "CARPARK", "NOR",
        "TSH", "BUILD_END", "NRG_SRC",
    ]:
        mask &= df[column].astype(str) == "_T"

    work = df.loc[mask, ["GEO", "TIME_PERIOD", "OCS", "OBS_VALUE"]].copy()
    work["OBS_VALUE"] = work["OBS_VALUE"].map(number)
    work["TIME_PERIOD"] = pd.to_numeric(work["TIME_PERIOD"], errors="coerce").astype(int)

    pivot = work.pivot_table(
        index=["GEO", "TIME_PERIOD"],
        columns="OCS",
        values="OBS_VALUE",
        aggfunc="sum",
    )
    if "_T" not in pivot or "DW_MAIN" not in pivot:
        raise ValueError("Les modalités OCS _T et DW_MAIN sont nécessaires")

    result = pd.DataFrame({
        "code_insee": pivot.index.get_level_values("GEO").map(clean_code),
        "annee": pivot.index.get_level_values("TIME_PERIOD"),
        "nb_logements": pivot["_T"].to_numpy(),
        "nb_residences_principales": pivot["DW_MAIN"].to_numpy(),
    })
    result["dep"] = result["code_insee"].str[:2]
    result["part_residences_principales"] = (
        result["nb_residences_principales"]
        / result["nb_logements"].replace(0, np.nan)
    )
    result = result.sort_values(["code_insee", "annee"]).reset_index(drop=True)
    return result


def logit_share(value: float, epsilon: float) -> float:
    return float(logit(np.clip(value, epsilon, 1 - epsilon)))


def share_from_logit(value: float) -> float:
    return float(np.clip(expit(value), 0, 1))


def linear_forecast(
    history: pd.DataFrame,
    years: list[int],
    epsilon: float,
) -> pd.DataFrame:
    rows = []
    for code, group in history.groupby("code_insee"):
        group = group.dropna(subset=["part_residences_principales"]).sort_values("annee")
        if len(group) < 2:
            continue
        model = LinearRegression().fit(
            group[["annee"]],
            group["part_residences_principales"].map(
                lambda value: logit_share(value, epsilon)
            ),
        )
        for year in years:
            prediction = share_from_logit(float(model.predict([[year]])[0]))
            rows.append({
                "code_insee": code,
                "annee": year,
                "part_predite": prediction,
                "modele": "linear_logit",
            })
    return pd.DataFrame(rows)


def eapl_forecast(
    history: pd.DataFrame,
    eapl: pd.DataFrame,
    years: list[int],
    cfg: dict,
) -> pd.DataFrame:
    epsilon = cfg["probability_epsilon"]
    reference = eapl.sort_values("annee").set_index("annee")["part_rp_pct"] / 100
    required_years = sorted(set(history["annee"]).union(years))

    missing = [year for year in required_years if year not in reference.index]
    if missing:
        recent = reference.iloc[-cfg["recent_years_for_national_trend"]:]
        trend = LinearRegression().fit(
            recent.index.to_numpy().reshape(-1, 1),
            recent.map(lambda value: logit_share(value, epsilon)),
        )
        for year in missing:
            reference.loc[year] = share_from_logit(
                float(trend.predict([[year]])[0])
            )

    beta_rows = []
    for code, group in history.groupby("code_insee"):
        indexed = group.set_index("annee")
        if 2012 not in indexed.index or 2023 not in indexed.index:
            beta = 1.0
        else:
            local_delta = (
                logit_share(indexed.loc[2023, "part_residences_principales"], epsilon)
                - logit_share(indexed.loc[2012, "part_residences_principales"], epsilon)
            )
            national_delta = (
                logit_share(reference.loc[2023], epsilon)
                - logit_share(reference.loc[2012], epsilon)
            )
            beta = 1.0 if abs(national_delta) < 1e-6 else local_delta / national_delta
        beta_rows.append({
            "code_insee": code,
            "dep": str(indexed["dep"].iloc[-1]),
            "beta": beta,
        })

    beta_df = pd.DataFrame(beta_rows)
    department_median = beta_df.groupby("dep")["beta"].median().to_dict()
    rows = []

    for code, group in history.groupby("code_insee"):
        last = group.sort_values("annee").iloc[-1]
        beta_row = beta_df[beta_df["code_insee"] == code].iloc[0]
        beta = (
            cfg["shrinkage_weight_local"] * beta_row["beta"]
            + (1 - cfg["shrinkage_weight_local"])
            * department_median.get(str(beta_row["dep"]), 1.0)
        )
        beta = float(np.clip(beta, cfg["beta_clip_min"], cfg["beta_clip_max"]))
        base = logit_share(last["part_residences_principales"], epsilon)

        for year in years:
            shift = (
                logit_share(reference.loc[year], epsilon)
                - logit_share(reference.loc[2023], epsilon)
            )
            rows.append({
                "code_insee": code,
                "annee": year,
                "part_predite": share_from_logit(base + beta * shift),
                "modele": "eapl_calibre_logit",
                "beta_local": beta_row["beta"],
                "beta_regularise": beta,
            })

    return pd.DataFrame(rows)


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    error = y_pred.to_numpy() - y_true.to_numpy()
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(r2_score(y_true, y_pred)),
        "bias": float(np.mean(error)),
        "max_abs_error": float(np.max(np.abs(error))),
    }


def backtest(
    history: pd.DataFrame,
    eapl: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    diagnostics = []

    for target_year, train_end in [(2017, 2012), (2023, 2017)]:
        train = history[history["annee"] <= train_end]
        actual = history[history["annee"] == target_year][
            ["code_insee", "part_residences_principales"]
        ]

        candidates = [
            (
                "linear_logit",
                linear_forecast(train, [target_year], cfg["probability_epsilon"]),
            ),
            (
                "eapl_calibre_logit",
                eapl_forecast(train, eapl, [target_year], cfg),
            ),
        ]

        for model_name, prediction in candidates:
            merged = actual.merge(
                prediction[["code_insee", "part_predite"]],
                on="code_insee",
                how="inner",
            )
            if merged.empty:
                continue

            model_metrics = regression_metrics(
                merged["part_residences_principales"],
                merged["part_predite"],
            )
            records.append({
                "forecast_year": target_year,
                "model": model_name,
                "n": len(merged),
                **model_metrics,
            })

            if len(merged) >= 30 and merged["part_predite"].nunique() > 1:
                residuals = (
                    merged["part_residences_principales"] - merged["part_predite"]
                )
                exog = sm.add_constant(merged["part_predite"])
                lm, lm_p, f_stat, f_p = het_breuschpagan(residuals, exog)
                diagnostics.append({
                    "forecast_year": target_year,
                    "model": model_name,
                    "lm_stat": float(lm),
                    "lm_pvalue": float(lm_p),
                    "f_stat": float(f_stat),
                    "f_pvalue": float(f_p),
                    "n": len(merged),
                })

    return pd.DataFrame(records), pd.DataFrame(diagnostics)


def logistic_diagnostic(
    history: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    wide = history.pivot(
        index="code_insee",
        columns="annee",
        values="part_residences_principales",
    )
    if not {2012, 2017, 2023}.issubset(wide.columns):
        return pd.DataFrame()

    features = pd.DataFrame({
        "rp_2012": wide[2012],
        "rp_2017": wide[2017],
        "delta": wide[2017] - wide[2012],
    }).dropna()
    target = (wide.loc[features.index, 2023] >= threshold).astype(int)

    if target.nunique() < 2 or len(features) < 100:
        return pd.DataFrame()

    groups = (
        history.drop_duplicates("code_insee")
        .set_index("code_insee")
        .loc[features.index, "dep"]
        .astype(str)
    )
    train, test = next(
        GroupShuffleSplit(
            n_splits=1,
            test_size=0.25,
            random_state=42,
        ).split(features, target, groups)
    )

    model = Pipeline([
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    model.fit(features.iloc[train], target.iloc[train])
    score = model.predict_proba(features.iloc[test])[:, 1]
    predicted = (score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        target.iloc[test],
        predicted,
        labels=[0, 1],
    ).ravel()

    return pd.DataFrame([{
        "threshold_share": threshold,
        "precision": precision_score(
            target.iloc[test], predicted, zero_division=0
        ),
        "recall": recall_score(
            target.iloc[test], predicted, zero_division=0
        ),
        "f1": f1_score(target.iloc[test], predicted, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "roc_auc": (
            roc_auc_score(target.iloc[test], score)
            if target.iloc[test].nunique() == 2
            else np.nan
        ),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }])


def run(config_path: str = "configs/default.yml") -> None:
    root = Path(config_path).resolve().parent.parent
    config = yaml.safe_load(
        (root / config_path).read_text(encoding="utf-8")
    )
    raw = root / config["paths"]["raw_dir"]
    processed = root / config["paths"]["processed_dir"]
    processed.mkdir(parents=True, exist_ok=True)

    source = raw / "logement_2023" / "DS_RP_LOGEMENT_PRINC_2023_data.csv"
    eapl = pd.read_csv(root / config["paths"]["eapl_file"])

    history = source_long(source, config["project"]["observed_years"])
    history = history.drop_duplicates(
        ["code_insee", "annee"]
    ).sort_values(["code_insee", "annee"])

    if (history["nb_residences_principales"] > history["nb_logements"]).any():
        raise ValueError("Résidences principales > logements")

    history.to_csv(processed / "commune_panel.csv", index=False)

    backtest_metrics, diagnostics = backtest(history, eapl, config["model"])
    backtest_metrics.to_csv(processed / "backtest_metrics.csv", index=False)
    diagnostics.to_csv(processed / "diagnostic_tests.csv", index=False)

    primary_model = "eapl_calibre_logit"
    if not backtest_metrics.empty:
        mean_mae = backtest_metrics.groupby("model")["mae"].mean()
        if mean_mae["linear_logit"] < mean_mae["eapl_calibre_logit"]:
            primary_model = "linear_logit"

    forecast_years = list(
        range(
            config["project"]["forecast_start_year"],
            config["project"]["output_year_end"] + 1,
        )
    )
    if primary_model == "linear_logit":
        future = linear_forecast(
            history,
            forecast_years,
            config["model"]["probability_epsilon"],
        )
    else:
        future = eapl_forecast(
            history,
            eapl,
            forecast_years,
            config["model"],
        )

    future["statut_donnee"] = "PREDITE"
    future["part_residences_principales"] = future["part_predite"]

    observed = history.copy()
    observed["statut_donnee"] = "OBSERVEE"
    observed["modele"] = "recensement"

    interpolated = []
    for code, group in history.groupby("code_insee"):
        group = group.sort_values("annee")
        for left, right in zip(
            group.iloc[:-1].itertuples(),
            group.iloc[1:].itertuples(),
        ):
            for year in range(left.annee + 1, right.annee):
                z_left = logit_share(
                    left.part_residences_principales,
                    config["model"]["probability_epsilon"],
                )
                z_right = logit_share(
                    right.part_residences_principales,
                    config["model"]["probability_epsilon"],
                )
                ratio = (year - left.annee) / (right.annee - left.annee)
                prediction = share_from_logit(
                    z_left + ratio * (z_right - z_left)
                )
                interpolated.append({
                    "code_insee": code,
                    "dep": left.dep,
                    "annee": year,
                    "part_residences_principales": prediction,
                    "statut_donnee": "INTERPOLEE",
                    "modele": "interpolation_logit",
                })

    result = pd.concat(
        [observed, pd.DataFrame(interpolated), future],
        ignore_index=True,
        sort=False,
    )
    result = result.sort_values(
        ["code_insee", "annee"]
    ).drop_duplicates(["code_insee", "annee"], keep="last")

    result.to_csv(processed / "predictions_long.csv", index=False)
    result.pivot(
        index="code_insee",
        columns="annee",
        values="part_residences_principales",
    ).reset_index().to_csv(
        processed / "predictions_wide.csv",
        index=False,
    )

    logistic = logistic_diagnostic(
        history,
        config["model"]["classification_threshold"],
    )
    logistic.to_csv(processed / "logistic_metrics.csv", index=False)

    registry = {
        "target": "part_residences_principales",
        "primary_model": primary_model,
        "baseline": "linear_logit",
        "secondary_model": "logistic_regression_threshold",
        "observed_years": config["project"]["observed_years"],
        "output_year_end": config["project"]["output_year_end"],
    }
    (processed / "model_registry.json").write_text(
        json.dumps(registry, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
