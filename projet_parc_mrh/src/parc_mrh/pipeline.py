from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
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
import statsmodels.api as sm
import yaml


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
    text = str(value).strip().lower().replace("\u00a0", " ")
    if text in NON_DATA:
        return np.nan
    text = re.sub(r"[^0-9eE+\\-.,]", "", text.replace(" ", "")).replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return np.nan


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    else:
        df = pd.read_excel(path)
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
            m for m in archive.infolist()
            if not m.is_dir() and m.filename.lower().endswith(suffixes)
        ]
        if not members:
            raise FileNotFoundError(f"Aucun fichier {suffixes} dans {zip_path}")
        member = max(members, key=lambda x: x.file_size)
        target = (out_dir / Path(member.filename).name).resolve()
        if out_dir.resolve() not in target.parents:
            raise ValueError("Archive non sûre")
        with archive.open(member) as src:
            target.write_bytes(src.read())
    return target


def find_column(df: pd.DataFrame, aliases: list[str], label: str) -> str:
    cols = {clean_col(c): c for c in df.columns}
    for alias in aliases:
        if clean_col(alias) in cols:
            return cols[clean_col(alias)]
    raise KeyError(f"Colonne introuvable pour {label}. Disponibles: {list(df.columns)[:60]}")


def historical(path: Path, year: int, cfg: dict) -> pd.DataFrame:
    df = read_table(path)
    geo = find_column(df, cfg["geo"], "code INSEE")
    dep = find_column(df, cfg["dep"], "département")
    log = find_column(df, cfg["total_logements"], "logements")
    rp = find_column(df, cfg["residences_principales"], "résidences principales")
    out = pd.DataFrame({
        "code_insee": df[geo].map(clean_code),
        "dep": df[dep].map(lambda x: clean_code(x, 2)),
        "nb_logements": df[log].map(number),
        "nb_residences_principales": df[rp].map(number),
    })
    out["annee"] = year
    out["part_residences_principales"] = (
        out["nb_residences_principales"] / out["nb_logements"].replace(0, np.nan)
    )
    return out


def source_2023_long(path: Path) -> pd.DataFrame:
    df = read_table(path)
    required = {
        "GEO", "GEO_OBJECT", "RP_MEASURE", "TIME_PERIOD", "OBS_STATUS",
        "OCS", "OBS_VALUE", "L_STAY", "TDW", "CARS", "CARPARK", "NOR",
        "TSH", "BUILD_END", "NRG_SRC",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans la source 2023: {sorted(missing)}")
    mask = (
        (df["GEO_OBJECT"].astype(str).str.upper() == "COM")
        & (df["RP_MEASURE"].astype(str).str.upper() == "DWELLINGS")
        & (pd.to_numeric(df["TIME_PERIOD"], errors="coerce") == 2023)
        & (df["OBS_STATUS"].astype(str).str.upper() == "A")
    )
    for col in ["L_STAY", "TDW", "CARS", "CARPARK", "NOR", "TSH", "BUILD_END", "NRG_SRC"]:
        mask &= df[col].astype(str) == "_T"
    work = df.loc[mask, ["GEO", "OCS", "OBS_VALUE"]].copy()
    work["OBS_VALUE"] = work["OBS_VALUE"].map(number)
    pivot = work.pivot_table(index="GEO", columns="OCS", values="OBS_VALUE", aggfunc="sum")
    if "_T" not in pivot or "DW_MAIN" not in pivot:
        raise ValueError("Les modalités OCS _T et DW_MAIN sont nécessaires")
    out = pd.DataFrame({
        "code_insee": pivot.index.map(clean_code),
        "dep": pivot.index.map(lambda x: clean_code(x, 2)),
        "nb_logements": pivot["_T"].to_numpy(),
        "nb_residences_principales": pivot["DW_MAIN"].to_numpy(),
    })
    out["annee"] = 2023
    out["part_residences_principales"] = (
        out["nb_residences_principales"] / out["nb_logements"].replace(0, np.nan)
    )
    return out.reset_index(drop=True)


def passage(path: Path, year: int, canonical: int) -> pd.DataFrame:
    df = read_table(path)
    old = find_column(
        df,
        [f"CODGEO_{year}", f"CODE_GEO_{year}", f"COM_{year}", "ANCIEN_CODE", "CODGEO_OLD"],
        "ancien code",
    )
    new = find_column(
        df,
        [f"CODGEO_{canonical}", f"CODE_GEO_{canonical}", f"COM_{canonical}", "NOUVEAU_CODE", "CODGEO_NEW"],
        "nouveau code",
    )
    return pd.DataFrame({
        "old_code": df[old].map(clean_code),
        "new_code": df[new].map(clean_code),
    }).drop_duplicates()


def harmonize(df: pd.DataFrame, mapping: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    counts = mapping.groupby("old_code")["new_code"].transform("nunique")
    work = mapping.copy()
    ref = reference[["code_insee", "nb_logements"]].copy()
    work = work.merge(ref, left_on="new_code", right_on="code_insee", how="left")
    totals = work.groupby("old_code")["nb_logements"].transform("sum")
    work["weight"] = np.where(
        counts > 1,
        work["nb_logements"] / totals,
        1.0,
    )
    if work.loc[counts > 1, "weight"].isna().any():
        raise ValueError("Allocation géographique impossible pour une scission sans référence 2023")
    merged = df.merge(work[["old_code", "new_code", "weight"]], left_on="code_insee", right_on="old_code", how="left")
    missing = merged["new_code"].isna().mean()
    if missing > 0.05:
        raise ValueError(f"{missing:.1%} des codes historiques ne sont pas mappés")
    for col in ["nb_logements", "nb_residences_principales"]:
        merged[col] = merged[col] * merged["weight"]
    out = merged.dropna(subset=["new_code"]).groupby("new_code", as_index=False).agg({
        "nb_logements": "sum",
        "nb_residences_principales": "sum",
        "dep": "first",
    }).rename(columns={"new_code": "code_insee"})
    out["part_residences_principales"] = (
        out["nb_residences_principales"] / out["nb_logements"].replace(0, np.nan)
    )
    return out


def logit_share(value: float, eps: float) -> float:
    return float(logit(np.clip(value, eps, 1 - eps)))


def share_from_logit(value: float) -> float:
    return float(np.clip(expit(value), 0, 1))


def linear_forecast(history: pd.DataFrame, years: list[int], eps: float) -> pd.DataFrame:
    rows = []
    for code, group in history.groupby("code_insee"):
        group = group.dropna(subset=["part_residences_principales"]).sort_values("annee")
        if len(group) < 2:
            continue
        model = LinearRegression().fit(
            group[["annee"]],
            group["part_residences_principales"].map(lambda x: logit_share(x, eps)),
        )
        for year in years:
            pred = share_from_logit(float(model.predict([[year]])[0]))
            rows.append({"code_insee": code, "annee": year, "part_predite": pred, "modele": "linear_logit"})
    return pd.DataFrame(rows)


def eapl_forecast(history: pd.DataFrame, eapl: pd.DataFrame, years: list[int], cfg: dict) -> pd.DataFrame:
    eps = cfg["probability_epsilon"]
    ref = eapl.sort_values("annee").set_index("annee")["part_rp_pct"] / 100
    all_years = sorted(set(history["annee"]).union(years))
    missing = [y for y in all_years if y not in ref.index]
    if missing:
        recent = ref.iloc[-cfg["recent_years_for_national_trend"]:]
        trend = LinearRegression().fit(
            recent.index.to_numpy().reshape(-1, 1),
            recent.map(lambda x: logit_share(x, eps)),
        )
        for year in missing:
            ref.loc[year] = share_from_logit(float(trend.predict([[year]])[0]))
    rows = []
    beta = []
    for code, group in history.groupby("code_insee"):
        group = group.set_index("annee")
        if 2012 not in group.index or 2023 not in group.index:
            b = 1.0
        else:
            local = logit_share(group.loc[2023, "part_residences_principales"], eps) - logit_share(group.loc[2012, "part_residences_principales"], eps)
            national = logit_share(ref.loc[2023], eps) - logit_share(ref.loc[2012], eps)
            b = 1.0 if abs(national) < 1e-6 else local / national
        beta.append({"code_insee": code, "dep": str(group["dep"].iloc[-1]), "beta": b})
    beta_df = pd.DataFrame(beta)
    dep_median = beta_df.groupby("dep")["beta"].median().to_dict()
    for code, group in history.groupby("code_insee"):
        last = group.sort_values("annee").iloc[-1]
        row = beta_df[beta_df["code_insee"] == code].iloc[0]
        b = 0.75 * row["beta"] + 0.25 * dep_median.get(str(row["dep"]), 1.0)
        b = float(np.clip(b, cfg["beta_clip_min"], cfg["beta_clip_max"]))
        base = logit_share(last["part_residences_principales"], eps)
        for year in years:
            shift = logit_share(ref.loc[year], eps) - logit_share(ref.loc[2023], eps)
            rows.append({
                "code_insee": code,
                "annee": year,
                "part_predite": share_from_logit(base + b * shift),
                "modele": "eapl_calibre_logit",
                "beta_local": row["beta"],
                "beta_regularise": b,
            })
    return pd.DataFrame(rows)


def metrics(y: pd.Series, pred: pd.Series) -> dict:
    err = pred.to_numpy() - y.to_numpy()
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(r2_score(y, pred)),
        "bias": float(np.mean(err)),
        "max_abs_error": float(np.max(np.abs(err))),
    }


def backtest(history: pd.DataFrame, eapl: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    records, diagnostics = [], []
    for target, train_end in [(2017, 2012), (2023, 2017)]:
        train = history[history["annee"] <= train_end]
        actual = history[history["annee"] == target][["code_insee", "part_residences_principales"]]
        pred_a = linear_forecast(train, [target], cfg["probability_epsilon"]).rename(columns={"part_predite": "pred"})
        pred_b = eapl_forecast(train, eapl, [target], cfg).rename(columns={"part_predite": "pred"})
        for name, pred in [("linear_logit", pred_a), ("eapl_calibre_logit", pred_b)]:
            merged = actual.merge(pred[["code_insee", "pred"]], on="code_insee")
            m = metrics(merged["part_residences_principales"], merged["pred"])
            records.append({"forecast_year": target, "model": name, "n": len(merged), **m})
            if len(merged) >= 30 and pred["pred"].nunique() > 1:
                residuals = merged["part_residences_principales"] - merged["pred"]
                exog = sm.add_constant(merged["pred"])
                lm, lm_p, f, f_p = het_breuschpagan(residuals, exog)
                diagnostics.append({
                    "forecast_year": target,
                    "model": name,
                    "lm_stat": float(lm),
                    "lm_pvalue": float(lm_p),
                    "f_stat": float(f),
                    "f_pvalue": float(f_p),
                    "n": len(merged),
                })
    return pd.DataFrame(records), pd.DataFrame(diagnostics)


def logistic_diagnostic(history: pd.DataFrame, threshold: float) -> pd.DataFrame:
    wide = history.pivot(index="code_insee", columns="annee", values="part_residences_principales")
    if not {2012, 2017, 2023}.issubset(wide.columns):
        return pd.DataFrame()
    x = pd.DataFrame({
        "rp_2012": wide[2012],
        "rp_2017": wide[2017],
        "delta": wide[2017] - wide[2012],
    }).dropna()
    y = (wide.loc[x.index, 2023] >= threshold).astype(int)
    if y.nunique() < 2 or len(x) < 100:
        return pd.DataFrame()
    groups = history.drop_duplicates("code_insee").set_index("code_insee").loc[x.index, "dep"].astype(str)
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(x, y, groups))
    model = Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression(max_iter=2000, random_state=42))])
    model.fit(x.iloc[train], y.iloc[train])
    score = model.predict_proba(x.iloc[test])[:, 1]
    pred = (score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y.iloc[test], pred, labels=[0, 1]).ravel()
    return pd.DataFrame([{
        "threshold_share": threshold,
        "precision": precision_score(y.iloc[test], pred, zero_division=0),
        "recall": recall_score(y.iloc[test], pred, zero_division=0),
        "f1": f1_score(y.iloc[test], pred, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "roc_auc": roc_auc_score(y.iloc[test], score) if y.iloc[test].nunique() == 2 else np.nan,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }])


def run(config_path: str = "configs/default.yml") -> None:
    root = Path(config_path).resolve().parent.parent
    cfg = yaml.safe_load((root / config_path).read_text(encoding="utf-8"))
    raw = root / cfg["paths"]["raw_dir"]
    processed = root / cfg["paths"]["processed_dir"]
    processed.mkdir(parents=True, exist_ok=True)

    f2017 = raw / "logement_2017" / "base_cc_logement_2017.csv"
    f2023 = raw / "logement_2023" / "DS_RP_LOGEMENT_PRINC_2023_data.csv"
    passage_files = list((raw / "geographie").glob("*"))
    passage_file = next(p for p in passage_files if p.suffix.lower() in {".csv", ".xlsx", ".xls"})
    eapl = pd.read_csv(root / cfg["paths"]["eapl_file"])

    d23 = source_2023_long(f2023)
    d12 = harmonize(historical(f2017, 2012, cfg["columns"]["2012"]), passage(passage_file, 2012, 2026), d23)
    d17 = harmonize(historical(f2017, 2017, cfg["columns"]["2017"]), passage(passage_file, 2017, 2026), d23)
    history = pd.concat([d12, d17, d23], ignore_index=True)
    history = history.drop_duplicates(["code_insee", "annee"]).sort_values(["code_insee", "annee"])
    if (history["nb_residences_principales"] > history["nb_logements"]).any():
        raise ValueError("Résidences principales > logements")
    history.to_csv(processed / "commune_panel.csv", index=False)

    bt, diag = backtest(history, eapl, cfg["model"])
    bt.to_csv(processed / "backtest_metrics.csv", index=False)
    diag.to_csv(processed / "diagnostic_tests.csv", index=False)

    primary = "eapl_calibre_logit"
    if not bt.empty:
        mean_mae = bt.groupby("model")["mae"].mean()
        if mean_mae["linear_logit"] < mean_mae["eapl_calibre_logit"]:
            primary = "linear_logit"

    years = list(range(cfg["project"]["forecast_start_year"], cfg["project"]["output_year_end"] + 1))
    if primary == "linear_logit":
        future = linear_forecast(history, years, cfg["model"]["probability_epsilon"])
    else:
        future = eapl_forecast(history, eapl, years, cfg["model"])
    future["statut_donnee"] = "PREDITE"
    future["part_residences_principales"] = future["part_predite"]

    observed = history.copy()
    observed["statut_donnee"] = "OBSERVEE"
    observed["modele"] = "recensement"
    interpolated = []
    for code, group in history.groupby("code_insee"):
        group = group.sort_values("annee")
        for left, right in zip(group.iloc[:-1].itertuples(), group.iloc[1:].itertuples()):
            for year in range(left.annee + 1, right.annee):
                z1 = logit_share(left.part_residences_principales, cfg["model"]["probability_epsilon"])
                z2 = logit_share(right.part_residences_principales, cfg["model"]["probability_epsilon"])
                pred = share_from_logit(z1 + (year-left.annee)/(right.annee-left.annee)*(z2-z1))
                interpolated.append({"code_insee": code, "dep": left.dep, "annee": year, "part_residences_principales": pred, "statut_donnee": "INTERPOLEE", "modele": "interpolation_logit"})
    result = pd.concat([observed, pd.DataFrame(interpolated), future], ignore_index=True, sort=False)
    result = result.sort_values(["code_insee", "annee"]).drop_duplicates(["code_insee", "annee"], keep="last")
    result.to_csv(processed / "predictions_long.csv", index=False)
    result.pivot(index="code_insee", columns="annee", values="part_residences_principales").reset_index().to_csv(processed / "predictions_wide.csv", index=False)

    logit_result = logistic_diagnostic(history, cfg["model"]["classification_threshold"])
    logit_result.to_csv(processed / "logistic_metrics.csv", index=False)
    (processed / "model_registry.json").write_text(json.dumps({
        "target": "part_residences_principales",
        "primary_model": primary,
        "baseline": "linear_logit",
        "secondary_model": "logistic_regression_threshold",
        "output_year_end": cfg["project"]["output_year_end"],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
