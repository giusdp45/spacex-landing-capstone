"""Reproducible SpaceX capstone: scraping, SQL, visual EDA and model evaluation."""
from pathlib import Path
from io import StringIO
import calendar
import json
import re
import sqlite3
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, TimeSeriesSplit, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", palette=["#ef4444", "#2563eb"], context="talk")


def clean_text(value):
    return re.sub(r"\[[^\]]*\]", "", str(value)).replace("\u00a0", " ").strip()


def scrape_wikipedia():
    """Parse historical HTML tables, skipping mission-description rows and Falcon Heavy."""
    source = DATA / "wikipedia_launches.html"
    soup = BeautifulSoup(source.read_text(encoding="utf-8"), "html.parser")
    frames = []
    for table in soup.find_all("table", class_="wikitable"):
        headers = [h.get_text(" ", strip=True) for h in table.find_all("th")[:10]]
        if not headers or headers[0] != "Flight No." or "Booster landing" not in headers:
            continue
        table_df = pd.read_html(StringIO(str(table)))[0]
        if table_df.shape[1] != 10:
            continue
        table_df.columns = ["FlightNumber", "DateText", "BoosterVersion", "LaunchSite", "Payload", "PayloadMassText", "Orbit", "Customer", "LaunchOutcome", "LandingOutcome"]
        frames.append(table_df)
    df = pd.concat(frames, ignore_index=True)
    for col in df.columns:
        df[col] = df[col].map(clean_text)
    dates = df.DateText.str.extract(r"^(\d{1,2} [A-Za-z]+ \d{4})", expand=False)
    df = df[dates.notna() & df.BoosterVersion.str.startswith("F9")].copy()
    df["Date"] = pd.to_datetime(dates.loc[df.index], format="%d %B %Y")
    df["FlightNumber"] = pd.to_numeric(df.FlightNumber.str.extract(r"^(\d+)", expand=False))
    masses = df.PayloadMassText.str.extract(r"([\d,]+(?:\.\d+)?)\s*kg", expand=False)
    df["PayloadMass"] = pd.to_numeric(masses.str.replace(",", "", regex=False), errors="coerce")
    df["LaunchSite"] = df.LaunchSite.str.replace(",", "", regex=False).str.replace("SLC-", "SLC ", regex=False).str.replace("LC-", "LC ", regex=False)
    df["LaunchSuccess"] = df.LaunchOutcome.str.startswith("Success").astype(int)
    df["LandingSuccess"] = df.LandingOutcome.str.startswith("Success").astype(int)
    df["LandingType"] = np.select([df.LandingOutcome.str.contains("drone ship", case=False), df.LandingOutcome.str.contains("ground pad", case=False), df.LandingOutcome.str.contains("ocean", case=False)], ["Drone ship", "Ground pad", "Ocean"], default="Other / no attempt")
    df["LandingResult"] = np.select([df.LandingOutcome.str.startswith("Success"), df.LandingOutcome.str.startswith("Failure"), df.LandingOutcome.str.contains("No attempt", case=False)], ["Success", "Failure", "No attempt"], default="Other")
    df = df.drop_duplicates("FlightNumber").sort_values("Date").reset_index(drop=True)
    df["Date"] = df.Date.dt.strftime("%Y-%m-%d")
    df.to_csv(DATA / "wikipedia_scraped.csv", index=False)
    return df


SQL_QUERIES = {
    "01_unique_launch_sites": "SELECT DISTINCT LaunchSite FROM launches ORDER BY LaunchSite",
    "02_ccafs_first_five": "SELECT Date, FlightNumber, LaunchSite FROM launches WHERE LaunchSite LIKE 'CCAFS%' ORDER BY Date LIMIT 5",
    "03_nasa_crs_total_payload": "SELECT COUNT(*) AS Launches, COUNT(PayloadMass) AS KnownMasses, SUM(PayloadMass) AS KnownPayloadKg FROM launches WHERE Customer LIKE '%NASA%CRS%'",
    "04_f9_v1_1_mean_payload": "SELECT COUNT(*) AS Launches, COUNT(PayloadMass) AS KnownMasses, AVG(PayloadMass) AS MeanPayloadKg FROM launches WHERE BoosterVersion LIKE 'F9 v1.1%'",
    "05_first_successful_ground_landing": "SELECT MIN(Date) AS FirstSuccessfulGroundLanding FROM launches WHERE LandingSuccess=1 AND LandingType='Ground pad'",
    "06_drone_landings_payload_4000_6000": "SELECT Date, BoosterVersion, PayloadMass FROM launches WHERE LandingSuccess=1 AND LandingType='Drone ship' AND PayloadMass>4000 AND PayloadMass<6000 ORDER BY Date",
    "07_launch_outcome_counts": "SELECT LaunchSuccess, COUNT(*) AS Launches FROM launches GROUP BY LaunchSuccess",
    "08_max_payload_boosters": "SELECT Date, BoosterVersion, PayloadMass FROM launches WHERE PayloadMass=(SELECT MAX(PayloadMass) FROM launches) ORDER BY Date",
    "09_failed_drone_landings_2015": "SELECT strftime('%m',Date) AS Month, BoosterVersion, LaunchSite, LandingOutcome FROM launches WHERE LandingResult='Failure' AND LandingType='Drone ship' AND strftime('%Y',Date)='2015' ORDER BY Date",
    "10_landing_outcomes_ranked": "SELECT LandingResult, LandingType, COUNT(*) AS Launches FROM launches WHERE Date BETWEEN '2010-06-04' AND '2017-03-20' GROUP BY LandingResult, LandingType ORDER BY Launches DESC",
    "11_site_landing_success_rates": "SELECT LaunchSite, COUNT(*) AS Launches, SUM(LandingSuccess) AS SuccessfulLandings, ROUND(100.0*AVG(LandingSuccess),1) AS LandingSuccessPct FROM launches GROUP BY LaunchSite ORDER BY LandingSuccessPct DESC",
}


def sql_analysis(scraped):
    connection = sqlite3.connect(DATA / "spacex.sqlite")
    scraped.to_sql("launches", connection, if_exists="replace", index=False)
    results = {}
    for name, query in SQL_QUERIES.items():
        frame = pd.read_sql_query(query, connection)
        frame.to_csv(DATA / f"sql_{name}.csv", index=False)
        results[name] = frame.replace({np.nan: None}).to_dict(orient="records")
    (ROOT / "scripts" / "queries.sql").write_text("\n\n".join(f"-- {key}\n{value};" for key, value in SQL_QUERIES.items()), encoding="utf-8")
    connection.close()
    return results


def visual_eda(df):
    scatter_specs = [
        ("01_flight_site", "FlightNumber", "LaunchSite", "Flight sequence and launch site", "Flight number", "Launch site"),
        ("02_payload_site", "PayloadMass", "LaunchSite", "Payload and launch site", "Payload mass (kg)", "Launch site"),
        ("04_flight_orbit", "FlightNumber", "Orbit", "Flight sequence and orbit", "Flight number", "Orbit"),
        ("05_payload_orbit", "PayloadMass", "Orbit", "Payload and orbit", "Payload mass (kg)", "Orbit"),
    ]
    for filename, x, y, title, xlabel, ylabel in scatter_specs:
        fig, ax = plt.subplots(figsize=(13, 6))
        sns.scatterplot(data=df, x=x, y=y, hue="Class", style="Class", palette={0: "#ef4444", 1: "#2563eb"}, s=90, ax=ax)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.legend(title="Landing class")
        fig.tight_layout()
        fig.savefig(FIG / f"{filename}.png", dpi=170)
        plt.close(fig)
    orbit = df.groupby("Orbit").Class.agg(["mean", "size"]).sort_values("mean", ascending=False)
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(orbit.index, orbit["mean"] * 100, color="#2563eb")
    ax.set(title="Landing success rate by orbit", ylabel="Landing class 1 (%)", xlabel="Orbit", ylim=(0, 110))
    for index, row in enumerate(orbit.itertuples()):
        ax.text(index, row.mean * 100 + 2, f"n={row.size}", ha="center", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / "03_orbit_success.png", dpi=170)
    plt.close(fig)
    yearly = df.groupby(df.Date.dt.year).Class.agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(yearly.index, yearly["mean"] * 100, marker="o", linewidth=2.5, color="#2563eb")
    ax.set(title="Landing success rate by year", ylabel="Landing class 1 (%)", xlabel="Year", ylim=(0, 110))
    ax.set_xticks(yearly.index)
    fig.tight_layout()
    fig.savefig(FIG / "06_yearly_success.png", dpi=170)
    plt.close(fig)
    return orbit.reset_index().to_dict(orient="records"), yearly.reset_index().rename(columns={"Date": "Year"}).to_dict(orient="records")


def model_analysis(df):
    numeric = ["FlightNumber", "PayloadMass", "Flights", "Block"]
    categorical = ["Orbit", "LaunchSite", "GridFins", "Reused", "Legs"]
    X = df[numeric + categorical].copy()
    # Reverse the frozen dataset's exact mean-imputation sentinel, then impute per fold.
    X["PayloadMass"] = X.PayloadMass.mask(np.isclose(X.PayloadMass, 6104.959411764705, rtol=1e-10))
    # Categorical values have consistent string types for sklearn's encoder.
    X[categorical] = X[categorical].astype(str)
    y = df.Class
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ])
    models = {
        "Logistic regression": (LogisticRegression(max_iter=10000), {"model__C": [0.01, 0.1, 1, 10]}),
        "Support vector machine": (SVC(), {"model__kernel": ["linear", "rbf"], "model__C": [0.1, 1, 10], "model__gamma": ["scale", 0.1]}),
        "Decision tree": (DecisionTreeClassifier(random_state=42), {"model__max_depth": [2, 3, 4, 5, None], "model__min_samples_leaf": [1, 2, 4]}),
        "K nearest neighbors": (KNeighborsClassifier(), {"model__n_neighbors": [3, 5, 7, 9], "model__p": [1, 2], "model__weights": ["uniform", "distance"]}),
    }
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    records, fitted = [], {}
    for name, (model, params) in models.items():
        pipeline = Pipeline([("prepare", preprocess), ("model", model)])
        search = GridSearchCV(pipeline, params, scoring="accuracy", cv=cv, n_jobs=1)
        search.fit(X_train, y_train)
        predicted = search.predict(X_test)
        records.append({"Model": name, "CV_accuracy": search.best_score_, "CV_std": search.cv_results_["std_test_score"][search.best_index_],
                        "Test_accuracy": accuracy_score(y_test, predicted), "Test_balanced_accuracy": balanced_accuracy_score(y_test, predicted),
                        "Test_precision": precision_score(y_test, predicted, zero_division=0), "Test_recall": recall_score(y_test, predicted, zero_division=0),
                        "Test_F1": f1_score(y_test, predicted, zero_division=0), "Parameters": search.best_params_,
                        "Confusion_matrix": confusion_matrix(y_test, predicted, labels=[0, 1]).tolist()})
        fitted[name] = search
    # Select by training CV, not by inspecting the test accuracy.
    winner = max(records, key=lambda r: r["CV_accuracy"])
    comparison = pd.DataFrame([{k: v for k, v in r.items() if k not in ["Parameters", "Confusion_matrix"]} for r in records])
    comparison.to_csv(DATA / "model_comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(13, 6))
    shown = comparison.melt(id_vars="Model", value_vars=["CV_accuracy", "Test_accuracy"], var_name="Evaluation", value_name="Accuracy")
    sns.barplot(data=shown, x="Model", y="Accuracy", hue="Evaluation", palette=["#64748b", "#2563eb"], ax=ax)
    ax.set(title="Four classifiers: training cross-validation and held-out test", ylabel="Accuracy", xlabel="", ylim=(0, 1.05))
    ax.tick_params(axis="x", labelsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "07_model_comparison.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(winner["Confusion_matrix"], annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Predicted 0", "Predicted 1"], yticklabels=["Actual 0", "Actual 1"], ax=ax)
    ax.set_title(f"Best CV model: {winner['Model']}")
    fig.tight_layout()
    fig.savefig(FIG / "08_confusion_matrix.png", dpi=170)
    plt.close(fig)
    # An independent temporal evaluation highlights the shift toward later successful landings.
    split = int(len(X) * 0.8)
    temporal_records = []
    temporal_cv = TimeSeriesSplit(n_splits=3)
    for name, (model, params) in models.items():
        search = GridSearchCV(Pipeline([("prepare", preprocess), ("model", model)]), params, scoring="accuracy", cv=temporal_cv, n_jobs=1)
        search.fit(X.iloc[:split], y.iloc[:split])
        temporal_records.append({"Model": name, "CV_accuracy": search.best_score_, "Test_accuracy": accuracy_score(y.iloc[split:], search.predict(X.iloc[split:]))})
    temporal_best = max(temporal_records, key=lambda r: r["CV_accuracy"])
    return {"records": records, "winner": winner, "train_n": len(X_train), "test_n": len(X_test),
            "test_successes": int(y_test.sum()), "random_majority_baseline": float(y_test.mean()),
            "features": numeric + categorical, "temporal": temporal_records, "temporal_winner": temporal_best,
            "temporal_test_successes": int(y.iloc[split:].sum()), "temporal_baseline": float(y.iloc[split:].mean()),
            "temporal_test_start": df.Date.iloc[split].strftime("%Y-%m-%d")}


def main():
    df = pd.read_csv(DATA / "dataset_part_2.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    bad_outcomes = {"False ASDS", "False Ocean", "False RTLS", "None ASDS", "None None"}
    calculated = df.Outcome.map(lambda value: int(value not in bad_outcomes))
    assert calculated.equals(df.Class), "Landing outcome and Class disagree"
    assert df.FlightNumber.is_unique and len(df) == 90
    scraped = scrape_wikipedia()
    sql = sql_analysis(scraped)
    orbit, yearly = visual_eda(df)
    models = model_analysis(df)
    summary = {"analysis_n": len(df), "analysis_start": df.Date.min().strftime("%Y-%m-%d"), "analysis_end": df.Date.max().strftime("%Y-%m-%d"),
               "successes": int(df.Class.sum()), "success_rate": float(df.Class.mean()), "scraped_n": len(scraped),
               "scraped_start": scraped.Date.min(), "scraped_end": scraped.Date.max(), "scraped_missing_payload": int(scraped.PayloadMass.isna().sum()),
               "sql": sql, "orbit": orbit, "yearly": yearly, "models": models,
               "site_rates": df.groupby("LaunchSite").Class.agg(["sum", "size", "mean"]).reset_index().to_dict(orient="records"),
               "outcomes": df.Outcome.value_counts().to_dict(), "native_records": df.replace({np.nan: None}).assign(Date=df.Date.dt.strftime("%Y-%m-%d")).to_dict(orient="records")}
    (DATA / "analysis_summary.json").write_text(json.dumps(summary, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x)), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["analysis_n", "successes", "scraped_n", "scraped_end", "scraped_missing_payload"]}, indent=2))
    print(comparison_text(models))


def comparison_text(models):
    return json.dumps({"models": [{"Model": r["Model"], "CV": round(r["CV_accuracy"], 4), "Test": round(r["Test_accuracy"], 4)} for r in models["records"]], "winner": models["winner"], "temporal": models["temporal_winner"], "temporal_baseline": models["temporal_baseline"]}, indent=2)


if __name__ == "__main__":
    main()
