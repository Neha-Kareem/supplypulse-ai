"""SupplyPulse AI - generic store pipeline.

Takes ANY store's sales table (different column names, optional stock/invoice columns),
cleans it, trains a forecasting model on that store's own history, tests itself on the
most recent part of the data, forecasts the next 1/7/14/30 days, and turns the forecast
into stockout risk + order advice + suspicious-transaction flags.
"""
import io
import math
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

HORIZONS = [1, 7, 14, 30]
MIN_DAYS_TOTAL = 90          # less than this: refuse (forecasts would be meaningless)
GOOD_HISTORY_DAYS = 180      # less than this: warn
MAX_PRODUCTS = 300

# ---------------------------------------------------------------- column guessing
SYNONYMS = {
    "date": ["date", "order date", "sale date", "sales date", "invoice date", "transaction date",
             "bill date", "datetime", "timestamp", "day", "posting date"],
    "quantity": ["quantity", "qty", "units sold", "units", "sales", "sales quantity", "sales_quantity",
                 "quantity sold", "sold", "pcs", "qty sold", "sale qty", "demand"],
    "product": ["product", "item", "sku", "sku id", "sku_id", "product name", "item name", "product id",
                "item code", "article", "product code", "description"],
    "stock": ["stock", "inventory", "on hand", "stock on hand", "current stock", "current_stock",
              "closing stock", "qty on hand", "balance", "stock level", "inventory level"],
    "invoice": ["invoice", "invoice no", "invoice number", "invoice id", "invoice_id", "bill no",
                "bill number", "receipt", "receipt no", "order id", "transaction id", "bill"],
    "price": ["price", "unit price", "amount", "revenue", "sales value", "sales_value", "total",
              "value", "line total"],
}
FIELD_ORDER = ["date", "quantity", "product", "stock", "invoice", "price"]
REQUIRED = ["date", "quantity"]


def _norm(name):
    return re.sub(r"[\s_\-\.]+", " ", str(name).strip().lower())


def _to_number(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (series.astype(str).str.replace(r"[,\s]", "", regex=True)
               .str.replace(r"[^\d\.\-eE]", "", regex=True).replace("", np.nan))
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_dates(series):
    """Read dates in any common layout. For ambiguous ones (04/10/2025) pick day-first or month-first
    by checking which reading makes consecutive rows step through time smoothly."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series, "already a date"
    s = series.astype(str).str.strip()
    cands = []
    for dayfirst in (False, True):
        parsed = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst, format="mixed")
        ok = parsed.dropna()
        jump = float(ok.diff().abs().dt.days.mean()) if len(ok) > 1 else 0.0
        cands.append((round(float(parsed.notna().mean()), 4), -jump, dayfirst, parsed))
    best = max(cands, key=lambda c: (c[0], c[1]))
    note = "day first (31/12/2025)" if best[2] else "month first or year first"
    return best[3], note


def guess_columns(df):
    """Return {field: column name or None} by matching column names, then sanity-checking values."""
    cols = list(df.columns)
    norm = {c: _norm(c) for c in cols}
    used, result = set(), {f: None for f in FIELD_ORDER}

    def acceptable(field, col):
        s = df[col]
        if field == "date":
            return _parse_dates(s.head(500))[0].notna().mean() >= 0.8
        if field in ("quantity", "stock", "price"):
            return _to_number(s.head(500)).notna().mean() >= 0.8
        return True

    for field in FIELD_ORDER:                       # exact name match first
        for syn in SYNONYMS[field]:
            hit = [c for c in cols if norm[c] == syn and c not in used]
            if hit and acceptable(field, hit[0]):
                result[field] = hit[0]; used.add(hit[0]); break
    for field in FIELD_ORDER:                       # then "contains" match
        if result[field] is not None:
            continue
        for syn in SYNONYMS[field]:
            hit = [c for c in cols if syn in norm[c] and c not in used and len(syn) >= 3]
            hit = [c for c in hit if acceptable(field, c)]
            if hit:
                result[field] = hit[0]; used.add(hit[0]); break
    return result


# ---------------------------------------------------------------- reading a file
def read_table(file_obj, filename):
    """Read a CSV or Excel upload into a DataFrame."""
    name = filename.lower()
    raw = file_obj.read() if hasattr(file_obj, "read") else file_obj
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw))
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, sep=None, engine="python")
        except Exception:
            continue
    raise ValueError("Could not read this file. Please upload a .csv or .xlsx file.")


# ---------------------------------------------------------------- cleaning
def clean_data(df, mapping):
    """Standardise the user's columns. Returns (transactions, report).

    report = {"errors": [...], "warnings": [...], "info": {...}}  - errors stop the run.
    """
    errors, warnings, info = [], [], {}
    for f in REQUIRED:
        if not mapping.get(f) or mapping[f] not in df.columns:
            errors.append(f"Please choose which column is the **{f}**.")
    chosen = [mapping.get(f) for f in FIELD_ORDER if mapping.get(f)]
    if len(chosen) != len(set(chosen)):
        errors.append("The same column is chosen for two different things. Each column can be used once.")
    if errors:
        return None, {"errors": errors, "warnings": warnings, "info": info}

    n0 = len(df)
    tx = pd.DataFrame(index=df.index)
    dates, note = _parse_dates(df[mapping["date"]])
    tx["date"] = pd.to_datetime(dates, errors="coerce").dt.normalize()
    tx["quantity"] = _to_number(df[mapping["quantity"]])
    tx["product"] = (df[mapping["product"]].astype(str).str.strip().replace({"": np.nan, "nan": np.nan})
                     if mapping.get("product") else "All products")
    tx["stock"] = _to_number(df[mapping["stock"]]) if mapping.get("stock") else np.nan
    tx["invoice"] = (df[mapping["invoice"]].astype(str).str.strip().replace({"": np.nan, "nan": np.nan})
                     if mapping.get("invoice") else np.nan)
    tx["price"] = _to_number(df[mapping["price"]]) if mapping.get("price") else np.nan
    tx["source_row"] = np.arange(2, n0 + 2)     # row number as seen in a spreadsheet (header = row 1)
    info["date_format"] = note

    bad_date = tx["date"].isna()
    bad_qty = tx["quantity"].isna()
    if bad_date.sum():
        warnings.append(f"{int(bad_date.sum()):,} rows had a date the app could not read and were skipped "
                        f"(for example spreadsheet row {int(tx.loc[bad_date, 'source_row'].iloc[0])}).")
    if (bad_qty & ~bad_date).sum():
        warnings.append(f"{int((bad_qty & ~bad_date).sum()):,} rows had a quantity that is not a number and were skipped.")
    tx = tx[~bad_date & ~bad_qty].copy()
    if tx.empty:
        return None, {"errors": ["No usable rows were left after reading dates and quantities. "
                                 "Check that you picked the right Date and Quantity columns."],
                      "warnings": warnings, "info": info}

    miss_prod = tx["product"].isna()
    if miss_prod.any():
        warnings.append(f"{int(miss_prod.sum()):,} rows had no product name and were grouped as 'Unknown'.")
        tx["product"] = tx["product"].fillna("Unknown")

    # flags used both for anomaly review and for keeping bad rows out of the demand numbers
    tx["negative_qty"] = tx["quantity"] < 0
    tx["zero_qty"] = tx["quantity"] == 0
    tx["duplicate_invoice"] = False
    if mapping.get("invoice"):
        has = tx["invoice"].notna()
        dup = tx[has].duplicated(subset=["invoice", "product", "date", "quantity"], keep="first")
        tx.loc[dup.index[dup], "duplicate_invoice"] = True
    tx["extreme_qty"] = False
    pos = tx[(tx["quantity"] > 0) & ~tx["duplicate_invoice"]]
    for prod, g in pos.groupby("product"):
        if len(g) < 20:
            continue
        med = g["quantity"].median()
        mad = (g["quantity"] - med).abs().median()
        limit = med + 8 * max(mad, 1.0)
        tx.loc[g.index[g["quantity"] > limit], "extreme_qty"] = True

    tx["used_for_demand"] = (tx["quantity"] > 0) & ~tx["duplicate_invoice"] & ~tx["extreme_qty"]
    info.update(rows_in_file=n0, rows_used=int(len(tx)), first_date=tx["date"].min(), last_date=tx["date"].max(),
                days_covered=int((tx["date"].max() - tx["date"].min()).days + 1),
                n_products=int(tx["product"].nunique()),
                has_stock=bool(mapping.get("stock")), has_invoice=bool(mapping.get("invoice")),
                has_price=bool(mapping.get("price")),
                n_negative=int(tx["negative_qty"].sum()), n_duplicate=int(tx["duplicate_invoice"].sum()),
                n_extreme=int(tx["extreme_qty"].sum()))
    if info["days_covered"] < MIN_DAYS_TOTAL:
        errors.append(f"Your data covers only {info['days_covered']} days. At least {MIN_DAYS_TOTAL} days "
                      f"(about 3 months) of daily history are needed for a trustworthy forecast.")
    elif info["days_covered"] < GOOD_HISTORY_DAYS:
        warnings.append(f"Your data covers {info['days_covered']} days. Forecasts work best with 6 months or more, "
                        "so treat them with extra care.")
    if info["n_negative"]:
        warnings.append(f"{info['n_negative']:,} rows have a negative quantity (returns?). They are kept out of "
                        "the demand numbers and listed under Suspicious bills.")
    if info["n_extreme"]:
        warnings.append(f"{info['n_extreme']:,} rows have an unusually large quantity for their product (a typing "
                        "mistake?). They are kept out of the demand numbers so they do not distort the forecast, "
                        "and are listed under Suspicious bills. Fix them in your file if they are real.")
    if info["n_duplicate"]:
        warnings.append(f"{info['n_duplicate']:,} rows look like duplicate bills (same invoice, product, date and "
                        "quantity). They are kept out of the demand numbers and listed under Suspicious bills.")
    return tx, {"errors": errors, "warnings": warnings, "info": info}


# ---------------------------------------------------------------- daily table + features
def build_daily(tx):
    """One row per product per day (days with no sales become 0)."""
    dem = tx[tx["used_for_demand"]]
    sales = dem.groupby(["product", "date"])["quantity"].sum().rename("sales")
    stock = None
    if tx["stock"].notna().any():
        stock = tx.dropna(subset=["stock"]).sort_values("date").groupby(["product", "date"])["stock"].last()
    start, end = tx["date"].min(), tx["date"].max()
    full_days = pd.date_range(start, end, freq="D")
    products = sorted(tx["product"].unique())
    idx = pd.MultiIndex.from_product([products, full_days], names=["product", "date"])
    daily = sales.reindex(idx).fillna(0.0).to_frame()
    daily["stock"] = stock.reindex(idx) if stock is not None else np.nan
    daily = daily.reset_index()
    if stock is not None:
        daily["stock"] = daily.groupby("product")["stock"].ffill()
    return daily


def _add_features(daily, use_stock):
    d = daily.sort_values(["product", "date"]).reset_index(drop=True).copy()
    g = d.groupby("product")["sales"]
    d["lag1"] = g.shift(1)
    d["lag7"] = g.shift(7)
    d["lag14"] = g.shift(14)
    base = g.shift(1)
    d["_base"] = base
    gb = d.groupby("product")["_base"]
    d["roll7"] = gb.transform(lambda x: x.rolling(7, min_periods=7).mean())
    d["roll14"] = gb.transform(lambda x: x.rolling(14, min_periods=7).mean())
    d["roll30"] = gb.transform(lambda x: x.rolling(30, min_periods=7).mean())
    d["roll7_std"] = gb.transform(lambda x: x.rolling(7, min_periods=7).std())
    d["trend"] = d["roll7"] / (d["roll30"] + 1e-6)
    d["dow"] = d["date"].dt.dayofweek
    d["is_weekend"] = (d["dow"] >= 5).astype(int)
    d["month"] = d["date"].dt.month
    d["dom"] = d["date"].dt.day
    feats = ["lag1", "lag7", "lag14", "roll7", "roll14", "roll30", "roll7_std", "trend", "dow", "is_weekend",
             "month", "dom"]
    if use_stock:
        d["stock_lag1"] = d.groupby("product")["stock"].shift(1).fillna(0.0)
        feats.append("stock_lag1")
    d = d.drop(columns="_base")
    valid = d[["lag1", "lag7", "lag14", "roll7", "roll14", "roll30"]].notna().all(axis=1)
    return d, feats, valid


def _wape(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    s = a.sum()
    return float(np.abs(a - f).sum() / s * 100) if s > 0 else float("nan")


def _model():
    return RandomForestRegressor(n_estimators=120, max_depth=10, min_samples_leaf=3,
                                 random_state=42, n_jobs=-1)


def rate_accuracy(model_wape, base_wape):
    if model_wape is None or (isinstance(model_wape, float) and math.isnan(model_wape)):
        return "Not enough data to judge"
    better = base_wape is not None and not math.isnan(base_wape) and model_wape < base_wape
    level = "Good" if model_wape <= 20 else "Fair" if model_wape <= 40 else "Weak"
    return f"{level}" + (" - beats the simple average" if better else " - NOT better than the simple average")


# ---------------------------------------------------------------- the whole run
def run_pipeline(tx, lead_time=7, pack_size=1, moq=0, safety_z=1.65, progress=None):
    """Train, backtest, forecast, recommend. `progress(text, fraction)` is optional."""
    def say(msg, frac):
        if progress:
            progress(msg, frac)

    warnings = []
    daily = build_daily(tx)
    use_stock = bool(daily["stock"].notna().mean() > 0.5)
    # keep products with enough days of actual selling history and cap the count
    vol = daily.groupby("product")["sales"].sum().sort_values(ascending=False)
    active_days = daily[daily["sales"] > 0].groupby("product")["date"].nunique()
    keep = [p for p in vol.index if active_days.get(p, 0) >= 10]
    dropped = [p for p in vol.index if p not in keep]
    if dropped:
        warnings.append(f"{len(dropped)} product(s) were left out of the forecast because they sold on fewer than "
                        f"10 different days: {', '.join(map(str, dropped[:8]))}{'...' if len(dropped) > 8 else ''}.")
    if len(keep) > MAX_PRODUCTS:
        warnings.append(f"Only the {MAX_PRODUCTS} best-selling products (of {len(keep)}) are forecast, to keep it fast.")
        dropped += keep[MAX_PRODUCTS:]
        keep = keep[:MAX_PRODUCTS]
    if not keep:
        raise ValueError("No product has enough sales days (at least 10) to build a forecast.")
    daily = daily[daily["product"].isin(keep)].copy()

    last_date = daily["date"].max()
    total_days = int((last_date - daily["date"].min()).days + 1)
    test_days = int(np.clip(0.2 * total_days, 30, 90))
    test_start = last_date - pd.Timedelta(days=test_days - 1)

    # one extra "tomorrow" row per product - its features only use the past, so it can be forecast
    tomorrow = pd.DataFrame({"product": keep, "date": last_date + pd.Timedelta(days=1)})
    tomorrow["sales"] = np.nan
    tomorrow["stock"] = np.nan
    ext = pd.concat([daily, tomorrow], ignore_index=True)
    ext["stock"] = ext.groupby("product")["stock"].ffill()
    feat, feats, valid = _add_features(ext, use_stock)
    is_future = feat["date"] > last_date

    rows, backtests, fc_rows = [], {}, []
    for i, h in enumerate(HORIZONS):
        say(f"Training the {h}-day model...", 0.1 + 0.8 * i / len(HORIZONS))
        target = feat.groupby("product")["sales"].transform(lambda x: x[::-1].rolling(h, min_periods=h).sum()[::-1])
        window_end = feat["date"] + pd.Timedelta(days=h - 1)
        tr = valid & ~is_future & (window_end < test_start) & target.notna()
        te = valid & ~is_future & (feat["date"] >= test_start) & (window_end <= last_date) & target.notna()
        full = valid & ~is_future & (window_end <= last_date) & target.notna()
        if tr.sum() < 60 or te.sum() < 20:
            rows.append({"horizon_days": h, "status": "skipped - not enough history for this horizon"})
            continue
        m = _model().fit(feat.loc[tr, feats], target[tr])
        pred = m.predict(feat.loc[te, feats])
        actual = target[te].values
        base = (feat.loc[te, "roll7"] * h).values
        bt = feat.loc[te, ["product", "date"]].copy()
        bt["actual"], bt["model"], bt["baseline"] = actual, pred, base
        backtests[h] = bt
        mw, bw = _wape(actual, pred), _wape(actual, base)
        rows.append({"horizon_days": h, "status": "ok", "train_rows": int(tr.sum()), "test_rows": int(te.sum()),
                     "model_WAPE_%": round(mw, 1), "baseline_WAPE_%": round(bw, 1),
                     "MAE": round(float(np.mean(np.abs(actual - pred))), 2),
                     "verdict": rate_accuracy(mw, bw)})
        # refit on everything known, then forecast the next h days
        m_full = _model().fit(feat.loc[full, feats], target[full])
        f = m_full.predict(feat.loc[is_future, feats])
        for p, v in zip(feat.loc[is_future, "product"], f):
            fc_rows.append({"product": p, "horizon_days": h, "forecast_units": float(max(v, 0.0))})
    horizons = pd.DataFrame(rows)
    forecasts = pd.DataFrame(fc_rows)
    if forecasts.empty:
        raise ValueError("The data is too short or too sparse to train any forecast model.")
    say("Building order advice...", 0.93)

    recs = None
    if use_stock:
        recs = _recommend(daily, forecasts, lead_time, pack_size, moq, safety_z)
    else:
        warnings.append("No stock column was chosen, so order advice and stockout risk are not shown. "
                        "Add a stock column to unlock them.")
    anomalies = _anomaly_table(tx)
    say("Done", 1.0)
    return {"daily": daily, "horizons": horizons, "backtests": backtests, "forecasts": forecasts,
            "recommendations": recs, "anomalies": anomalies, "warnings": warnings, "dropped": dropped,
            "used_stock": use_stock, "last_date": last_date, "test_start": test_start,
            "total_days": total_days, "products": keep}


def _recommend(daily, forecasts, lead_time, pack_size, moq, z):
    last = daily["date"].max()
    out = []
    for p, g in daily.groupby("product"):
        f = forecasts[forecasts["product"] == p].set_index("horizon_days")["forecast_units"]
        if 7 in f.index:
            rate = f[7] / 7.0
        elif 14 in f.index:
            rate = f[14] / 14.0
        elif 1 in f.index:
            rate = float(f[1])
        else:
            rate = float(g["sales"].tail(30).mean())
        recent = g["sales"].tail(60)
        sd = float(recent.std()) if len(recent) > 1 else 0.0
        stock_now = g.sort_values("date")["stock"].iloc[-1]
        stock_now = 0.0 if pd.isna(stock_now) else float(stock_now)
        safety = z * sd * math.sqrt(max(lead_time, 1))
        need = rate * lead_time + safety - stock_now
        qty = 0 if need <= 0 else max(int(math.ceil(need / pack_size) * pack_size), int(moq))
        cover = stock_now / rate if rate > 1e-9 else float("inf")
        if cover <= lead_time:
            risk = "Critical"
        elif cover <= 1.5 * lead_time:
            risk = "High"
        elif cover <= 2 * lead_time:
            risk = "Medium"
        else:
            risk = "Low"
        priority = "No Action" if qty == 0 else risk
        if qty == 0:
            advice = (f"No order needed. About {cover:.0f} days of stock left." if np.isfinite(cover)
                      else "No order needed. This product is not selling at the moment.")
        else:
            if not np.isfinite(cover):
                lead_txt = "your stock is low"
            elif cover <= lead_time:
                lead_txt = f"you have only {cover:.1f} days of stock left, and a delivery takes {lead_time} days, so you may run out before it arrives"
            else:
                lead_txt = f"you have {cover:.1f} days of stock left and a delivery takes {lead_time} days, so a top-up now keeps you safe"
            advice = f"Order about {qty:,} units. {lead_txt[0].upper() + lead_txt[1:]}."
        out.append({"product": p, "current_stock": stock_now, "expected_daily_sales": round(rate, 2),
                    "days_of_cover": round(cover, 1) if np.isfinite(cover) else np.nan,
                    "lead_time_days": lead_time, "safety_stock": round(safety, 1),
                    "recommended_order_qty": qty, "risk": risk, "priority": priority, "advice": advice})
    rec = pd.DataFrame(out)
    rank = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3, "No Action": 4}
    rec["_r"] = rec["priority"].map(rank)
    rec = rec.sort_values(["_r", "recommended_order_qty"], ascending=[True, False]).drop(columns="_r")
    return rec.reset_index(drop=True)


def _anomaly_table(tx):
    reasons = {"negative_qty": "Negative quantity (return or typing mistake?)",
               "duplicate_invoice": "Duplicate bill (same invoice, product, date and quantity)",
               "extreme_qty": "Unusually large quantity for this product"}
    parts = []
    for col, text in reasons.items():
        sub = tx[tx[col]]
        if len(sub):
            s = sub[["source_row", "date", "product", "quantity", "invoice"]].copy()
            s["reason"] = text
            parts.append(s)
    if not parts:
        return pd.DataFrame(columns=["source_row", "date", "product", "quantity", "invoice", "reason"])
    a = pd.concat(parts).sort_values(["source_row", "reason"]).reset_index(drop=True)
    a["date"] = a["date"].dt.strftime("%Y-%m-%d")
    return a


def template_csv():
    """A tiny example file people can copy."""
    rows = ["Date,Product,Quantity Sold,Stock,Invoice No"]
    d0 = pd.Timestamp("2026-01-01")
    for i in range(6):
        d = (d0 + pd.Timedelta(days=i // 2)).strftime("%Y-%m-%d")
        prod = ["Rice 5kg", "Cooking Oil 1L"][i % 2]
        rows.append(f"{d},{prod},{3 + i},{40 - i},INV-{1000 + i}")
    return "\n".join(rows) + "\n"
