"""Streamlit page: 'Use your own data' - upload any store's sales file and get forecasts + advice."""
import altair as alt
import pandas as pd
import streamlit as st

import store_pipeline as sp

NONE_LABEL = "— not in my file —"
PRIORITY_COLORS = {"Critical": "#C0392B", "High": "#E67E22", "Medium": "#B7950B", "Low": "#1E8449",
                   "No Action": "#7F8C8D"}
PRIORITY_ORDER = ["Critical", "High", "Medium", "Low", "No Action"]
FIELD_LABELS = {
    "date": ("Date *", "The day of each sale"),
    "quantity": ("Quantity sold *", "How many units were sold"),
    "product": ("Product / item", "Optional. Leave as 'not in my file' if the file is for one product only"),
    "stock": ("Stock on hand", "Optional. Needed for order advice and stockout risk"),
    "invoice": ("Invoice / bill number", "Optional. Used to catch duplicate bills"),
    "price": ("Price or amount", "Optional"),
}


def _reset(keep_file=False):
    keys = ["st_res", "st_tx", "st_report", "st_map_sig"]
    if not keep_file:
        keys += ["st_df", "st_fname", "st_step"]
    keys += [k for k in list(st.session_state.keys()) if k.startswith("st_f_")]   # filters tied to old results
    for k in keys:
        st.session_state.pop(k, None)
    if not keep_file:
        st.session_state["st_step"] = 1


# ---------------------------------------------------------------- step 1: upload
def _step_upload():
    st.markdown("#### Step 1 of 3 · Upload your store's sales file")
    st.write("Upload a **CSV or Excel** file with one row per sale (or per day). SupplyPulse AI learns from "
             "**your** data, so it works for any shop.")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(
            "**What the file needs**\n"
            "- **Date** and **Quantity sold** (required)\n"
            "- **Product / item name** (optional - if missing, the whole store is treated as one product)\n"
            "- **Stock on hand** (optional - needed for order advice)\n"
            "- **Invoice / bill number** (optional - to catch duplicate bills)\n"
            "- At least **3 months** of daily history (6 months or more is better)\n\n"
            "Column names do not matter. You will match them in the next step."
        )
    with c2:
        st.download_button("⬇ Download an example file", sp.template_csv(), file_name="example_sales_file.csv",
                           mime="text/csv")
        st.caption("Open it to see the layout. Your file can have more columns, such as Month or Price. "
                   "Extra columns are ignored.")
    up = st.file_uploader("Choose your file", type=["csv", "xlsx", "xls"], key="st_uploader")
    if up is not None:
        try:
            df = sp.read_table(up, up.name)
        except Exception as e:  # noqa: BLE001
            st.error(f"Sorry, that file could not be read ({e}). Please upload a .csv or .xlsx file.")
            return
        if df.empty or len(df.columns) < 2:
            st.error("The file looks empty or has fewer than two columns. Please check it and upload again.")
            return
        st.success(f"Read **{up.name}**: {len(df):,} rows and {len(df.columns)} columns.")
        st.dataframe(df.head(8), hide_index=True)
        if st.button("Next: match my columns →", type="primary", key="st_next1"):
            st.session_state["st_df"], st.session_state["st_fname"] = df, up.name
            st.session_state["st_step"] = 2
            st.rerun()


# ---------------------------------------------------------------- step 2: match columns
def _step_match():
    df = st.session_state["st_df"]
    st.markdown("#### Step 2 of 3 · Tell the app what your columns mean")
    st.caption(f"File: **{st.session_state['st_fname']}** · {len(df):,} rows. The app guessed the columns below. "
               "Change any that are wrong.")
    guess = sp.guess_columns(df)
    options = [NONE_LABEL] + list(df.columns)
    mapping = {}
    cols = st.columns(3)
    for i, field in enumerate(sp.FIELD_ORDER):
        label, helptext = FIELD_LABELS[field]
        default = guess.get(field)
        idx = options.index(default) if default in options else 0
        choice = cols[i % 3].selectbox(label, options, index=idx, help=helptext, key=f"st_map_{field}")
        mapping[field] = None if choice == NONE_LABEL else choice

    s1, s2, s3 = st.columns(3)
    lead = s1.number_input("Supplier delivery time (days)", 1, 90, 7, key="st_lead",
                           help="How many days between placing an order and receiving it")
    pack = s2.number_input("Order in packs of", 1, 10000, 1, key="st_pack", help="1 = any quantity")
    moq = s3.number_input("Minimum order quantity", 0, 100000, 0, key="st_moq")

    tx, rep = sp.clean_data(df, mapping)
    info = rep["info"]
    if rep["errors"]:
        for e in rep["errors"]:
            st.error(e)
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Rows used", f"{info['rows_used']:,}")
        m2.metric("Products", f"{info['n_products']:,}")
        m3.metric("Days of history", f"{info['days_covered']:,}")
        st.caption(f"Your data runs from **{info['first_date']:%d %b %Y}** to **{info['last_date']:%d %b %Y}**. "
                   f"Dates were read as: {info['date_format']}.")
    if rep["warnings"]:
        with st.expander(f"⚠ Things I noticed in your data ({len(rep['warnings'])})", expanded=not rep["errors"]):
            for w in rep["warnings"]:
                st.warning(w)

    b1, b2 = st.columns([1, 1])
    if b1.button("← Back: choose another file", key="st_back2"):
        _reset()
        st.rerun()
    if b2.button("▶ Run forecast", type="primary", disabled=bool(rep["errors"]), key="st_run"):
        bar = st.progress(0.0, text="Starting...")
        try:
            res = sp.run_pipeline(tx, lead_time=int(lead), pack_size=int(pack), moq=int(moq),
                                  progress=lambda t, f: bar.progress(min(f, 1.0), text=t))
        except ValueError as e:
            bar.empty()
            st.error(str(e))
            return
        st.session_state.update(st_res=res, st_tx=tx, st_report=rep, st_step=3)
        st.rerun()


# ---------------------------------------------------------------- step 3: results
def _chip(text):
    color = PRIORITY_COLORS.get(text, "#555")
    return f"<span class='sp-chip' style='background:{color}'>{text}</span>"


def _csv(df):
    return df.to_csv(index=False).encode("utf-8")


def _step_results():
    res, tx, rep = st.session_state["st_res"], st.session_state["st_tx"], st.session_state["st_report"]
    info = rep["info"]
    st.markdown("#### Step 3 of 3 · Your results")
    nav1, nav2, _ = st.columns([1.3, 1, 3])
    if nav1.button("← Back: change column matching", key="st_back3"):
        _reset(keep_file=True)
        st.session_state["st_step"] = 2
        st.rerun()
    if nav2.button("⟲ Start over", key="st_over"):
        _reset()
        st.rerun()
    for w in res["warnings"]:
        st.info(w)

    recs = res["recommendations"]
    products = res["products"]
    n_crit = int((recs["priority"] == "Critical").sum()) if recs is not None else None
    sm = st.columns(4)
    sm[0].metric("Products forecast", len(products))
    sm[1].metric("Days of history", f"{res['total_days']:,}")
    sm[2].metric("Need ordering now (Critical)", n_crit if n_crit is not None else "n/a")
    sm[3].metric("Suspicious bills", len(res["anomalies"]))

    # ---- filters
    with st.container(border=True):
        st.markdown("**Filters**")
        f1, f2, f3 = st.columns([2.2, 1.6, 1])
        sel_products = f1.multiselect("Products", products, default=products, key="st_f_prod",
                                      placeholder="All products")
        sel_products = sel_products or products
        if recs is not None:
            sel_risk = f2.multiselect("Priority", PRIORITY_ORDER, default=PRIORITY_ORDER, key="st_f_risk")
            sel_risk = sel_risk or PRIORITY_ORDER
        else:
            sel_risk = PRIORITY_ORDER
        window = f3.selectbox("History shown", ["Last 30 days", "Last 60 days", "Last 90 days",
                                                "Last 180 days", "All"], index=2, key="st_f_win")
    win_days = None if window == "All" else int(window.split()[1])

    t_do, t_fc, t_rel, t_bad, t_dl = st.tabs(["✅ What should I do?", "📈 Forecast", "🎯 How reliable is it?",
                                              "🚨 Suspicious bills", "⬇ Download"])

    # ---- what to do
    with t_do:
        if recs is None:
            st.info("Order advice needs a **stock** column. Go back, match your stock column, and run again. "
                    "The forecasts are still available in the next tabs.")
        else:
            view = recs[recs["product"].isin(sel_products) & recs["priority"].isin(sel_risk)]
            st.caption("Products are sorted from most urgent to least urgent. "
                       "**Days of cover** = how many days your current stock will last at the expected sales speed.")
            cnt = view["priority"].value_counts().reindex(PRIORITY_ORDER).fillna(0).reset_index()
            cnt.columns = ["priority", "products"]
            ch = alt.Chart(cnt).mark_bar().encode(
                x=alt.X("priority:N", sort=PRIORITY_ORDER, title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("products:Q", title="Number of products"),
                color=alt.Color("priority:N", scale=alt.Scale(domain=list(PRIORITY_COLORS),
                                                              range=list(PRIORITY_COLORS.values())), legend=None),
                tooltip=["priority", "products"]).properties(height=200)
            st.altair_chart(ch)
            todo = view[view["priority"] != "No Action"]
            if todo.empty:
                st.success("Nothing needs ordering for the selected products. 🎉")
            for _, r in todo.head(15).iterrows():
                with st.container(border=True):
                    a, b, c, d = st.columns([1.5, 3, 1, 1])
                    a.markdown(_chip(r["priority"]), unsafe_allow_html=True)
                    a.markdown(f"<p class='sp-sku'>{r['product']}</p>", unsafe_allow_html=True)
                    b.markdown(f"**What to do:** {r['advice']}")
                    c.metric("Order qty", f"{int(r['recommended_order_qty']):,}")
                    cover = r["days_of_cover"]
                    d.metric("Days of cover", "n/a" if pd.isna(cover) else f"{cover:.1f}")
            if len(todo) > 15:
                st.caption(f"Showing the 15 most urgent of {len(todo)}. The full list is in the table below.")
            with st.expander("Full table"):
                st.dataframe(view, hide_index=True)

    # ---- forecast
    with t_fc:
        hs = [h for h in sp.HORIZONS if h in res["forecasts"]["horizon_days"].values]
        c1, c2 = st.columns([1, 2])
        h = c1.selectbox("Look ahead", hs, index=min(1, len(hs) - 1), format_func=lambda x: f"Next {x} day(s)",
                         key="st_f_h")
        one = c2.selectbox("Show", ["All selected products together"] + list(sel_products), key="st_f_one")
        plist = list(sel_products) if one.startswith("All") else [one]
        daily = res["daily"]
        d = daily[daily["product"].isin(plist)].groupby("date")["sales"].sum().reset_index()
        last = res["last_date"]
        if win_days:
            d = d[d["date"] > last - pd.Timedelta(days=win_days)]
        f = res["forecasts"]
        total_f = f[f["product"].isin(plist) & (f["horizon_days"] == h)]["forecast_units"].sum()
        fdf = pd.DataFrame({"date": pd.date_range(last + pd.Timedelta(days=1), periods=h), "sales": total_f / h})
        d = d.assign(series="What you sold")
        fdf = fdf.assign(series="Expected per day (forecast)")
        base = alt.Chart(pd.concat([d, fdf])).encode(
            x=alt.X("date:T", title=None), y=alt.Y("sales:Q", title="Units per day"),
            color=alt.Color("series:N", title=None,
                            scale=alt.Scale(domain=["What you sold", "Expected per day (forecast)"],
                                            range=["#1F4E79", "#E67E22"])),
            tooltip=["series", alt.Tooltip("date:T"), alt.Tooltip("sales:Q", format=",.1f")])
        line = base.transform_filter(alt.datum.series == "What you sold").mark_line()
        proj = base.transform_filter(alt.datum.series != "What you sold").mark_line(strokeDash=[6, 4], point=True)
        rule = alt.Chart(pd.DataFrame({"date": [last]})).mark_rule(color="#999", strokeDash=[2, 2]).encode(x="date:T")
        st.altair_chart((line + proj + rule).properties(height=320))
        st.metric(f"Expected total sales, next {h} day(s)", f"{total_f:,.0f} units")
        st.caption("The orange dashed line is the **average expected per day**. The dotted grey line marks "
                   "the last day of your data.")
        table = (f[f["product"].isin(sel_products)]
                 .pivot(index="product", columns="horizon_days", values="forecast_units").round(0)
                 .rename(columns=lambda x: f"Next {x} day(s)").reset_index())
        st.markdown("**Expected units to be sold, by product**")
        st.dataframe(table, hide_index=True)

    # ---- reliability
    with t_rel:
        hz = res["horizons"]
        ok = hz[hz["status"] == "ok"].copy()
        st.write(f"To check itself, the app **hid the last {int((res['last_date'] - res['test_start']).days + 1)} days "
                 "of your data**, forecast them, and compared the forecast with what really sold. "
                 "That makes the scores below an honest test, not a guess.")
        if ok.empty:
            st.warning("There was not enough history to test any forecast.")
        else:
            long = pd.concat([
                ok[["horizon_days", "model_WAPE_%"]].rename(columns={"model_WAPE_%": "error"}).assign(who="Our forecast"),
                ok[["horizon_days", "baseline_WAPE_%"]].rename(columns={"baseline_WAPE_%": "error"}).assign(
                    who="Simple average (last 7 days)")])
            long["Look ahead"] = long["horizon_days"].map(lambda x: f"{x} day(s)")
            ch = alt.Chart(long).mark_bar().encode(
                x=alt.X("Look ahead:N", sort=[f"{x} day(s)" for x in sp.HORIZONS], title=None, axis=alt.Axis(labelAngle=0)),
                xOffset="who:N", y=alt.Y("error:Q", title="Average error % (lower is better)"),
                color=alt.Color("who:N", title=None, scale=alt.Scale(
                    domain=["Our forecast", "Simple average (last 7 days)"], range=["#1F4E79", "#B0B7C3"])),
                tooltip=["who", "Look ahead", alt.Tooltip("error:Q", format=".1f")]).properties(height=280)
            st.altair_chart(ch)
            for _, r in ok.iterrows():
                st.markdown(f"- **{int(r['horizon_days'])}-day forecast:** typically about "
                            f"**{r['model_WAPE_%']:.0f}%** away from what really sold. {r['verdict']}.")
            st.caption("A 10% error means the forecast was, on average, 10% away from the real sales. Forecasts "
                       "for a single day are always less exact than forecasts for a week or a month, because "
                       "daily sales jump around.")
            skipped = hz[hz["status"] != "ok"]
            for _, r in skipped.iterrows():
                st.caption(f"{int(r['horizon_days'])}-day forecast: {r['status']}.")
            bh = st.selectbox("See forecast vs reality for", list(ok["horizon_days"]),
                              format_func=lambda x: f"{x}-day forecast", key="st_f_bt")
            bt = res["backtests"][bh]
            bt = bt[bt["product"].isin(sel_products)].groupby("date")[["actual", "model", "baseline"]].sum().reset_index()
            bt = bt.rename(columns={"actual": "What really sold", "model": "Our forecast",
                                    "baseline": "Simple average"})
            ml = bt.melt("date", var_name="series", value_name="units")
            ch2 = alt.Chart(ml).mark_line().encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("units:Q", title=f"Units over the {bh} day(s) starting that day"),
                color=alt.Color("series:N", title=None, scale=alt.Scale(
                    domain=["What really sold", "Our forecast", "Simple average"],
                    range=["#1F4E79", "#E67E22", "#B0B7C3"])),
                tooltip=["series", "date:T", alt.Tooltip("units:Q", format=",.0f")]).properties(height=280)
            st.altair_chart(ch2)

    # ---- suspicious
    with t_bad:
        an = res["anomalies"]
        st.caption("These rows look wrong or unusual. Check them in your original file. Rows that are clearly "
                   "wrong are kept out of the forecast so they do not mislead it. **Row** is the row number you "
                   "see in Excel (the header is row 1).")
        if an.empty:
            st.success("No suspicious bills found. 🎉")
        else:
            reasons = sorted(an["reason"].unique())
            pick = st.multiselect("Show these kinds", reasons, default=reasons, key="st_f_reason")
            shown = an[an["reason"].isin(pick or reasons) & an["product"].isin(sel_products)]
            st.dataframe(shown.rename(columns={"source_row": "Row"}), hide_index=True)
            st.caption(f"{len(shown):,} suspicious rows shown.")

    # ---- download
    with t_dl:
        st.write("Save your results as spreadsheets (CSV files).")
        d1, d2, d3 = st.columns(3)
        d1.download_button("⬇ Forecasts", _csv(res["forecasts"]), "forecasts.csv", "text/csv", key="st_dl1")
        if recs is not None:
            d2.download_button("⬇ Order advice", _csv(recs), "order_advice.csv", "text/csv", key="st_dl2")
        d3.download_button("⬇ Suspicious bills", _csv(res["anomalies"]), "suspicious_bills.csv", "text/csv",
                           key="st_dl3")


# ---------------------------------------------------------------- entry point
def render():
    st.session_state.setdefault("st_step", 1)
    step = st.session_state["st_step"]
    if step == 3 and "st_res" not in st.session_state:
        step = st.session_state["st_step"] = 2 if "st_df" in st.session_state else 1
    if step == 2 and "st_df" not in st.session_state:
        step = st.session_state["st_step"] = 1
    st.subheader("🏪 Use your own store's data")
    if step == 1:
        _step_upload()
    elif step == 2:
        _step_match()
    else:
        _step_results()
