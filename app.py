import sqlite3
from datetime import datetime

import pandas as pd
import streamlit as st

import store_ui

DB_PATH = "supplypulse.db"
CSV_PATH = "week8_replenishment_recommendations.csv"

st.set_page_config(page_title="SupplyPulse AI - Control Tower", page_icon="📦", layout="wide")

PRIORITY_COLORS = {"Critical": "#C0392B", "High": "#E67E22", "Medium": "#B7950B", "Low": "#1E8449"}
PRIORITY_RANK = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}

st.markdown(
    """
    <style>
    .block-container {padding-top: 4rem; max-width: 1250px;}
    .sp-header {background: linear-gradient(90deg, #1F4E79 0%, #2E75B6 100%);
                padding: 1.1rem 1.5rem; border-radius: 12px; margin-bottom: 1.2rem;}
    .sp-header h1 {color: #ffffff; margin: 0; font-size: 1.7rem;}
    .sp-header p {color: #DCE9F5; margin: 0.2rem 0 0 0; font-size: 0.95rem;}
    .sp-chip {display: inline-block; color: #ffffff; padding: 2px 12px; border-radius: 999px;
              font-weight: 600; font-size: 0.8rem; letter-spacing: 0.3px;}
    .sp-sku {font-size: 1.25rem; font-weight: 700; margin: 0.3rem 0 0 0;}
    .sp-sub {color: #6b7280; font-size: 0.85rem;}
    .sp-reason {font-size: 0.85rem; color: #374151; line-height: 1.5;}
    div[data-testid="stMetric"] {background: #F3F7FB; border: 1px solid #D6E4F0;
                                 padding: 0.6rem 0.9rem; border-radius: 10px;}
    div[data-testid="stMetricLabel"] p {white-space: normal !important; overflow: visible !important; text-overflow: clip !important;}
    div[data-testid="stButton"] button {white-space: nowrap; padding: 0.25rem 0.4rem; font-size: 0.85rem; width: 100%;}
    div[data-testid="stButton"] button p {overflow: visible !important; text-overflow: clip !important;}
    div[data-testid="stButton"] button {padding: 0.3rem 0.5rem; white-space: nowrap;}
    div[data-testid="stButton"] button p {font-size: 0.85rem; overflow: visible; text-overflow: clip;}
    div[data-testid="stHorizontalBlock"] {flex-wrap: wrap; row-gap: 0.4rem;}
    div[data-testid="stColumn"] {min-width: 130px;}
    </style>
    <div class="sp-header">
      <h1>📦 SupplyPulse AI — Control Tower</h1>
      <p>Review the AI's replenishment recommendations. Approve, modify or reject each one — every decision is logged.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------- data layer ----------
def get_conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS approval_audit_log (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            distributor_id TEXT, sku_id TEXT, date TEXT,
            recommended_qty REAL, action TEXT, final_qty REAL,
            reviewer TEXT, note TEXT, decided_at TEXT
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS purchase_orders (
            po_id INTEGER PRIMARY KEY AUTOINCREMENT,
            distributor_id TEXT, sku_id TEXT, date TEXT,
            order_qty REAL, created_at TEXT
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS decision_reversals (
            reversal_id INTEGER PRIMARY KEY AUTOINCREMENT,
            log_id INTEGER, reversed_by TEXT, reason TEXT, reversed_at TEXT
        )""")
    try:  # older databases: add a status column to purchase_orders
        conn.execute("ALTER TABLE purchase_orders ADD COLUMN status TEXT DEFAULT 'active'")
    except sqlite3.OperationalError:
        pass
    conn.execute("UPDATE purchase_orders SET status='active' WHERE status IS NULL")
    conn.commit()
    conn.close()


@st.cache_data
def load_recommendations():
    d = pd.read_csv(CSV_PATH)
    d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    return d


def read_table(name):
    conn = get_conn()
    out = pd.read_sql(f"SELECT * FROM {name}", conn)
    conn.close()
    return out


def review_recommendation(row, action, reviewer, modified_qty=None, note=""):
    """Same logic as Week 11: approve -> recommended qty, modify -> new qty, reject -> 0."""
    action = action.lower()
    if action == "approve":
        final_qty = float(row["final_order_qty"])
    elif action == "modify":
        final_qty = float(modified_qty)
    elif action == "reject":
        final_qty = 0.0
    else:
        raise ValueError(f"Unknown action: {action}")

    now = datetime.now().isoformat(timespec="seconds")
    conn = get_conn()
    conn.execute(
        "INSERT INTO approval_audit_log "
        "(distributor_id, sku_id, date, recommended_qty, action, final_qty, reviewer, note, decided_at) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (row["distributor_id"], row["sku_id"], str(row["date"]), float(row["final_order_qty"]),
         action, final_qty, reviewer, note, now),
    )
    if final_qty > 0:
        conn.execute(
            "INSERT INTO purchase_orders (distributor_id, sku_id, date, order_qty, created_at) VALUES (?,?,?,?,?)",
            (row["distributor_id"], row["sku_id"], str(row["date"]), final_qty, now),
        )
    conn.commit()
    conn.close()
    return final_qty


def reverse_decision(log_id, reviewer, reason=""):
    """Undo a decision: cancel its purchase order and put the item back in the queue.
    Nothing is deleted - the reversal is recorded so the audit trail stays complete."""
    now = datetime.now().isoformat(timespec="seconds")
    conn = get_conn()
    already = conn.execute("SELECT COUNT(*) FROM decision_reversals WHERE log_id=?", (int(log_id),)).fetchone()[0]
    if already:
        conn.close()
        return False
    row = conn.execute(
        "SELECT distributor_id, sku_id, date, decided_at FROM approval_audit_log WHERE log_id=?",
        (int(log_id),)).fetchone()
    conn.execute("INSERT INTO decision_reversals (log_id, reversed_by, reason, reversed_at) VALUES (?,?,?,?)",
                 (int(log_id), reviewer, reason, now))
    conn.execute(
        "UPDATE purchase_orders SET status='cancelled' "
        "WHERE distributor_id=? AND sku_id=? AND date=? AND created_at=? AND status='active'", tuple(row))
    conn.commit()
    conn.close()
    return True


init_db()
df = load_recommendations()
audit = read_table("approval_audit_log")
reversals = read_table("decision_reversals")
audit["reversed"] = audit["log_id"].isin(reversals["log_id"]).map({True: "Yes", False: "No"})
active = audit[audit["reversed"] == "No"]
decided = set(zip(active["distributor_id"], active["sku_id"], active["date"].astype(str)))

if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"), icon="✅")

# ---------- sidebar filters ----------
with st.sidebar:
    st.header("Approval queue filters")
    reviewer = st.text_input("Reviewer name", value="Neha")
    distributors = sorted(df["distributor_id"].unique())
    dist = st.selectbox("Distributor", distributors)
    actionable = df[(df["final_order_qty"] > 0) & (df["distributor_id"] == dist)]
    dates = sorted(actionable["date"].unique())
    day = st.selectbox("Date", dates, index=0)
    prios = st.multiselect("Priority", list(PRIORITY_RANK), default=list(PRIORITY_RANK))
    max_cards = st.slider("Max recommendations shown", 5, 30, 10)
    st.caption("Recommendations with no order needed ('No Action') are never shown.")

queue = actionable[(actionable["date"] == day) & (actionable["priority"].isin(prios))].copy()
queue = queue[[(d, s, dt) not in decided for d, s, dt in zip(queue["distributor_id"], queue["sku_id"], queue["date"])]]
queue["_rank"] = queue["priority"].map(PRIORITY_RANK)
queue = queue.sort_values(["_rank", "final_order_qty"], ascending=[True, False])

po = read_table("purchase_orders")

tab_queue, tab_audit, tab_po, tab_rev, tab_store = st.tabs(["📋 Approval queue", "🧾 Audit trail", "📦 Purchase orders", "↩ Reverse a decision", "🏪 Use your own data"])

with tab_queue:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Pending recommendations", len(queue))
    k2.metric("Critical pending", int((queue["priority"] == "Critical").sum()))
    k3.metric("Units pending", f"{int(queue['final_order_qty'].sum()):,}")
    k4.metric("Decisions logged (all time)", len(audit))
    st.write("")

    if queue.empty:
        st.info("Nothing pending for this distributor, date and priority selection. Try another date or priority.")

    for _, row in queue.head(max_cards).iterrows():
        key = f"{row['distributor_id']}_{row['sku_id']}_{row['date']}"
        color = PRIORITY_COLORS.get(row["priority"], "#555555")
        with st.container(border=True):
            top = st.columns([1.2, 1, 1, 1])
            with top[0]:
                st.markdown(f"<span class='sp-chip' style='background:{color}'>{row['priority']}</span>",
                            unsafe_allow_html=True)
                st.markdown(f"<p class='sp-sku'>{row['sku_id']}</p>", unsafe_allow_html=True)
                st.markdown(f"<span class='sp-sub'>{row['distributor_id']} · {row['date']}</span>",
                            unsafe_allow_html=True)
            top[1].metric("Order qty", f"{int(row['final_order_qty']):,}")
            top[2].metric("Days of cover", f"{row['days_of_cover']:.1f}")
            top[3].metric("Lead time", f"{int(row['lead_time_days'])} d")
            with st.expander("Why this recommendation?"):
                for reason in str(row["reason_codes"]).split("; "):
                    st.markdown(f"- {reason}")
            c1, c2, c3, c4, c5 = st.columns([1.3, 2.2, 1, 1, 1], vertical_alignment="bottom")
            new_qty = c1.number_input("Quantity (if you modify)", min_value=0,
                                      value=int(row["final_order_qty"]), step=1, key=f"qty_{key}")
            note = c2.text_input("Note (optional)", key=f"note_{key}")
            if c3.button("✅ Approve", key=f"ap_{key}"):
                q = review_recommendation(row, "approve", reviewer, note=note)
                st.session_state["flash"] = f"Approved {row['sku_id']}: purchase order for {int(q):,} units created."
                st.rerun()
            if c4.button("✏️ Modify", key=f"mo_{key}"):
                if new_qty <= 0:
                    st.warning("Enter a quantity above 0, or use Reject.")
                else:
                    q = review_recommendation(row, "modify", reviewer, modified_qty=new_qty, note=note)
                    st.session_state["flash"] = f"Modified {row['sku_id']}: purchase order for {int(q):,} units created."
                    st.rerun()
            if c5.button("❌ Reject", key=f"re_{key}"):
                review_recommendation(row, "reject", reviewer, note=note)
                st.session_state["flash"] = f"Rejected {row['sku_id']}: no purchase order created."
                st.rerun()

    if len(queue) > max_cards:
        st.caption(f"Showing {max_cards} of {len(queue)} pending. Increase the slider in the sidebar to see more.")

with tab_audit:
    st.subheader("Every decision, newest first")
    if audit.empty:
        st.info("No decisions yet.")
    else:
        st.dataframe(audit.sort_values("log_id", ascending=False), hide_index=True)

with tab_po:
    st.subheader("Purchase orders generated")
    if po.empty:
        st.info("No purchase orders yet.")
    else:
        st.dataframe(po.sort_values("po_id", ascending=False), hide_index=True)

with tab_rev:
    st.subheader("Made a mistake? Reverse a decision")
    st.caption("The purchase order is cancelled and the item returns to the Approval queue. "
               "The original decision stays in the Audit trail and the reversal is recorded.")
    if active.empty:
        st.info("There are no active decisions to reverse.")
    else:
        opts = {
            f"#{r.log_id} · {r.sku_id} · {r.distributor_id} · {r.date} · {r.action} · final qty {int(r.final_qty)}": int(r.log_id)
            for r in active.sort_values("log_id", ascending=False).itertuples()
        }
        pick = st.selectbox("Choose the decision to reverse", list(opts))
        why = st.text_input("Reason for reversing (optional)", key="rev_reason")
        if st.button("↩ Reverse this decision", key="rev_btn"):
            if reverse_decision(opts[pick], reviewer, why):
                st.session_state["flash"] = "Decision reversed: purchase order cancelled and the item is back in the Approval queue."
                st.rerun()
            else:
                st.warning("This decision was already reversed.")

with tab_store:
    store_ui.render()
