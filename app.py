"""
FPL Tracker – Streamlit App (single file)

Quickstart
----------
1) Create venv & install deps:
   pip install streamlit requests pandas matplotlib
2) Run the app:
   streamlit run app.py
"""

import os
import time
import sqlite3
from typing import Dict, List, Tuple
import io

import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"

API_BASE = "https://fantasy.premierleague.com/api"
HEADERS = {"User-Agent": "FPL-Tracker/1.0"}
DB_PATH = "fpl_tracker.db"

# -------------------------------------------------------------------
# DB Setup
# -------------------------------------------------------------------
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS teams (
    entry_id INTEGER PRIMARY KEY,
    player_name TEXT,
    team_name TEXT,
    league_id INTEGER,
    last_updated INTEGER
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    name TEXT,
    deadline_time TEXT
);
CREATE TABLE IF NOT EXISTS team_event_stats (
    entry_id INTEGER,
    event INTEGER,
    points INTEGER,
    total_points INTEGER,
    rank INTEGER,
    bank INTEGER,
    value INTEGER,
    transfers INTEGER,
    transfers_cost INTEGER,
    points_on_bench INTEGER,
    captain_element INTEGER,
    captain_points INTEGER,
    captain_base_points INTEGER,
    chip_used INTEGER,
    active_chip TEXT,
    goals_starting_xi INTEGER,
    transfer_efficiency INTEGER,
    created_at INTEGER,
    PRIMARY KEY (entry_id, event)
);
"""

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA_SQL)

# -------------------------------------------------------------------
# API Helpers
# -------------------------------------------------------------------
def get_json(url: str) -> dict:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_bootstrap(): return get_json(f"{API_BASE}/bootstrap-static/")
def fetch_entry_history(entry): return get_json(f"{API_BASE}/entry/{entry}/history/")
def fetch_transfers(entry): return get_json(f"{API_BASE}/entry/{entry}/transfers/")
def fetch_picks(entry, event): return get_json(f"{API_BASE}/entry/{entry}/event/{event}/picks/")
def fetch_event_live(event): return get_json(f"{API_BASE}/event/{event}/live/")

def fetch_league_standings(league_id: int, page: int = 1) -> dict:
    for kind in ("leagues-classic", "leagues-h2h"):
        url = f"{API_BASE}/{kind}/{league_id}/standings/?page_standings={page}"
        try:
            return get_json(url)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                continue
            raise
    raise ValueError("League not found as Classic or H2H")

# -------------------------------------------------------------------
# Computations
# -------------------------------------------------------------------
def build_element_points_map(event_live: dict) -> Dict[int, int]:
    return {e["id"]: e["stats"].get("total_points", 0) for e in event_live.get("elements", [])}

def build_element_goals_map(event_live: dict) -> Dict[int, int]:
    return {e["id"]: e["stats"].get("goals_scored", 0) for e in event_live.get("elements", [])}

def compute_transfer_efficiency(transfers: List[dict], event: int, elem_points: Dict[int, int], h: dict) -> int:
    delta = 0
    for t in transfers:
        if t.get("event") == event:
            delta += elem_points.get(t["element_in"], 0) - elem_points.get(t["element_out"], 0)
    return delta - h.get("event_transfers_cost", 0)

# -------------------------------------------------------------------
# Ingestion
# -------------------------------------------------------------------
def ingest_league(league_id: int) -> pd.DataFrame:
    init_db()
    with get_conn() as conn:
        bootstrap = fetch_bootstrap()
        conn.executemany(
            "INSERT OR IGNORE INTO events (id, name, deadline_time) VALUES (?, ?, ?)",
            [(e["id"], e["name"], e["deadline_time"]) for e in bootstrap.get("events", [])]
        )

        # Members
        members = []
        page = 1
        while True:
            data = fetch_league_standings(league_id, page)
            stnd = data.get("standings", {})
            for r in stnd.get("results", []):
                eid = r["entry"]
                members.append((eid, r.get("player_name",""), r.get("entry_name","")))
                conn.execute(
                    "INSERT OR REPLACE INTO teams VALUES (?,?,?,?,?)",
                    (eid, r.get("player_name",""), r.get("entry_name",""), league_id, int(time.time()))
                )
            if stnd.get("has_next"): page += 1
            else: break

        # Each team history
        for eid, _, _ in members:
            hist = fetch_entry_history(eid)
            transfers = fetch_transfers(eid)

            for h in hist.get("current", []):
                gw = h["event"]
                live = fetch_event_live(gw)
                elem_points = build_element_points_map(live)
                elem_goals = build_element_goals_map(live)

                try:
                    picks = fetch_picks(eid, gw)
                except Exception:
                    picks = {"picks": [], "active_chip": None}

                # Captain
                captain_element, captain_base, captain_total = 0, 0, 0
                for p in picks.get("picks", []):
                    if p.get("is_captain"):
                        captain_element = p["element"]
                        base = elem_points.get(captain_element, 0)
                        mult = p.get("multiplier", 1)
                        captain_base, captain_total = base, base * mult
                        break

                # Goals XI
                active_chip = picks.get("active_chip")
                goals_ids = []
                for p in picks.get("picks", []):
                    mult = p.get("multiplier", 0)
                    if mult > 0 or (active_chip == "bboost" and mult == 0):
                        goals_ids.append(p["element"])
                goals_starting_xi = sum(elem_goals.get(eid,0) for eid in set(goals_ids))

                teff = compute_transfer_efficiency(transfers, gw, elem_points, h)

                row = (
                    eid, gw,
                    h.get("points",0), h.get("total_points",0), h.get("overall_rank",0),
                    h.get("bank",0), h.get("value",0),
                    h.get("event_transfers",0), h.get("event_transfers_cost",0),
                    h.get("points_on_bench",0),
                    captain_element, captain_total, captain_base,
                    1 if active_chip else 0, active_chip,
                    goals_starting_xi, teff,
                    int(time.time())
                )
                conn.execute("""
                    INSERT OR REPLACE INTO team_event_stats
                    (entry_id,event,points,total_points,rank,bank,value,
                     transfers,transfers_cost,points_on_bench,
                     captain_element,captain_points,captain_base_points,
                     chip_used,active_chip,goals_starting_xi,transfer_efficiency,created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, row)

        return pd.DataFrame(members, columns=["entry_id","player_name","team_name"])

# -------------------------------------------------------------------
# Queries & Leaderboards
# -------------------------------------------------------------------
def _points_sql_expr(points_are_raw: bool) -> dict:
    if points_are_raw:
        return {
            "raw": "tes.points",
            "net": "tes.points - tes.transfers_cost",
            "hits": "tes.transfers_cost"
        }
    else:
        return {
            "raw": "tes.points + tes.transfers_cost",
            "net": "tes.points",
            "hits": "tes.transfers_cost"
        }

def leaderboard_period(league_id, gw_from, gw_to, metric, points_are_raw=True):
    exprs = _points_sql_expr(points_are_raw)
    metric_expr = {
        "points_net": exprs["net"],
        "points_raw": exprs["raw"],
        "captain_points": "tes.captain_points",
        "captain_base_points": "tes.captain_base_points",
        "transfer_efficiency": "tes.transfer_efficiency",
        "points_on_bench": "tes.points_on_bench",
        "transfers_cost": "tes.transfers_cost",
        "transfers": "tes.transfers",
        "chip_used": "tes.chip_used"
    }
    expr = metric_expr.get(metric, exprs["net"])
    with get_conn() as conn:
        sql = f"""
        SELECT tes.entry_id, t.player_name, t.team_name,
               SUM(CASE WHEN tes.event BETWEEN ? AND ? THEN {expr} ELSE 0 END) AS value
        FROM team_event_stats tes
        JOIN teams t ON t.entry_id = tes.entry_id
        WHERE t.league_id = ?
        GROUP BY tes.entry_id
        ORDER BY value DESC
        """
        return pd.read_sql_query(sql, conn, params=(gw_from, gw_to, league_id))

def monthly_leaderboard(league_id, gw_from, gw_to, points_are_raw=True):
    expr = _points_sql_expr(points_are_raw)
    with get_conn() as conn:
        sql = f"""
        SELECT t.entry_id, t.player_name, t.team_name,
            SUM(CASE WHEN tes.event BETWEEN ? AND ? THEN {expr['raw']} ELSE 0 END) AS total_points_raw,
            SUM(CASE WHEN tes.event BETWEEN ? AND ? THEN {expr['hits']} ELSE 0 END) AS hit_cost,
            SUM(CASE WHEN tes.event BETWEEN ? AND ? THEN {expr['net']} ELSE 0 END) AS total_points_net,
            SUM(CASE WHEN tes.event BETWEEN ? AND ? THEN tes.chip_used ELSE 0 END) AS chips_used,
            SUM(CASE WHEN tes.event BETWEEN ? AND ? THEN tes.transfers ELSE 0 END) AS transfers_made,
            SUM(CASE WHEN tes.event BETWEEN ? AND ? THEN tes.goals_starting_xi ELSE 0 END) AS goals_starting_xi,
            SUM(CASE WHEN tes.event BETWEEN ? AND ? THEN tes.captain_points ELSE 0 END) AS captain_points
        FROM team_event_stats tes
        JOIN teams t ON t.entry_id = tes.entry_id
        WHERE t.league_id=?
        GROUP BY t.entry_id,t.player_name,t.team_name
        """
        df = pd.read_sql_query(sql, conn, params=(gw_from,gw_to,gw_from,gw_to,gw_from,gw_to,
                                                  gw_from,gw_to,gw_from,gw_to,gw_from,gw_to,
                                                  gw_from,gw_to,league_id))
    # sort
    df = df.sort_values(by=["total_points_net","chips_used","transfers_made","goals_starting_xi","captain_points"],
                        ascending=[False,True,True,False,False]).reset_index(drop=True)
    df.insert(0,"rank",df.index+1)
    return df.rename(columns={
        "rank":"Позиция","player_name":"Мениджър","team_name":"Отбор",
        "total_points_net":"Точки","chips_used":"Използвани чипове",
        "transfers_made":"Трансфери","goals_starting_xi":"Отбелязани голове",
        "captain_points":"Точки от капитан","total_points_raw":"Точки (бруто)",
        "hit_cost":"Минуси от трансфери"
    })

# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------
def main():
    st.set_page_config(page_title="FPL Tracker", layout="wide")
    st.title("⚽ FPL Tracker")

    init_db()

    # League input
    c1, c2 = st.columns([2,1])
    with c1:
        league_id = st.number_input("League ID", min_value=1, step=1, value=int(os.getenv("FPL_LEAGUE_ID","2448")))
    with c2:
        if st.button("📥 Import / Refresh League Data", use_container_width=True):
            with st.spinner("Fetching league data…"):
                roster = ingest_league(league_id)
                st.success(f"Imported/updated {len(roster)} teams.")

    # Period leaderboards
    st.subheader("Period Leaderboards")
    events_df = pd.read_sql_query("SELECT id as event FROM events ORDER BY event", get_conn())
    if events_df.empty: return
    gw_min, gw_max = int(events_df["event"].min()), int(events_df["event"].max())
    c3,c4,c5 = st.columns(3)
    with c3: gw_from = st.number_input("GW from", min_value=gw_min, max_value=gw_max, value=gw_min, step=1)
    with c4: gw_to = st.number_input("GW to", min_value=gw_min, max_value=gw_max, value=gw_max, step=1)
    with c5:
        metric_label = st.selectbox("Metric", [
            ("points_net","Total GW points (net) – Z→A"),
            ("points_raw","Total GW points (raw) – Z→A"),
            ("chip_used","Chips used – A→Z"),
            ("transfers","Transfers made – A→Z"),
            ("goals_starting_xi","Goals by starting XI – Z→A"),
            ("captain_points","Captain points (with C/TC) – Z→A"),
            ("captain_base_points","Captain points (raw, undoubled) – Z→A"),
            ("transfer_efficiency","Transfer efficiency – Z→A"),
            ("points_on_bench","Bench points – Z→A"),
            ("transfers_cost","Hit cost – A→Z")], format_func=lambda x:x[1])
    metric = metric_label[0]

    points_are_raw = st.toggle("Treat GW 'points' as raw (before hits)", value=True)
    lb = leaderboard_period(league_id, gw_from, gw_to, metric, points_are_raw)
    asc = metric in ("chip_used","transfers","transfers_cost")
    st.dataframe(lb.sort_values("value", ascending=asc), use_container_width=True, hide_index=True)

    # Monthly leaderboard
    st.subheader("Monthly Leaderboard")
    mlb = monthly_leaderboard(league_id, gw_from, gw_to, points_are_raw)
    if not mlb.empty:
        st.dataframe(mlb, use_container_width=True, hide_index=True)
        raw, hits, net = mlb["Точки (бруто)"].sum(), mlb["Минуси от трансфери"].sum(), mlb["Точки"].sum()
        if abs((raw - hits) - net) < 1e-6:
            st.success("Проверка: Точки (бруто) – Минуси от трансфери = Точки ✅")
        else:
            st.error("Разминаване! Проверете настройката 'points_are_raw' ❌")

if __name__=="__main__": main()
