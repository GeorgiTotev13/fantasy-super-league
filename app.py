"""
FPL Tracker – Streamlit App (single file)

Quickstart
----------
1) Create a virtual env and install deps:
   pip install streamlit requests pandas
2) Run the app:
   streamlit run app.py

Notes
-----
• Works locally with public FPL API (no auth). 
• Persists data to a local SQLite file: fpl_tracker.db
• Supports: league import, per‑GW stats, period leaderboards, and history snapshots.
• Tested against the 2024/25+ FPL API format; endpoints are stable historically.

Env vars (optional)
-------------------
• FPL_LEAGUE_ID: default league id loaded at startup.

"""

import os
import time
import sqlite3
from typing import Dict, List, Tuple

import pandas as pd
import requests
import streamlit as st
import io
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"


# ------------------------------
# Constants & API helpers
# ------------------------------
API_BASE = "https://fantasy.premierleague.com/api"
HEADERS = {
    "User-Agent": "FPL-Tracker/1.0 (+https://fantasy.premierleague.com)"
}
DB_PATH = "fpl_tracker.db"

# New columns are handled via init_db() migration above.

# ------------------------------
# DB helpers
# ------------------------------
SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

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
    transfer_efficiency INTEGER,
    created_at INTEGER,
    PRIMARY KEY (entry_id, event)
);

CREATE TABLE IF NOT EXISTS transfers (
    entry_id INTEGER,
    event INTEGER,
    element_in INTEGER,
    element_out INTEGER,
    created_at INTEGER
);
"""


def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA_SQL)
        # Migrate existing DBs to add new columns if missing
        cols = {row[1] for row in conn.execute("PRAGMA table_info(team_event_stats)")}
        if "chip_used" not in cols:
            conn.execute("ALTER TABLE team_event_stats ADD COLUMN chip_used INTEGER DEFAULT 0")
        if "active_chip" not in cols:
            conn.execute("ALTER TABLE team_event_stats ADD COLUMN active_chip TEXT")
        if "goals_starting_xi" not in cols:
            conn.execute("ALTER TABLE team_event_stats ADD COLUMN goals_starting_xi INTEGER DEFAULT 0")
        if "captain_base_points" not in cols:
            conn.execute("ALTER TABLE team_event_stats ADD COLUMN captain_base_points INTEGER DEFAULT 0")


# ------------------------------
# API fetchers
# ------------------------------

def get_json(url: str) -> dict:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_bootstrap() -> dict:
    return get_json(f"{API_BASE}/bootstrap-static/")


def fetch_league_standings(league_id: int, page: int = 1) -> dict:
    """Fetch standings for Classic leagues; if not found, try H2H automatically.
    The correct endpoints are:
      • /leagues-classic/{id}/standings/
      • /leagues-h2h/{id}/standings/
    """
    kinds = ("leagues-classic", "leagues-h2h")
    last_err = None
    for kind in kinds:
        url = f"{API_BASE}/{kind}/{league_id}/standings/?page_standings={page}"
        try:
            return get_json(url)
        except requests.HTTPError as e:
            last_err = e
            if getattr(e, "response", None) is not None and e.response.status_code == 404:
                # Try the next kind
                continue
            raise
    # If both classic and h2h returned 404, raise a clearer error
    raise ValueError(
        "League not found as Classic or H2H. Double‑check the League ID and type (Classic vs H2H)."
    )


def fetch_entry_history(entry_id: int) -> dict:
    return get_json(f"{API_BASE}/entry/{entry_id}/history/")


def fetch_entry(entry_id: int) -> dict:
    return get_json(f"{API_BASE}/entry/{entry_id}/")


def fetch_transfers(entry_id: int) -> List[dict]:
    return get_json(f"{API_BASE}/entry/{entry_id}/transfers/")


def fetch_picks(entry_id: int, event: int) -> dict:
    return get_json(f"{API_BASE}/entry/{entry_id}/event/{event}/picks/")


def fetch_event_live(event: int) -> dict:
    return get_json(f"{API_BASE}/event/{event}/live/")

# ------------------------------
# Computations
# ------------------------------

def build_element_points_map(event_live: dict) -> Dict[int, int]:
    # Map element_id -> total_points for the GW
    m = {}
    for e in event_live.get("elements", []):
        m[e["id"]] = e["stats"].get("total_points", 0)
    return m


def build_element_goals_map(event_live: dict) -> Dict[int, int]:
    # Map element_id -> goals_scored for the GW
    m = {}
    for e in event_live.get("elements", []):
        m[e["id"]] = e["stats"].get("goals_scored", 0)
    return m


def compute_captain_points(picks: dict, elem_points: Dict[int, int]) -> Tuple[int, int]:
    """Return (captain_element_id, captain_points_contribution)
    Contribution = points * (multiplier - 1), i.e., extra points from C/TC.
    """
    cap_id = None
    cap_bonus = 0
    for p in picks.get("picks", []):
        if p.get("is_captain"):
            cap_id = p["element"]
            mult = p.get("multiplier", 1)
            pts = elem_points.get(cap_id, 0)
            cap_bonus = pts * (mult - 1)
            break
    return cap_id or 0, cap_bonus


def compute_transfer_efficiency(entry_transfers: List[dict], event: int, elem_points: Dict[int, int], history_row: dict) -> int:
    """Approximate transfer efficiency for a GW:
    Sum(points_in - points_out) - hit_cost.
    Uses live points for transferred players that GW.
    """
    delta = 0
    for t in entry_transfers:
        if t.get("event") == event:
            delta += elem_points.get(t["element_in"], 0) - elem_points.get(t["element_out"], 0)
    hit = history_row.get("event_transfers_cost", 0)
    return delta - hit

# ------------------------------
# Persistence
# ------------------------------

def upsert_team(conn, entry_id: int, player_name: str, team_name: str, league_id: int):
    conn.execute(
        """
        INSERT INTO teams (entry_id, player_name, team_name, league_id, last_updated)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(entry_id) DO UPDATE SET
            player_name=excluded.player_name,
            team_name=excluded.team_name,
            league_id=excluded.league_id,
            last_updated=excluded.last_updated
        """,
        (entry_id, player_name, team_name, league_id, int(time.time())),
    )


def save_events(conn, bootstrap: dict):
    rows = [(e["id"], e["name"], e["deadline_time"]) for e in bootstrap.get("events", [])]
    conn.executemany(
        "INSERT OR IGNORE INTO events (id, name, deadline_time) VALUES (?, ?, ?)", rows
    )


def save_transfers(conn, entry_id: int, transfers: List[dict]):
    rows = [
        (entry_id, t.get("event"), t.get("element_in"), t.get("element_out"), int(time.time()))
        for t in transfers
    ]
    if rows:
        conn.executemany(
            "INSERT INTO transfers (entry_id, event, element_in, element_out, created_at) VALUES (?, ?, ?, ?, ?)",
            rows,
        )


def save_team_event_stats(conn, row: dict):
    cols = [
        "entry_id","event","points","total_points","rank","bank","value",
        "transfers","transfers_cost","points_on_bench","captain_element","captain_points",
        "transfer_efficiency","chip_used","active_chip","goals_starting_xi","captain_base_points",
        "created_at"
    ]
    placeholders = ",".join(["?"] * len(cols))
    values = [row[c] for c in cols]
    conn.execute(
        f"INSERT OR REPLACE INTO team_event_stats ({','.join(cols)}) VALUES ({placeholders})",
        values,
    )

# ------------------------------
# Ingest pipeline
# ------------------------------

def ingest_league(league_id: int) -> pd.DataFrame:
    """Fetch league, members, and per‑GW stats -> store to DB. Returns a roster DF."""
    init_db()
    with get_conn() as conn:
        bootstrap = fetch_bootstrap()
        save_events(conn, bootstrap)

        # paginate standings
        members: List[Tuple[int, str, str]] = []
        page = 1
        while True:
            data = fetch_league_standings(league_id, page)
            stnd = data.get("standings", {})
            results = stnd.get("results", [])
            for r in results:
                entry_id = r["entry"]
                player_name = r.get("player_name", "")
                team_name = r.get("entry_name", "")
                members.append((entry_id, player_name, team_name))
                upsert_team(conn, entry_id, player_name, team_name, league_id)
            if stnd.get("has_next"):
                page += 1
            else:
                break

        # For each entry, collect history + transfers and compute per‑GW derived stats
        for (entry_id, _, _) in members:
            history = fetch_entry_history(entry_id)
            current_rows = history.get("current", [])  # per‑GW rows
            transfers = fetch_transfers(entry_id)
            save_transfers(conn, entry_id, transfers)

            for h in current_rows:
                gw = h["event"]
                # live element points for the GW (once per loop ok; could cache)
                live = fetch_event_live(gw)
                elem_points = build_element_points_map(live)
                elem_goals = build_element_goals_map(live)
                # captain
                try:
                    picks = fetch_picks(entry_id, gw)
                except Exception:
                    picks = {"picks": [], "active_chip": None}
                captain_element, captain_points = compute_captain_points(picks, elem_points)
                captain_base_points = elem_points.get(captain_element, 0)
                active_chip = picks.get("active_chip")
                # Goals by starting XI (include bench if Bench Boost played)
                goals_ids = []
                for p in picks.get("picks", []):
                    mult = p.get("multiplier", 0)
                    if mult > 0 or (active_chip == "bboost" and mult == 0):
                        goals_ids.append(p["element"])
                goals_starting_xi = sum(elem_goals.get(eid, 0) for eid in set(goals_ids))
                # transfer efficiency
                teff = compute_transfer_efficiency(transfers, gw, elem_points, h)

                row = {
                    "entry_id": entry_id,
                    "event": gw,
                    "points": h.get("points", 0),
                    "total_points": h.get("total_points", 0),
                    "rank": h.get("overall_rank", 0),
                    "bank": h.get("bank", 0),
                    "value": h.get("value", 0),
                    "transfers": h.get("event_transfers", 0),
                    "transfers_cost": h.get("event_transfers_cost", 0),
                    "points_on_bench": h.get("points_on_bench", 0),
                    "captain_element": captain_element,
                    "captain_points": captain_points,
                    "transfer_efficiency": teff,
                    "chip_used": 1 if active_chip else 0,
                    "active_chip": active_chip,
                    "goals_starting_xi": goals_starting_xi,
                    "captain_base_points": captain_base_points,
                    "created_at": int(time.time()),
                }
                save_team_event_stats(conn, row)

        df = pd.DataFrame(members, columns=["entry_id", "player_name", "team_name"])
        return df

# ------------------------------
# Queries & Leaderboards
# ------------------------------

def get_events_df() -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query("SELECT id as event, name, deadline_time FROM events ORDER BY event", conn)


def get_roster_df(league_id: int) -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query(
            "SELECT entry_id, player_name, team_name FROM teams WHERE league_id=? ORDER BY player_name",
            conn,
            params=(league_id,),
        )


def leaderboard_period(league_id: int, gw_from: int, gw_to: int, metric: str) -> pd.DataFrame:
    with get_conn() as conn:
        base = """
        SELECT tes.entry_id, t.player_name, t.team_name,
               SUM(CASE WHEN tes.event BETWEEN ? AND ? THEN {metric} ELSE 0 END) AS value
        FROM team_event_stats tes
        JOIN teams t ON t.entry_id = tes.entry_id
        WHERE t.league_id = ?
        GROUP BY tes.entry_id
        ORDER BY value DESC
        """.format(metric=metric)
        df = pd.read_sql_query(base, conn, params=(gw_from, gw_to, league_id))
        return df

def monthly_leaderboard(league_id: int, gw_from: int, gw_to: int) -> pd.DataFrame:
    """Combined leaderboard with multi-criteria ordering for the selected GW range.
    Order: total_points_net DESC, chips_used ASC, transfers_made ASC,
           goals_starting_xi DESC, captain_points DESC.
    Also returns columns for each metric.
    """
    with get_conn() as conn:
        sql = """
            SELECT
                t.entry_id,
                t.player_name,
                t.team_name,
                -- raw points (points + hits)
                SUM(CASE WHEN tes.event BETWEEN ? AND ? THEN tes.points + tes.transfers_cost ELSE 0 END) AS total_points_raw,
                -- hit cost
                SUM(CASE WHEN tes.event BETWEEN ? AND ? THEN tes.transfers_cost ELSE 0 END) AS hit_cost,
                -- net points (already includes hit costs in tes.points)
                SUM(CASE WHEN tes.event BETWEEN ? AND ? THEN tes.points ELSE 0 END) AS total_points_net,
                -- chips used
                SUM(CASE WHEN tes.event BETWEEN ? AND ? THEN tes.chip_used ELSE 0 END) AS chips_used,
                -- transfers made
                SUM(CASE WHEN tes.event BETWEEN ? AND ? THEN tes.transfers ELSE 0 END) AS transfers_made,
                -- goals by XI
                SUM(CASE WHEN tes.event BETWEEN ? AND ? THEN tes.goals_starting_xi ELSE 0 END) AS goals_starting_xi,
                -- captain raw points
                SUM(CASE WHEN tes.event BETWEEN ? AND ? THEN tes.captain_base_points ELSE 0 END) AS captain_points
            FROM team_event_stats tes
            JOIN teams t ON t.entry_id = tes.entry_id
            WHERE t.league_id = ?
            GROUP BY t.entry_id, t.player_name, t.team_name
        """
        params = (
            gw_from, gw_to,   # total_points_raw
            gw_from, gw_to,   # hit_cost
            gw_from, gw_to,   # total_points_net
            gw_from, gw_to,   # chips_used
            gw_from, gw_to,   # transfers_made
            gw_from, gw_to,   # goals_starting_xi
            gw_from, gw_to,   # captain_points
            league_id,
        )
        df = pd.read_sql_query(sql, conn, params=params)

        # Multi-criteria sort
        df = df.sort_values(
            by=["total_points_net", "chips_used", "transfers_made", "goals_starting_xi", "captain_points"],
            ascending=[False, True, True, False, False],
            kind="mergesort",
        ).reset_index(drop=True)

        # Add rank column
        df.insert(0, "rank", df.index + 1)

        # Column order
        cols = [
            "rank", "player_name", "team_name",
            "total_points_net", "chips_used", "transfers_made", "goals_starting_xi", "captain_points",
            "total_points_raw", "hit_cost",
        ]
        existing = [c for c in cols if c in df.columns]
        df = df[existing]

        # Rename columns for nicer display
        df = df.rename(columns={
            "rank": "Позиция",
            "player_name": "Мениджър",
            "team_name": "Отбор",
            "total_points_net": "Точки",
            "chips_used": "Използвани чипове",
            "transfers_made": "Трансфери",
            "goals_starting_xi": "Отбелязани голове",
            "captain_points": "Точки от капитан",
            "total_points_raw": "Точки (бруто)",
            "hit_cost": "Минуси от трансфери"
        })

        return df


def monthly_leaderboard_image(df: pd.DataFrame, title: str = "Monthly Leaderboard") -> tuple[bytes, bytes]:
    """Render the monthly leaderboard DataFrame to a high-resolution PNG and JPEG.
    Returns (png_bytes, jpg_bytes).
    If df is empty, returns (b"", b"").
    """
    if df.empty:
        return b"", b""

    # Only keep display columns
    show_cols = df.columns.tolist()  # use all columns in their current names
    tab = df[show_cols].copy()

    # Figure sizing based on rows/cols
    n_rows, n_cols = tab.shape
    col_width = 2.0
    row_height = 0.55
    fig_w = max(10, n_cols * col_width)
    fig_h = max(2.5, n_rows * row_height + 2.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.axis('off')
    ax.set_title(title, fontsize=18, pad=12)

    # Build table
    table = ax.table(cellText=tab.values,
                     colLabels=tab.columns,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)

    # Style header
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # header row
            cell.set_facecolor('#1f77b4')
            cell.set_text_props(color='white', weight='bold')

    # Highlight Top 3 (data rows start at 1 in matplotlib table)
    colors = ['#ffd700', '#c0c0c0', '#cd7f32']  # gold, silver, bronze
    for i, color in enumerate(colors, start=1):
        if i <= n_rows:
            for col in range(n_cols):
                table[(i, col)].set_facecolor(color)

    # Tight layout and save to buffers
    fig.tight_layout()

    buf_png = io.BytesIO()
    fig.savefig(buf_png, format='png', bbox_inches='tight')
    buf_png.seek(0)

    buf_jpg = io.BytesIO()
    fig.savefig(buf_jpg, format='jpeg', bbox_inches='tight')
    buf_jpg.seek(0)

    plt.close(fig)
    return buf_png.getvalue(), buf_jpg.getvalue()




def timeseries_for_team(entry_id: int) -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query(
            """
            SELECT event, points, total_points, rank, transfers, transfers_cost,
                   points_on_bench, captain_points, transfer_efficiency
            FROM team_event_stats
            WHERE entry_id=?
            ORDER BY event
            """,
            conn,
            params=(entry_id,),
        )

# ------------------------------
# UI – Streamlit
# ------------------------------

def main():
    st.set_page_config(page_title="FPL Tracker", layout="wide")
    st.title("⚽ FPL Tracker")
    st.caption("Monitor your mini-league with period leaderboards and history snapshots.")

    init_db()

    c1, c2 = st.columns([2, 1])
    with c1:
        league_id = st.number_input("League ID", min_value=1, step=1, value=int(os.getenv("FPL_LEAGUE_ID", "2448")))
    with c2:
        if st.button("📥 Import / Refresh League Data", use_container_width=True):
            with st.spinner("Fetching league data and updating history…"):
                roster = ingest_league(league_id)
                st.success(f"Imported/updated {len(roster)} teams.")

    st.divider()

    # Roster
    roster_df = get_roster_df(league_id)
    st.subheader("League Roster")
    if roster_df.empty:
        st.info("No teams loaded yet. Enter your League ID and click Refresh.")
    else:
        st.dataframe(roster_df, use_container_width=True, hide_index=True)

    st.divider()

    # Period Leaderboards
    st.subheader("Period Leaderboards")
    events_df = get_events_df()
    if events_df.empty:
        st.info("No events in DB yet. Click Refresh to pull the season metadata.")
        return

    c3, c4, c5 = st.columns(3)
    with c3:
        gw_from = st.number_input("GW from", min_value=int(events_df["event"].min()), max_value=int(events_df["event"].max()), value=int(events_df["event"].min()))
    with c4:
        gw_to = st.number_input("GW to", min_value=int(events_df["event"].min()), max_value=int(events_df["event"].max()), value=int(events_df["event"].max()))
    with c5:
        metric_label = st.selectbox(
            "Metric",
            [
                ("points", "Total points (net) – Z→A"),
                ("chip_used", "Chips used – A→Z"),
                ("transfers", "Transfers made – A→Z"),
                ("goals_starting_xi", "Goals by starting XI – Z→A"),
                ("captain_base_points", "Captain points (raw) – Z→A"),
                ("captain_points", "Captain bonus (extra) – Z→A"),
                ("transfer_efficiency", "Transfer efficiency – Z→A"),
                ("points_on_bench", "Bench points – Z→A"),
                ("transfers_cost", "Hit cost – A→Z")
            ],
            index=0,
            format_func=lambda x: x[1],
        )
    metric = metric_label[0]

    lb = leaderboard_period(league_id, gw_from, gw_to, metric)
    asc = metric in ("chip_used", "transfers", "transfers_cost")
    lb = lb.sort_values("value", ascending=asc)
    st.dataframe(lb.rename(columns={"value": metric}), use_container_width=True, hide_index=True)

    st.caption("Notes: Total points already include hit costs. 'Chips used' counts GWs where any chip was active. 'Goals by starting XI' includes bench only when Bench Boost was active.")

    st.subheader("Monthly Leaderboard (multi‑criteria)")
    mlb = monthly_leaderboard(league_id, gw_from, gw_to)
    if not mlb.empty:
        st.dataframe(mlb, use_container_width=True, hide_index=True)
        # Downloads
        csv = mlb.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name=f"monthly_leaderboard_gw{gw_from}-{gw_to}.csv", mime="text/csv")
        try:
            png_bytes, jpg_bytes = monthly_leaderboard_image(mlb, title=f"Monthly Leaderboard • GW {gw_from}–{gw_to}")
            st.download_button("Download PNG (hi‑res)", data=png_bytes, file_name=f"monthly_leaderboard_gw{gw_from}-{gw_to}.png", mime="image/png")
            st.download_button("Download JPEG (hi‑res)", data=jpg_bytes, file_name=f"monthly_leaderboard_gw{gw_from}-{gw_to}.jpg", mime="image/jpeg")
            st.image(png_bytes, caption="Preview (PNG)", use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render image export: {e}")
    else:
        st.info("No data for the selected range yet. Try importing or widen the GW range.")

    st.divider()

    # Team detail
    st.subheader("Team Timeseries")
    if not roster_df.empty:
        entry_id = st.selectbox(
            "Select team",
            roster_df["entry_id"].tolist(),
            format_func=lambda eid: f"{eid} – {roster_df.set_index('entry_id').loc[eid, 'player_name']} ({roster_df.set_index('entry_id').loc[eid, 'team_name']})",
        )
        ts = timeseries_for_team(entry_id)
        st.dataframe(ts, use_container_width=True)

        # Quick charts
        if not ts.empty:
            st.line_chart(ts.set_index("event")["points"], use_container_width=True)
            st.bar_chart(ts.set_index("event")["transfer_efficiency"], use_container_width=True)

    st.divider()
    st.caption(
        "Made with ❤️. Data © Premier League – unofficial API. This app stores only public league data locally."
    )


if __name__ == "__main__":
    main()
