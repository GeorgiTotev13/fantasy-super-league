# season_awards_tab.py
# Drop-in Streamlit tab for Fantasy Super League Season Awards.
# Usage in app.py:
#   from season_awards_tab import render_season_awards_tab
#   render_season_awards_tab(db_path="fpl_tracker.db", league_id=2448)

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st


@dataclass
class Cols:
    team_id: str
    event: str
    player_name: str
    entry_name: str
    total_points: Optional[str]
    net_points: Optional[str]
    captain_base_points: Optional[str]
    captain_points: Optional[str]
    active_chip: Optional[str]
    chip_used: Optional[str]
    goals_starting_xi: Optional[str]
    own_goals: Optional[str]


def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    q = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
    return con.execute(q, (table,)).fetchone() is not None


def _cols(con: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()]


def _pick(cols: list[str], *names: str) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def _detect_columns(con: sqlite3.Connection) -> Cols:
    tes_cols = _cols(con, "team_event_stats")
    teams_cols = _cols(con, "teams")

    team_id = _pick(tes_cols, "entry", "entry_id", "team_id", "id") or "entry"
    event = _pick(tes_cols, "event", "event_id", "gw", "gameweek") or "event"

    player_name = _pick(teams_cols, "player_name", "manager_name", "name") or "player_name"
    entry_name = _pick(teams_cols, "entry_name", "team_name") or "entry_name"

    return Cols(
        team_id=team_id,
        event=event,
        player_name=player_name,
        entry_name=entry_name,
        total_points=_pick(tes_cols, "points", "event_points", "gw_points", "total_points"),
        net_points=_pick(tes_cols, "net_points", "points_net", "event_net_points"),
        captain_base_points=_pick(tes_cols, "captain_base_points", "captain_points_base"),
        captain_points=_pick(tes_cols, "captain_points", "captain_total_points", "captain_score"),
        active_chip=_pick(tes_cols, "active_chip", "chip", "chip_name"),
        chip_used=_pick(tes_cols, "chip_used"),
        goals_starting_xi=_pick(tes_cols, "goals_starting_xi", "starting_xi_goals", "goals_xi"),
        own_goals=_pick(tes_cols, "own_goals", "own_goals_starting_xi", "own_goals_xi"),
    )


def _load_data(db_path: str | Path, league_id: Optional[int] = None) -> pd.DataFrame:
    con = sqlite3.connect(str(db_path))
    try:
        if not _table_exists(con, "teams") or not _table_exists(con, "team_event_stats"):
            raise RuntimeError("Expected tables 'teams' and 'team_event_stats' in the SQLite DB.")

        c = _detect_columns(con)
        teams_cols = _cols(con, "teams")
        tes_cols = _cols(con, "team_event_stats")

        # Join key detection: normally teams.entry / team_event_stats.entry or entry_id.
        teams_entry = _pick(teams_cols, "entry", "entry_id", "team_id", "id") or "entry"

        select_bits = [
            f"s.{c.team_id} AS entry_id",
            f"s.{c.event} AS event",
            f"t.{c.player_name} AS player_name",
            f"t.{c.entry_name} AS entry_name",
        ]

        def add(col: Optional[str], alias: str):
            if col:
                select_bits.append(f"s.{col} AS {alias}")
            else:
                select_bits.append(f"NULL AS {alias}")

        add(c.total_points, "gw_points")
        add(c.net_points, "net_points")
        add(c.captain_base_points, "captain_base_points")
        add(c.captain_points, "captain_points")
        add(c.active_chip, "active_chip")
        add(c.chip_used, "chip_used")
        add(c.goals_starting_xi, "goals_starting_xi")
        add(c.own_goals, "own_goals")

        where = ""
        params: list[Any] = []
        if league_id is not None and "league_id" in teams_cols:
            where = "WHERE t.league_id = ?"
            params.append(league_id)

        sql = f"""
            SELECT {', '.join(select_bits)}
            FROM team_event_stats s
            JOIN teams t ON t.{teams_entry} = s.{c.team_id}
            {where}
            ORDER BY s.{c.event}, t.{c.entry_name}
        """
        df = pd.read_sql_query(sql, con, params=params)
    finally:
        con.close()

    for col in ["event", "gw_points", "net_points", "captain_base_points", "captain_points", "goals_starting_xi", "own_goals"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # If captain_points is not stored, derive from captain_base_points and chip.
    # Normal captain multiplier = 2; Triple Captain multiplier = 3.
    if df["captain_points"].isna().all() and not df["captain_base_points"].isna().all():
        chip = df["active_chip"].fillna("").astype(str).str.lower()
        multiplier = chip.apply(lambda x: 3 if x in {"3xc", "triple_captain", "triple captain"} else 2)
        df["captain_points"] = df["captain_base_points"].fillna(0) * multiplier

    # Prefer net points for GW highs/lows if available, otherwise raw GW points.
    df["award_gw_points"] = df["net_points"].where(df["net_points"].notna(), df["gw_points"])

    return df


def _winner_rows(df: pd.DataFrame, col: str, mode: str = "max") -> pd.DataFrame:
    clean = df.dropna(subset=[col]).copy()
    if clean.empty:
        return clean
    value = clean[col].max() if mode == "max" else clean[col].min()
    return clean[clean[col] == value].sort_values(["player_name", "entry_name", "event"])


def _team_totals(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby(["entry_id", "player_name", "entry_name"], as_index=False).agg(
        season_points=("award_gw_points", "sum"),
        captain_points_total=("captain_points", "sum"),
        goals_starting_xi_total=("goals_starting_xi", "sum"),
        own_goals_total=("own_goals", "sum"),
    )
    return agg


def _format_people(rows: pd.DataFrame, value_col: str, include_gw: bool = True) -> dict[str, Any]:
    if rows.empty:
        return {"label": "No data", "value": None, "count": 0, "rows": []}
    value = rows[value_col].iloc[0]
    out_rows = []
    for _, r in rows.iterrows():
        out_rows.append({
            "player_name": r.get("player_name"),
            "entry_name": r.get("entry_name"),
            "event": int(r["event"]) if include_gw and pd.notna(r.get("event")) else None,
            "value": int(value) if float(value).is_integer() else float(value),
        })
    if len(out_rows) == 1:
        label = f"{out_rows[0]['player_name']} ({out_rows[0]['entry_name']})"
        if include_gw and out_rows[0]["event"]:
            label += f" — GW{out_rows[0]['event']}"
    else:
        label = f"{len(out_rows)} играчи"
    return {"label": label, "value": out_rows[0]["value"], "count": len(out_rows), "rows": out_rows}


def build_season_awards(db_path: str | Path = "fpl_tracker.db", league_id: int = 2448) -> dict[str, Any]:
    df = _load_data(db_path, league_id)
    totals = _team_totals(df)

    top3 = totals.sort_values("season_points", ascending=False).head(3).copy()

    max_gw = _winner_rows(df, "award_gw_points", "max")
    min_gw = _winner_rows(df, "award_gw_points", "min")

    cap_total = _winner_rows(totals.rename(columns={"captain_points_total": "captain_points"}), "captain_points", "max")
    goals_total = _winner_rows(totals.rename(columns={"goals_starting_xi_total": "goals_starting_xi"}), "goals_starting_xi", "max")
    own_goals_total = _winner_rows(totals.rename(columns={"own_goals_total": "own_goals"}), "own_goals", "max")

    max_cap_gw = _winner_rows(df, "captain_points", "max")
    min_cap_gw = _winner_rows(df, "captain_points", "min")

    awards = {
        "league_id": league_id,
        "top3": top3[["player_name", "entry_name", "season_points"]].to_dict("records"),
        "highest_gw_score": _format_people(max_gw, "award_gw_points", True),
        "lowest_gw_score": _format_people(min_gw, "award_gw_points", True),
        "most_captain_points_total": _format_people(cap_total, "captain_points", False),
        "most_goals_starting_xi": _format_people(goals_total, "goals_starting_xi", False),
        "most_captain_points_single_gw": _format_people(max_cap_gw, "captain_points", True),
        "fewest_captain_points_single_gw": _format_people(min_cap_gw, "captain_points", True),
        "most_own_goals": _format_people(own_goals_total, "own_goals", False),
    }
    return awards


def render_season_awards_tab(db_path: str | Path = "fpl_tracker.db", league_id: int = 2448) -> None:
    st.header("🏆 Fantasy Super League — Season Awards")

    db_path = Path(db_path)
    if not db_path.exists():
        st.error(f"Database not found: {db_path}")
        return

    try:
        awards = build_season_awards(db_path=db_path, league_id=league_id)
    except Exception as e:
        st.exception(e)
        return

    top3 = awards["top3"]
    if len(top3) >= 3:
        c1, c2, c3 = st.columns([1, 1.2, 1])
        with c1:
            st.metric("🥈 2nd", f"{top3[1]['season_points']:.0f}", top3[1]["player_name"])
            st.caption(top3[1]["entry_name"])
        with c2:
            st.metric("🥇 Champion", f"{top3[0]['season_points']:.0f}", top3[0]["player_name"])
            st.caption(top3[0]["entry_name"])
        with c3:
            st.metric("🥉 3rd", f"{top3[2]['season_points']:.0f}", top3[2]["player_name"])
            st.caption(top3[2]["entry_name"])

    st.divider()

    a = awards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("🚀 Най-силен GW")
        st.metric(a["highest_gw_score"]["label"], a["highest_gw_score"]["value"])
        st.subheader("👑 Captain King")
        st.metric(a["most_captain_points_total"]["label"], a["most_captain_points_total"]["value"])
    with c2:
        st.subheader("⚽ Goal Machine")
        st.metric(a["most_goals_starting_xi"]["label"], a["most_goals_starting_xi"]["value"])
        st.subheader("🔥 Най-силен капитан в GW")
        st.metric(a["most_captain_points_single_gw"]["label"], a["most_captain_points_single_gw"]["value"])
    with c3:
        st.subheader("💀 Най-слаб GW")
        st.metric(a["lowest_gw_score"]["label"], a["lowest_gw_score"]["value"])
        st.subheader("🤡 Captain Fail")
        st.metric(a["fewest_captain_points_single_gw"]["label"], a["fewest_captain_points_single_gw"]["value"])

    st.subheader("🥅 Own Goal King")
    st.metric(a["most_own_goals"]["label"], a["most_own_goals"]["value"])

    st.divider()
    st.download_button(
        "⬇️ Export Infographic Data JSON",
        data=json.dumps(awards, ensure_ascii=False, indent=2),
        file_name="season_awards_2448.json",
        mime="application/json",
    )

    with st.expander("Raw awards JSON"):
        st.json(awards)
