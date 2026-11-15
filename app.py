
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from astroquery.jplhorizons import Horizons
from astropy.time import Time

st.set_page_config(page_title="LunaWindow", layout="wide")
st.title("LunaWindow")
st.caption("NASA Space Apps 2026 | Lunar Landing Site Navigator")
st.write("Compare sunlight and direct-Earth visibility at two locations on the Moon.")

@st.cache_data(ttl=3600)
def get_site_data(lat, lon, start_day, days, step_hours):
    start = datetime.combine(start_day, datetime.min.time(), tzinfo=timezone.utc)
    end = start + timedelta(days=days)
    times = {
        "start": start.strftime("%Y-%m-%d %H:%M"),
        "stop": end.strftime("%Y-%m-%d %H:%M"),
        "step": f"{step_hours}h",
    }
    location = {"lon": lon, "lat": lat, "elevation": 0, "body": 301}
    results = []

    for name, body_id in [("Sun", "10"), ("Earth", "399")]:
        table = Horizons(
            id=body_id, location=location, epochs=times
        ).ephemerides(quantities="4", refraction=False)

        if len(table) == 0:
            raise ValueError(f"No data returned for {name}.")

        utc = Time(
            np.array(table["datetime_jd"], dtype=float),
            format="jd", scale="utc"
        )
        timestamps = pd.to_datetime(utc.isot, utc=True).round("s")

        results.append(pd.DataFrame({
            "UTC": timestamps,
            "Body": name,
            "Elevation (degrees)": np.array(table["EL"], dtype=float),
        }))

    return pd.concat(results, ignore_index=True)

with st.form("inputs"):
    left, right = st.columns(2)

    with left:
        st.subheader("Site A")
        lat_a = st.number_input("Latitude A", -90.0, 90.0, -89.0)
        lon_a = st.number_input("East longitude A", -180.0, 180.0, 0.0)

    with right:
        st.subheader("Site B")
        lat_b = st.number_input("Latitude B", -90.0, 90.0, -89.0)
        lon_b = st.number_input("East longitude B", -180.0, 180.0, 30.0)

    start_day = st.date_input("Start date (UTC)", date(2026, 11, 15))
    days = st.selectbox("Duration (days)", [1, 2, 3], index=1)
    step = st.selectbox("Sampling interval (hours)", [1, 2, 3, 6], index=2)
    analyze = st.form_submit_button("Analyze with NASA Horizons")

if analyze:
    with st.spinner("Loading ephemerides from NASA JPL Horizons..."):
        try:
            a = get_site_data(lat_a, lon_a, start_day, days, step)
            b = get_site_data(lat_b, lon_b, start_day, days, step)
            a["Site"], b["Site"] = "A", "B"
            st.session_state.data = pd.concat([a, b], ignore_index=True)
        except Exception as error:
            st.error("Horizons request failed. Check internet and error below.")
            st.exception(error)

if "data" in st.session_state:
    df = st.session_state.data

    st.subheader("Sun and Earth elevations")
    chart = px.line(
        df, x="UTC", y="Elevation (degrees)",
        color="Site", line_dash="Body",
        title="Angles relative to an ideal lunar horizon"
    )
    chart.add_hline(y=0, line_dash="dot", annotation_text="Ideal horizon")
    st.plotly_chart(chart, use_container_width=True)

    summary = []
    for site in ["A", "B"]:
        subset = df[df["Site"] == site]
        pivot = subset.pivot(
            index="UTC", columns="Body", values="Elevation (degrees)"
        )
        summary.append({
            "Site": site,
            "Sun visible (%)": round(100 * (pivot["Sun"] > 0).mean(), 1),
            "Earth visible (%)": round(100 * (pivot["Earth"] > 0).mean(), 1),
            "Both visible (%)": round(
                100 * ((pivot["Sun"] > 0) & (pivot["Earth"] > 0)).mean(), 1
            ),
        })

    st.subheader("Site comparison")
    st.dataframe(pd.DataFrame(summary), hide_index=True)
    st.download_button(
        "Export NASA data (CSV)",
        df.to_csv(index=False),
        "lunawindow.csv"
    )
    st.warning(
        "Prototype only: terrain obstruction, solar panel orientation, "
        "and communication link budgets are not modeled. Percentages "
        "represent sampled times, not precise durations."
    )
else:
    st.info("Choose two locations and click Analyze to start.")
