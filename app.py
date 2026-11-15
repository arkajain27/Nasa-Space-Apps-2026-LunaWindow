
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
            "Azimuth (degrees)": np.array(table["AZ"], dtype=float),
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


# Lightweight 3D Moon visualization
if "data" in st.session_state:
    import plotly.graph_objects as go

    st.divider()
    st.subheader("3D Lunar Site Explorer")
    st.caption(
        "Schematic lunar globe. Site markers use your coordinates. "
        "Surface terrain is not modeled."
    )

    longitude, latitude = np.meshgrid(
        np.linspace(-np.pi, np.pi, 48),
        np.linspace(-np.pi / 2, np.pi / 2, 28)
    )

    x = np.cos(latitude) * np.cos(longitude)
    y = np.cos(latitude) * np.sin(longitude)
    z = np.sin(latitude)

    moon = go.Figure()

    moon.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale=[
            [0, "#364759"],
            [1, "#9aa9b9"]
        ],
        showscale=False,
        opacity=0.95,
        hoverinfo="skip"
    ))

    sites = [
        ("Site A", lat_a, lon_a, "#49e5af"),
        ("Site B", lat_b, lon_b, "#ffb454")
    ]

    for name, lat, lon, color in sites:
        a = np.radians(lat)
        b = np.radians(lon)

        radius = 1.035
        sx = radius * np.cos(a) * np.cos(b)
        sy = radius * np.cos(a) * np.sin(b)
        sz = radius * np.sin(a)

        moon.add_trace(go.Scatter3d(
            x=[sx], y=[sy], z=[sz],
            mode="markers+text",
            marker=dict(size=7, color=color),
            text=[name],
            textposition="top center",
            name=name
        ))

    moon.update_layout(
        height=450,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#10151c",
        font=dict(color="white"),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="#10151c",
            aspectmode="cube",
            camera=dict(eye=dict(
                x=0.7, y=0.7, z=-2.4
            ))
        )
    )

    st.plotly_chart(moon, use_container_width=True)

# LunaWindow: operational analysis + local-horizon sky view
if "data" in st.session_state:
    import plotly.graph_objects as go

    st.divider()
    st.header("Mission Operations Analyzer")
    st.caption(
        "Exploratory estimates from NASA Horizons samples "
        "and an idealized lunar horizon."
    )

    def joint_visibility_windows(site_df):
        p = site_df.pivot_table(
            index="UTC",
            columns="Body",
            values="Elevation (degrees)",
            aggfunc="mean"
        ).sort_index()

        if not {"Sun", "Earth"}.issubset(p.columns):
            return []

        p = p[["Sun", "Earth"]].dropna()
        windows = []

        for i in range(len(p) - 1):
            t0, t1 = p.index[i], p.index[i + 1]
            sun0, earth0 = p.iloc[i][["Sun", "Earth"]]
            sun1, earth1 = p.iloc[i + 1][["Sun", "Earth"]]

            cutoffs = [0.0, 1.0]

            for y0, y1 in [(sun0, sun1), (earth0, earth1)]:
                if y0 != y1:
                    f = -float(y0) / float(y1 - y0)
                    if 0 < f < 1:
                        cutoffs.append(f)

            cutoffs = sorted(set(cutoffs))

            for left, right in zip(cutoffs, cutoffs[1:]):
                mid = (left + right) / 2

                sun_mid = sun0 + (sun1 - sun0) * mid
                earth_mid = earth0 + (earth1 - earth0) * mid

                if sun_mid > 0 and earth_mid > 0:
                    begin = t0 + (t1 - t0) * left
                    finish = t0 + (t1 - t0) * right

                    if windows and abs(
                        begin - windows[-1][1]
                    ) < pd.Timedelta(seconds=1):
                        windows[-1] = (windows[-1][0], finish)
                    else:
                        windows.append((begin, finish))

        return windows

    all_windows = []
    comparison = []

    for site in ["A", "B"]:
        site_df = st.session_state.data[
            st.session_state.data["Site"] == site
        ]
        windows = joint_visibility_windows(site_df)

        for begin, finish in windows:
            duration = (finish - begin).total_seconds() / 3600
            all_windows.append({
                "Site": site,
                "Start (UTC)": begin,
                "End (UTC)": finish,
                "Hours": duration
            })

        total = sum(
            (end - start).total_seconds() / 3600
            for start, end in windows
        )

        longest = max(
            (
                (end - start).total_seconds() / 3600
                for start, end in windows
            ),
            default=0.0
        )

        comparison.append({
            "Site": site,
            "Joint hours": total,
            "Longest continuous window (h)": longest
        })

    c1, c2 = st.columns(2)

    for col, summary in zip([c1, c2], comparison):
        with col:
            st.subheader("Site " + summary["Site"])
            st.metric(
                "Estimated joint visibility",
                f"{summary['Joint hours']:.1f} h"
            )
            st.metric(
                "Longest window",
                f"{summary['Longest continuous window (h)']:.1f} h"
            )

    if all_windows:
        win_df = pd.DataFrame(all_windows).sort_values(
            ["Site", "Start (UTC)"]
        )

        st.subheader("Candidate operational windows")
        st.dataframe(
            win_df.assign(Hours=win_df["Hours"].round(2)),
            hide_index=True
        )

        st.download_button(
            "Export operational windows (CSV)",
            win_df.to_csv(index=False),
            "lunawindow_operational_windows.csv",
            mime="text/csv"
        )

        timeline = px.timeline(
            win_df,
            x_start="Start (UTC)",
            x_end="End (UTC)",
            y="Site",
            color="Site",
            title="Simultaneous geometric Sun and Earth visibility"
        )

        timeline.update_yaxes(autorange="reversed")
        timeline.update_layout(
            height=260,
            showlegend=False
        )

        
        # Show operational windows against the full analyzed period
        times = pd.to_datetime(
            st.session_state.data["UTC"], utc=True
        )
        
        timeline.update_xaxes(
            range=[times.min(), times.max()],
            title="Mission timeline (UTC)"
        )


        st.plotly_chart(timeline, use_container_width=True)

    else:
        st.info(
            "No simultaneous positive Sun/Earth elevation "
            "intervals in this sampled period."
        )

    winner = max(
        comparison,
        key=lambda row: row["Joint hours"]
    )

    if abs(
        comparison[0]["Joint hours"] -
        comparison[1]["Joint hours"]
    ) < 0.01:
        st.info(
            "For this limited metric, the sites are tied "
            "over the selected interval."
        )
    else:
        st.success(
            f"Site {winner['Site']} has more estimated "
            "simultaneous geometric Sun/Earth visibility. "
            "This is not a complete landing-site "
            "suitability ranking."
        )

    st.warning(
        "Windows use straight-line interpolation between "
        "sampled elevations. They are estimates, not exact "
        "rise/set times. Terrain, eclipses, solar-panel "
        "pointing, and radio link budgets are not included."
    )

    st.subheader("Local horizon sky view")

    if "Azimuth (degrees)" not in st.session_state.data.columns:
        st.info(
            "Add AZ to the Horizons query and click Analyze "
            "again to enable the sky view."
        )
    else:
        selected_site = st.selectbox(
            "Inspect site",
            ["A", "B"],
            key="sky_site"
        )

        sky_df = st.session_state.data[
            st.session_state.data["Site"] == selected_site
        ]

        moment_options = sorted(sky_df["UTC"].unique())

        selected_moment = st.select_slider(
            "UTC observation time",
            options=moment_options,
            format_func=lambda d: pd.Timestamp(d).strftime(
                "%b %d %H:%M UTC"
            ),
            key="sky_utc"
        )

        current = sky_df[
            sky_df["UTC"] == selected_moment
        ]

        fig = go.Figure()

        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
            annotation_text="Ideal horizon"
        )

        for body, symbol, color in [
            ("Sun", "circle", "#ffca65"),
            ("Earth", "diamond", "#73d6f4")
        ]:
            r = current[current["Body"] == body]

            if not r.empty:
                fig.add_trace(go.Scatter(
                    x=r["Azimuth (degrees)"],
                    y=r["Elevation (degrees)"],
                    mode="markers+text",
                    text=[body],
                    textposition="top center",
                    name=body,
                    marker=dict(
                        size=19,
                        symbol=symbol,
                        color=color
                    ),
                    hovertemplate=(
                        "Azimuth: %{x:.2f}°"
                        "<br>Elevation: %{y:.2f}°"
                        "<extra></extra>"
                    )
                ))

        fig.update_xaxes(
            title="Azimuth (degrees clockwise from north)",
            range=[0, 360]
        )

        fig.update_yaxes(
            title="Elevation above ideal horizon (degrees)"
        )

        fig.update_layout(
            height=380,
            title="Topocentric sky coordinates from Horizons"
        )

        st.plotly_chart(fig, use_container_width=True)
