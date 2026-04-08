import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
import os
from datetime import datetime, timedelta

# ── Config ────────────────────────────────────────────
# Works both locally and on Streamlit Cloud
API = os.getenv("API_URL", "http://localhost:8000")
try:
    API = st.secrets["API_URL"]
except Exception:
    pass

try:
    DB_CONN = {
        "host":     st.secrets["DB_HOST"],
        "port":     int(st.secrets["DB_PORT"]),
        "database": st.secrets["DB_NAME"],
        "user":     st.secrets["DB_USER"],
        "password": st.secrets["DB_PASSWORD"],
    }
except Exception:
    DB_CONN = {
        "host": "localhost", "port": 5432,
        "database": "retail_db",
        "user": "retail_user", "password": "retail_pass"
    }

DATASET_START = datetime(2013, 1, 1).date()
DATASET_END   = datetime(2015, 7, 31).date()

st.set_page_config(
    page_title="Retail Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="metric-container"] {
    background: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 10px;
    padding: 12px;
}
.alert-critical { background:#3d1f1f; border-left:4px solid #ff4444; padding:8px; border-radius:4px; }
.alert-warning  { background:#3d2e1f; border-left:4px solid #ffaa44; padding:8px; border-radius:4px; }
.alert-ok       { background:#1f3d2e; border-left:4px solid #44ff88; padding:8px; border-radius:4px; }
</style>
""", unsafe_allow_html=True)

# ── Cached fetchers ───────────────────────────────────
@st.cache_data(ttl=60)
def get_summary():
    return requests.get(f"{API}/sales-summary", timeout=60).json()

@st.cache_data(ttl=60)
def get_anomalies(limit):
    return requests.get(f"{API}/get-anomalies",
                        params={"limit": limit}, timeout=60).json()

@st.cache_data(ttl=60)
def get_forecast(product_id, store_id):
    return requests.get(f"{API}/predict-demand",
                        params={"product_id": product_id,
                                "store_id": store_id}, timeout=60).json()


@st.cache_data(ttl=60)
def get_90day_forecast(product_id, store_id):
    return requests.get(f"{API}/predict-90-days",
                        params={"product_id": product_id,
                                "store_id": store_id}, timeout=120).json()

@st.cache_data(ttl=60)
def get_inventory(store_id):
    return requests.get(f"{API}/get-inventory-recommendations",
                        params={"store_id": store_id}, timeout=60).json()

@st.cache_data(ttl=120)
def get_sales_trend(store_id, start_date, end_date):
    try:
        conn = psycopg2.connect(**DB_CONN)
        df   = pd.read_sql("""
            SELECT sale_date, SUM(total_quantity) AS total_sales
            FROM aggregated_sales
            WHERE store_id = %s
              AND sale_date BETWEEN %s AND %s
            GROUP BY sale_date
            ORDER BY sale_date
        """, conn, params=(store_id, str(start_date), str(end_date)))
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=120)
def get_top_stores(n=5):
    try:
        conn = psycopg2.connect(**DB_CONN)
        df   = pd.read_sql("""
            SELECT
                store_id,
                SUM(total_quantity)  AS total_sales,
                AVG(total_quantity)  AS avg_daily_sales,
                COUNT(*)             AS total_days
            FROM aggregated_sales
            GROUP BY store_id
            ORDER BY total_sales DESC
            LIMIT %s
        """, conn, params=(n,))
        conn.close()
        df["store_label"] = "Store " + df["store_id"].astype(str)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=120)
def get_actual_data(store_id):
    try:
        conn = psycopg2.connect(**DB_CONN)
        df   = pd.read_sql("""
            SELECT sale_date, total_quantity
            FROM aggregated_sales
            WHERE store_id = %s
              AND sale_date >= '2015-05-01'
              AND sale_date <= '2015-07-31'
            ORDER BY sale_date ASC
        """, conn, params=(store_id,))
        conn.close()
        df["sale_date"]     = pd.to_datetime(df["sale_date"])
        df["total_quantity"] = df["total_quantity"].astype(float)
        return df
    except Exception:
        return pd.DataFrame()

# ── Sidebar ───────────────────────────────────────────
with st.sidebar:
    st.title("🛒 Retail Intelligence")
    st.markdown("---")
    st.header("🔧 Controls")
    store_id      = st.slider("Store ID",   1, 50, 1)
    product_id    = st.slider("Product ID", 1, 50, 1)
    anomaly_limit = st.slider("Anomalies to show", 5, 50, 10)

    st.markdown("---")
    st.header("📅 Date Range")
    start_date = st.date_input("From",
                               datetime(2015, 1, 1).date(),
                               min_value=DATASET_START,
                               max_value=DATASET_END)
    end_date   = st.date_input("To",
                               DATASET_END,
                               min_value=DATASET_START,
                               max_value=DATASET_END)
    st.caption(f"Dataset: {DATASET_START} → {DATASET_END}")

    st.markdown("---")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    st.caption("Retail Intelligence Platform v2.0")

# ── Header ────────────────────────────────────────────
st.title("🛒 Retail Intelligence Dashboard")
st.markdown("Real-time demand forecasting · Anomaly detection · Inventory optimization")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview", "🔮 Forecasting", "🚨 Anomalies", "📦 Inventory"
])

# ════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════
with tab1:
    with st.spinner("Loading summary..."):
        try:
            summary    = get_summary()
            anom_res   = get_anomalies(100)
            anom_count = anom_res.get("total", 0)

            c1,c2,c3,c4,c5,c6 = st.columns(6)
            c1.metric("Total Records",  f"{summary.get('total_records',0):,}")
            c2.metric("Total Units",    f"{summary.get('total_units',0):,.0f}")
            c3.metric("Total Stores",   summary.get('total_stores','N/A'))
            c4.metric("Total Products", summary.get('total_products','N/A'))
            c5.metric("Latest Date",    summary.get('latest_date','N/A'))
            c6.metric("🚨 Anomalies",   anom_count,
                      delta="⚠️ Review" if anom_count > 5 else "✅ Normal",
                      delta_color="inverse")
        except Exception as e:
            st.error(f"API unavailable: {e}")
            st.stop()

    st.divider()
    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.subheader("📈 Sales Trend")
        df_sales = get_sales_trend(store_id, start_date, end_date)
        if not df_sales.empty:
            fig = px.area(df_sales, x="sale_date", y="total_sales",
                          title=f"Daily Sales — Store {store_id}",
                          labels={"sale_date":"Date","total_sales":"Units Sold"})
            fig.update_traces(line_color="#534AB7",
                              fillcolor="rgba(83,74,183,0.15)")
            fig.update_layout(hovermode="x unified",
                              xaxis=dict(rangeslider=dict(visible=True)))
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("⬇️ Download Sales Data",
                               df_sales.to_csv(index=False),
                               f"sales_store_{store_id}.csv", "text/csv")
        else:
            st.info("No sales data for selected range.")

    with col_r:
        st.subheader("🏆 Top 5 Stores by Volume")
        df_top = get_top_stores(n=5)
        if not df_top.empty:
            fig2 = px.bar(df_top, x="total_sales", y="store_label",
                          orientation="h",
                          title="Top 5 Stores by Total Sales",
                          labels={"total_sales":"Total Units Sold",
                                  "store_label":"Store"},
                          color="avg_daily_sales",
                          color_continuous_scale="Teal",
                          text="total_sales")
            fig2.update_traces(texttemplate="%{text:.2s}",
                               textposition="outside")
            fig2.update_layout(yaxis=dict(categoryorder="total ascending"),
                               coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

            df_disp = df_top[["store_label","total_sales",
                               "avg_daily_sales","total_days"]].copy()
            df_disp["total_sales"]     = df_disp["total_sales"].apply(
                lambda x: f"{x:,.0f}")
            df_disp["avg_daily_sales"] = df_disp["avg_daily_sales"].apply(
                lambda x: f"{x:,.1f}")
            df_disp.columns = ["Store","Total Units",
                               "Avg Daily Sales","Days Active"]
            st.dataframe(df_disp, use_container_width=True, hide_index=True)
        else:
            st.info("No store data available.")

# ════════════════════════════════════════════════════
# TAB 2 — FORECASTING
# ════════════════════════════════════════════════════
with tab2:
    st.subheader(f"🔮 Demand Forecast — Store {store_id} | Product {product_id}")
    col_metric, col_chart = st.columns([1, 2])

    with col_metric:
        with st.spinner("Running forecast..."):
            try:
                res  = get_forecast(product_id, store_id)
                pred = res.get("predicted_demand", 0)
                lat  = res.get("latency_ms", 0)
                st.metric("Tomorrow's Demand", f"{pred:,.2f} units")
                st.metric("API Latency", f"{lat} ms")
                st.markdown("""
                <div class="alert-ok">
                ✅ Model: LSTM v2 | Features: 8 | Seq: 14 days
                </div>""", unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Forecast error: {e}")

    with col_chart:
        with st.spinner("Generating 3-month forecast..."):
            try:
                res90 = get_90day_forecast(product_id, store_id)
                fc    = res90.get("daily_forecast", [])
                monthly = res90.get("monthly_summary", [])
                weekly  = res90.get("weekly_summary", [])

                if not fc:
                    st.warning("No forecast data returned.")
                else:
                    # Build forecast df
                    df_fc     = pd.DataFrame(fc)
                    fc_dates  = df_fc["date"].tolist()
                    fc_values = df_fc["predicted_demand"].astype(float).tolist()

                    # Build actual df
                    df_act    = get_actual_data(store_id)
                    act_dates  = df_act["sale_date"].dt.strftime(
                        "%Y-%m-%d").tolist() if not df_act.empty else []
                    act_values = df_act["total_quantity"].tolist() \
                        if not df_act.empty else []

                    # Confidence band — widens over time
                    upper = [v * (1 + 0.002 * i) * 1.15
                             for i, v in enumerate(fc_values)]
                    lower = [v * (1 - 0.002 * i) * 0.85
                             for i, v in enumerate(fc_values)]
                    band_x = fc_dates + fc_dates[::-1]
                    band_y = upper + lower[::-1]

                    # Tabs for different views
                    view_tab1, view_tab2, view_tab3 = st.tabs([
                        "📅 Daily", "📆 Weekly", "🗓️ Monthly"
                    ])

                    with view_tab1:
                        fig3 = go.Figure()

                        # Actual
                        if act_dates:
                            fig3.add_trace(go.Scatter(
                                x=act_dates, y=act_values,
                                name="Actual (May–Jul 2015)",
                                line=dict(color="#1D9E75", width=2),
                                mode="lines+markers",
                                marker=dict(size=3)
                            ))

                        # Forecast
                        fig3.add_trace(go.Scatter(
                            x=fc_dates, y=fc_values,
                            name="90-Day Forecast",
                            line=dict(color="#FF6B6B",
                                      width=2, dash="dash"),
                            mode="lines"
                        ))

                        # Confidence band
                        fig3.add_trace(go.Scatter(
                            x=band_x, y=band_y,
                            fill="toself",
                            fillcolor="rgba(255,107,107,0.10)",
                            line=dict(color="rgba(255,107,107,0)"),
                            name="Confidence Band"
                        ))

                        # Divider
                        fig3.add_shape(
                            type="line",
                            x0="2015-07-31", x1="2015-07-31",
                            y0=0, y1=1,
                            xref="x", yref="paper",
                            line=dict(color="gray",
                                      dash="dot", width=1.5)
                        )
                        fig3.add_annotation(
                            x="2015-07-31", y=1,
                            xref="x", yref="paper",
                            text="Forecast →",
                            showarrow=False,
                            font=dict(color="gray", size=11),
                            xanchor="left"
                        )

                        fig3.update_layout(
                            title=f"3-Month Daily Forecast — Store {store_id}",
                            xaxis_title="Date",
                            yaxis_title="Units Sold",
                            hovermode="x unified",
                            legend=dict(orientation="h", y=-0.25),
                            xaxis=dict(
                                range=["2015-05-01", "2015-10-31"],
                                tickformat="%b %d %Y",
                                type="date"
                            ),
                            yaxis=dict(rangemode="tozero")
                        )
                        st.plotly_chart(fig3, use_container_width=True)

                    with view_tab2:
                        if weekly:
                            df_weekly = pd.DataFrame(weekly)
                            fig_w = px.bar(
                                df_weekly,
                                x="week",
                                y="avg_demand",
                                title=f"Weekly Avg Demand — Store {store_id}",
                                labels={"week": "Week",
                                        "avg_demand": "Avg Daily Demand"},
                                color="avg_demand",
                                color_continuous_scale="Teal",
                                text="avg_demand"
                            )
                            fig_w.update_traces(
                                texttemplate="%{text:,.0f}",
                                textposition="outside"
                            )
                            st.plotly_chart(fig_w,
                                            use_container_width=True)
                            st.dataframe(df_weekly,
                                         use_container_width=True,
                                         hide_index=True)

                    with view_tab3:
                        if monthly:
                            df_monthly = pd.DataFrame(monthly)
                            fig_m = px.bar(
                                df_monthly,
                                x="month",
                                y=["avg_demand", "total_demand"],
                                title=f"Monthly Forecast — Store {store_id}",
                                labels={"month": "Month",
                                        "value": "Units",
                                        "variable": "Metric"},
                                barmode="group",
                                color_discrete_map={
                                    "avg_demand":   "#1D9E75",
                                    "total_demand": "#534AB7"
                                }
                            )
                            st.plotly_chart(fig_m,
                                            use_container_width=True)

                            # Monthly summary table
                            df_monthly["avg_demand"] = df_monthly[
                                "avg_demand"].apply(lambda x: f"{x:,.2f}")
                            df_monthly["total_demand"] = df_monthly[
                                "total_demand"].apply(lambda x: f"{x:,.2f}")
                            df_monthly.columns = [
                                "Month", "Start Date", "End Date",
                                "Avg Daily Demand", "Total Demand"
                            ]
                            st.dataframe(df_monthly,
                                         use_container_width=True,
                                         hide_index=True)

                    # Download full forecast
                    st.download_button(
                        "⬇️ Download 90-Day Forecast",
                        df_fc.to_csv(index=False),
                        f"forecast_90day_store{store_id}.csv",
                        "text/csv"
                    )

            except Exception as e:
                st.warning(f"Forecast error: {e}")
# ════════════════════════════════════════════════════
# TAB 3 — ANOMALIES
# ════════════════════════════════════════════════════
with tab3:
    st.subheader("🚨 Anomaly Detection")
    with st.spinner("Loading anomalies..."):
        try:
            res       = get_anomalies(anomaly_limit)
            anomalies = res.get("anomalies", [])

            if anomalies:
                df_anom = pd.DataFrame(anomalies)

                def severity(z):
                    if z is None: return "Unknown"
                    if abs(z) > 3: return "🔴 Critical"
                    if abs(z) > 2: return "🟠 High"
                    return "🟡 Medium"

                df_anom["severity"] = df_anom["z_score"].apply(severity)
                critical = df_anom[df_anom["severity"] == "🔴 Critical"]
                high     = df_anom[df_anom["severity"] == "🟠 High"]

                if len(critical) > 0:
                    st.markdown(f"""<div class="alert-critical">
                    🔴 <b>{len(critical)} Critical anomalies</b> — immediate attention!
                    </div>""", unsafe_allow_html=True)
                if len(high) > 0:
                    st.markdown(f"""<div class="alert-warning">
                    🟠 <b>{len(high)} High severity anomalies</b> detected
                    </div>""", unsafe_allow_html=True)

                st.markdown("")
                col_a, col_b = st.columns(2)

                with col_a:
                    st.dataframe(
                        df_anom[["store_id","product_id","z_score",
                                 "severity","detected_at"]],
                        use_container_width=True, height=350)
                    st.download_button("⬇️ Download Anomalies",
                                       df_anom.to_csv(index=False),
                                       "anomalies.csv", "text/csv")

                with col_b:
                    fig4 = px.bar(
                        df_anom, x="product_id", y="z_score",
                        color="z_score",
                        color_continuous_scale="RdYlGn_r",
                        title="Z-Score by Product",
                        labels={"z_score":"Z-Score","product_id":"Product"},
                        hover_data=["store_id","severity"]
                    )
                    fig4.add_hline(y=2, line_dash="dash",
                                   line_color="orange",
                                   annotation_text="High (2σ)")
                    fig4.add_hline(y=3, line_dash="dash",
                                   line_color="red",
                                   annotation_text="Critical (3σ)")
                    st.plotly_chart(fig4, use_container_width=True)

                # Anomaly map — OUTSIDE columns, full width
                st.subheader("Store vs Product Anomaly Map")
                df_map = df_anom.copy()
                df_map["store_label"]   = "Store "   + df_map["store_id"].astype(str)
                df_map["product_label"] = "Product " + df_map["product_id"].astype(str)
                df_map["abs_z"]         = df_map["z_score"].abs()

                fig5 = px.scatter(
                    df_map,
                    x="store_label",
                    y="product_label",
                    size="abs_z",
                    color="z_score",
                    color_continuous_scale="RdYlGn_r",
                    title="Anomaly Map — Bubble size = Z-Score severity",
                    labels={
                        "store_label":   "Store",
                        "product_label": "Product",
                        "z_score":       "Z-Score",
                        "abs_z":         "Severity"
                    },
                    hover_data={
                        "store_label":   True,
                        "product_label": True,
                        "z_score":       ":.3f",
                        "severity":      True,
                        "abs_z":         False
                    },
                    size_max=40
                )
                fig5.update_layout(
                    xaxis_title="Store",
                    yaxis_title="Product",
                    xaxis=dict(categoryorder="category ascending"),
                    yaxis=dict(categoryorder="category ascending"),
                    height=500
                )
                st.plotly_chart(fig5, use_container_width=True)

                # Summary table
                st.markdown("**Anomaly Summary by Store**")
                store_summary = df_anom.groupby("store_id").agg(
                    anomaly_count=("z_score", "count"),
                    avg_z_score=("z_score",   "mean"),
                    max_z_score=("z_score",   "max")
                ).reset_index()
                store_summary["store_id"]    = "Store " + store_summary["store_id"].astype(str)
                store_summary["avg_z_score"] = store_summary["avg_z_score"].round(3)
                store_summary["max_z_score"] = store_summary["max_z_score"].round(3)
                store_summary.columns = ["Store","Anomaly Count",
                                         "Avg Z-Score","Max Z-Score"]
                st.dataframe(store_summary, use_container_width=True,
                             hide_index=True)

            else:
                st.markdown("""<div class="alert-ok">
                ✅ No anomalies detected.</div>""",
                            unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Anomaly data unavailable: {e}")
# ════════════════════════════════════════════════════
# TAB 4 — INVENTORY
# ════════════════════════════════════════════════════
with tab4:
    st.subheader(f"📦 Inventory Recommendations — Store {store_id}")
    with st.spinner("Loading inventory data..."):
        try:
            res  = get_inventory(store_id)
            recs = res.get("recommendations", [])

            if recs:
                df_rec = pd.DataFrame(recs)

                # Fix column types
                df_rec["current_demand"] = df_rec["current_demand"].astype(float)
                df_rec["rolling_mean"]   = df_rec["rolling_mean"].astype(float)
                df_rec["reorder_point"]  = df_rec["reorder_point"].astype(float)
                df_rec["safety_stock"]   = df_rec["safety_stock"].astype(float)

                reorder_needed = df_rec[df_rec["status"] == "⚠️ Reorder Soon"]

                # KPI metrics
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Days of Data",    len(df_rec))
                c2.metric("Need Reorder",    len(reorder_needed),
                          delta="⚠️ Action needed" if len(reorder_needed) > 0 else "✅ All OK",
                          delta_color="inverse")
                c3.metric("Avg Daily Demand",
                          f"{df_rec['current_demand'].mean():,.0f}")
                c4.metric("Avg Safety Stock",
                          f"{df_rec['safety_stock'].mean():,.0f}")

                if len(reorder_needed) > 0:
                    st.markdown(f"""
                    <div class="alert-warning">
                    ⚠️ <b>{len(reorder_needed)} days</b> show demand below
                    reorder threshold for store {store_id}
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="alert-ok">
                    ✅ Inventory levels are healthy across all periods
                    </div>""", unsafe_allow_html=True)

                st.markdown("")
                col_l, col_r = st.columns([1, 1])

                with col_l:
                    st.markdown("**Daily Inventory Status (Last 20 Days)**")
                    df_display = df_rec[[
                        "sale_date", "current_demand",
                        "rolling_mean", "reorder_point",
                        "safety_stock", "status"
                    ]].copy()
                    df_display.columns = [
                        "Date", "Demand", "7-Day Avg",
                        "Reorder Point", "Safety Stock", "Status"
                    ]
                    st.dataframe(df_display,
                                 use_container_width=True,
                                 height=400)
                    st.download_button(
                        "⬇️ Download Recommendations",
                        df_rec.to_csv(index=False),
                        f"inventory_store_{store_id}.csv",
                        "text/csv"
                    )

                with col_r:
                    # Demand vs reorder point over time
                    fig6 = go.Figure()

                    fig6.add_trace(go.Scatter(
                        x=df_rec["sale_date"],
                        y=df_rec["current_demand"],
                        name="Daily Demand",
                        line=dict(color="#1D9E75", width=2),
                        mode="lines+markers",
                        marker=dict(size=4)
                    ))

                    fig6.add_trace(go.Scatter(
                        x=df_rec["sale_date"],
                        y=df_rec["rolling_mean"],
                        name="7-Day Rolling Mean",
                        line=dict(color="#534AB7", width=2, dash="dot"),
                        mode="lines"
                    ))

                    fig6.add_trace(go.Scatter(
                        x=df_rec["sale_date"],
                        y=df_rec["safety_stock"],
                        name="Safety Stock",
                        line=dict(color="#FF6B6B", width=1.5, dash="dash"),
                        mode="lines"
                    ))

                    # Highlight reorder days
                    reorder_days = df_rec[df_rec["status"] == "⚠️ Reorder Soon"]
                    if not reorder_days.empty:
                        fig6.add_trace(go.Scatter(
                            x=reorder_days["sale_date"],
                            y=reorder_days["current_demand"],
                            name="⚠️ Reorder Needed",
                            mode="markers",
                            marker=dict(
                                color="red", size=10,
                                symbol="triangle-down"
                            )
                        ))

                    fig6.update_layout(
                        title=f"Demand vs Safety Stock — Store {store_id}",
                        xaxis_title="Date",
                        yaxis_title="Units",
                        hovermode="x unified",
                        legend=dict(orientation="h", y=-0.3),
                        yaxis=dict(rangemode="tozero")
                    )
                    st.plotly_chart(fig6, use_container_width=True)

            else:
                st.info(f"No inventory data for store {store_id}.")
        except Exception as e:
            st.warning(f"Inventory data unavailable: {e}")

st.divider()
st.caption("Retail Intelligence Platform v2.0 | "
           "Kafka + PostgreSQL + Airflow + LSTM + FastAPI + Streamlit")