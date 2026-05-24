# ============================================================
# DYNAMIC AI PRICING ENGINE
# MCA Data Science - Final Year Project
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="AI Pricing Engine",
    page_icon="💹",
    layout="wide"
)

# ── Custom CSS (makes cards look good in dark mode) ───────
st.markdown("""
<style>
.metric-box {
    background: rgba(128,128,128,0.08);
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 8px;
}
.metric-label { font-size: 13px; color: #888; margin-bottom: 4px; }
.metric-value { font-size: 26px; font-weight: 700; }
.result-box {
    background: rgba(29,158,117,0.1);
    border: 2px solid #1D9E75;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    margin-top: 12px;
}
.result-price { font-size: 52px; font-weight: 800; color: #1D9E75; }
.result-label { font-size: 14px; color: #1D9E75; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

def metric_card(label, value):
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# STEP 1 — LOAD & PREPARE DATA
# ============================================================
@st.cache_data(show_spinner="Loading data...")
def load_data():
    df = pd.read_csv("data/dataset.csv")

    # Parse dates
    df['Date']  = pd.to_datetime(df['Date'], dayfirst=True)
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter

    # New feature
    df['Revenue'] = df['Price'] * df['Units Sold']

    # Encode text columns to numbers
    encoders = {}
    for col in ['Category', 'Region', 'Seasonality', 'Weather Condition']:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders

# ============================================================
# STEP 2 — TRAIN ML MODEL
# ============================================================
@st.cache_resource(show_spinner="Training AI model...")
def train_model(df):
    FEATURES = [
        'Competitor Pricing',
        'Units Sold',
        'Discount',
        'Month',
        'Category_enc',
        'Region_enc',
        'Seasonality_enc'
    ]
    X = df[FEATURES]
    y = df['Price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        'R2':  round(r2_score(y_test, y_pred), 4),
        'MAE': round(mean_absolute_error(y_test, y_pred), 2)
    }
    return model, metrics, FEATURES

# ── Load everything ───────────────────────────────────────
df, encoders = load_data()
model, metrics, FEATURES = train_model(df)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 💹 AI Pricing Engine")
    st.caption("MCA Data Science · Final Year Project")
    st.divider()

    page = st.radio("Navigate", [
        "🏠 Overview",
        "📊 Data Analysis",
        "📉 Price Elasticity",
        "🤖 AI Price Predictor",
        "💰 Revenue Simulator"
    ])

    st.divider()
    st.markdown("**Dataset**")
    st.caption(f"Records : {len(df):,}")
    st.caption("Period  : 2022 – 2023")
    st.caption("Stores  : 5 | Products: 20")

    st.divider()
    st.markdown("**Model**")
    st.caption(f"R² Score : {metrics['R2']}")
    st.caption(f"Avg Error: ₹{metrics['MAE']}")

    st.divider()
    st.markdown("**Filters**")
    sel_cat = st.multiselect(
        "Category",
        df['Category'].unique(),
        default=list(df['Category'].unique())
    )
    sel_reg = st.multiselect(
        "Region",
        df['Region'].unique(),
        default=list(df['Region'].unique())
    )

# Apply filters
fdf = df[df['Category'].isin(sel_cat) & df['Region'].isin(sel_reg)]

# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================
if page == "🏠 Overview":
    st.title("🏠 Sales & Pricing Overview")
    st.caption(f"Showing {len(fdf):,} records · Jan 2022 – Dec 2023")
    st.divider()

    # KPI Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Revenue", f"₹{fdf['Revenue'].sum():,.0f}")
    with c2:
        metric_card("Average Price", f"₹{fdf['Price'].mean():.2f}")
    with c3:
        metric_card("Total Units Sold", f"{fdf['Units Sold'].sum():,}")
    with c4:
        top = fdf.groupby('Category')['Revenue'].sum().idxmax()
        metric_card("Top Category", top)

    st.divider()

    # Chart 1 — Monthly Revenue
    monthly = fdf.groupby(fdf['Date'].dt.to_period('M'))['Revenue'].sum().reset_index()
    monthly['Date'] = monthly['Date'].astype(str)
    fig1 = px.line(monthly, x='Date', y='Revenue',
                   title='Monthly Revenue Trend (2022–2023)',
                   markers=True, color_discrete_sequence=['#2E75B6'])
    fig1.update_layout(xaxis_title='Month', yaxis_title='Revenue (₹)',
                       hovermode='x unified')
    fig1.update_traces(line_width=2.5)

    # Chart 2 — Region Pie
    reg = fdf.groupby('Region')['Revenue'].sum().reset_index()
    fig2 = px.pie(reg, names='Region', values='Revenue',
                  title='Revenue Share by Region',
                  color_discrete_sequence=['#2E75B6','#1D9E75','#EF9F27','#D85A30'])
    fig2.update_traces(textposition='inside', textinfo='percent+label')

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)

    # Chart 3 & 4
    col3, col4 = st.columns(2)

    with col3:
        cat_units = fdf.groupby('Category')['Units Sold'].sum().reset_index()
        fig3 = px.bar(cat_units, x='Category', y='Units Sold',
                      title='Total Units Sold by Category',
                      color='Category',
                      color_discrete_sequence=['#2E75B6','#1D9E75','#EF9F27','#D85A30','#7F77DD'])
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        hol = fdf.groupby('Holiday/Promotion')['Revenue'].mean().reset_index()
        hol['Type'] = hol['Holiday/Promotion'].map({0:'Normal Day', 1:'Holiday/Promo'})
        fig4 = px.bar(hol, x='Type', y='Revenue',
                      title='Normal Day vs Holiday Revenue',
                      color='Type',
                      color_discrete_sequence=['#B5D4F4','#1D9E75'],
                      text_auto='.0f')
        fig4.update_layout(showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

# ============================================================
# PAGE 2 — DATA ANALYSIS
# ============================================================
elif page == "📊 Data Analysis":
    st.title("📊 Data Analysis")
    st.caption("Explore pricing, sales, and demand patterns")
    st.divider()

    tab1, tab2 = st.tabs(["Pricing Analysis", "Sales Analysis"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            # Our price vs competitor
            comp = fdf.groupby('Category')[['Price','Competitor Pricing']].mean().reset_index().round(2)
            fig5 = px.bar(comp, x='Category', y=['Price','Competitor Pricing'],
                          title='Our Price vs Competitor Price',
                          barmode='group',
                          color_discrete_sequence=['#2E75B6','#D85A30'])
            fig5.update_layout(yaxis_title='Avg Price (₹)')
            st.plotly_chart(fig5, use_container_width=True)

        with col2:
            # Price vs Units Sold scatter
            sample = fdf.sample(1500, random_state=42)
            fig6 = px.scatter(sample, x='Price', y='Units Sold',
                              color='Category',
                              title='Price vs Units Sold',
                              opacity=0.6,
                              color_discrete_sequence=['#2E75B6','#1D9E75','#EF9F27','#D85A30','#7F77DD'])
            fig6.update_layout(xaxis_title='Price (₹)', yaxis_title='Units Sold')
            st.plotly_chart(fig6, use_container_width=True)

        # Correlation heatmap
        num_cols = ['Price','Competitor Pricing','Units Sold',
                    'Demand Forecast','Inventory Level','Discount','Revenue']
        corr = fdf[num_cols].corr().round(2)
        fig7 = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale='RdBu', zmid=0,
            text=corr.values, texttemplate='%{text}',
            textfont={"size": 10}))
        fig7.update_layout(title='Correlation Matrix — How Features Relate to Each Other')
        st.plotly_chart(fig7, use_container_width=True)

    with tab2:
        col3, col4 = st.columns(2)

        with col3:
            disc = fdf.groupby('Discount')['Units Sold'].mean().reset_index()
            fig8 = px.bar(disc, x='Discount', y='Units Sold',
                          title='Avg Units Sold by Discount Level',
                          color='Units Sold',
                          color_continuous_scale='Teal',
                          text_auto='.0f')
            fig8.update_layout(xaxis_title='Discount (%)', yaxis_title='Avg Units Sold')
            st.plotly_chart(fig8, use_container_width=True)

        with col4:
            season = fdf.groupby('Seasonality')['Units Sold'].mean().reset_index()
            fig9 = px.bar(season, x='Seasonality', y='Units Sold',
                          title='Avg Units Sold by Season',
                          color='Seasonality',
                          color_discrete_sequence=['#EF9F27','#2E75B6','#1D9E75','#D85A30'])
            fig9.update_layout(showlegend=False)
            st.plotly_chart(fig9, use_container_width=True)

# ============================================================
# PAGE 3 — PRICE ELASTICITY
# ============================================================
elif page == "📉 Price Elasticity":
    st.title("📉 Price Elasticity Analysis")
    st.info("""
    **What is Price Elasticity?**
    It measures how much sales change when price changes.
    - **Elasticity < -1** → Elastic (customers are price sensitive — small price rise = big sales drop)
    - **-1 to 0** → Inelastic (customers are NOT very sensitive — price rise = small sales drop)
    """)
    st.divider()

    # Calculate elasticity for each category
    def get_elasticity(data, group_col):
        results = []
        for group, gdf in data.groupby(group_col):
            gdf = gdf.sort_values('Price')
            if len(gdf) < 10:
                continue
            pct_price = gdf['Price'].pct_change().replace([np.inf, -np.inf], np.nan)
            pct_qty   = gdf['Units Sold'].pct_change().replace([np.inf, -np.inf], np.nan)
            e = (pct_qty / pct_price).replace([np.inf, -np.inf], np.nan).dropna().median()
            results.append({group_col: group, 'Elasticity': round(e, 2)})
        return pd.DataFrame(results).sort_values('Elasticity')

    # Elasticity by Category
    cat_e = get_elasticity(fdf, 'Category')
    cat_e['Type'] = cat_e['Elasticity'].apply(
        lambda e: 'Elastic (Price Sensitive)' if e < -1 else
                  'Inelastic (Not Sensitive)' if e < 0 else 'Unusual')

    st.subheader("Elasticity by Category")
    st.dataframe(cat_e.rename(columns={'Category': 'Product Category'}),
                 use_container_width=True, hide_index=True)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        colors = ['#D85A30' if e < -1 else '#1D9E75' for e in cat_e['Elasticity']]
        fig10 = go.Figure(go.Bar(
            x=cat_e['Elasticity'], y=cat_e['Category'],
            orientation='h', marker_color=colors,
            text=cat_e['Elasticity'], textposition='outside'))
        fig10.add_vline(x=-1, line_dash='dash', line_color='gray',
                        annotation_text='Threshold = -1')
        fig10.update_layout(title='Price Elasticity by Category',
                             xaxis_title='Elasticity Value',
                             xaxis=dict(range=[-4, 2]))
        st.plotly_chart(fig10, use_container_width=True)

    with col2:
        reg_e = get_elasticity(fdf, 'Region')
        fig11 = px.bar(reg_e, x='Region', y='Elasticity',
                       color='Elasticity',
                       color_continuous_scale='RdYlGn',
                       title='Price Elasticity by Region',
                       text='Elasticity')
        fig11.add_hline(y=-1, line_dash='dash', line_color='gray')
        fig11.update_traces(textposition='outside')
        st.plotly_chart(fig11, use_container_width=True)

    # Season elasticity
    sea_e = get_elasticity(fdf, 'Seasonality')
    fig12 = px.bar(sea_e, x='Seasonality', y='Elasticity',
                   color='Seasonality',
                   title='Price Elasticity by Season',
                   color_discrete_sequence=['#EF9F27','#2E75B6','#1D9E75','#D85A30'],
                   text='Elasticity')
    fig12.add_hline(y=-1, line_dash='dash', line_color='gray')
    fig12.update_traces(textposition='outside')
    fig12.update_layout(showlegend=False)
    st.plotly_chart(fig12, use_container_width=True)

# ============================================================
# PAGE 4 — AI PRICE PREDICTOR
# ============================================================
elif page == "🤖 AI Price Predictor":
    st.title("🤖 AI Price Predictor")
    st.caption("Enter product details — Random Forest model predicts the best price")
    st.divider()

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Input Parameters")

        category = st.selectbox("Product Category", df['Category'].unique())
        region   = st.selectbox("Region", df['Region'].unique())
        season   = st.selectbox("Season", df['Seasonality'].unique())

        competitor_price = st.slider("Competitor Price (₹)", 10.0, 105.0, 55.0, 0.5)
        units_sold       = st.slider("Expected Units Sold", 0, 500, 120)
        discount         = st.selectbox("Discount (%)", [0, 5, 10, 15, 20], index=2)
        month            = st.slider("Month (1=Jan, 12=Dec)", 1, 12, 6)

        predict_btn = st.button(
            "🎯 Predict Optimal Price",
            use_container_width=True,
            type="primary"
        )

    with right:
        st.subheader("Prediction Result")

        if predict_btn:
            # Convert text to numbers using encoders
            cat_enc = encoders['Category'].transform([category])[0]
            reg_enc = encoders['Region'].transform([region])[0]
            sea_enc = encoders['Seasonality'].transform([season])[0]

            # Build input row
            input_row = pd.DataFrame([{
                'Competitor Pricing': competitor_price,
                'Units Sold':         units_sold,
                'Discount':           discount,
                'Month':              month,
                'Category_enc':       cat_enc,
                'Region_enc':         reg_enc,
                'Seasonality_enc':    sea_enc
            }])

            # Predict
            predicted = round(model.predict(input_row)[0], 2)

            # Extra calculations
            gap       = round(predicted - competitor_price, 2)
            direction = "above" if gap > 0 else "below"
            eff_price = round(predicted * (1 - discount / 100), 2)
            est_rev   = round(eff_price * units_sold, 0)

            # Big green result box
            st.markdown(f"""
            <div class="result-box">
                <div class="result-label">Recommended Optimal Price</div>
                <div class="result-price">₹{predicted}</div>
            </div>
            """, unsafe_allow_html=True)

            st.divider()

            # 3 stat cards
            m1, m2, m3 = st.columns(3)
            with m1:
                metric_card("After Discount", f"₹{eff_price}")
            with m2:
                metric_card("Est. Revenue", f"₹{est_rev:,.0f}")
            with m3:
                metric_card("vs Competitor", f"₹{abs(gap):.2f} {direction}")

            st.divider()

            # Feature Importance Chart
            st.subheader("📊 What Influenced This Prediction?")
            importance_df = pd.DataFrame({
                'Feature':    FEATURES,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)

            fig_imp = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Feature Importance — Which inputs matter most?',
                color='Importance',
                color_continuous_scale='Blues'
            )
            fig_imp.update_layout(xaxis_title='Importance Score', yaxis_title='')
            st.plotly_chart(fig_imp, use_container_width=True)

        else:
            st.info("👈 Fill the form and click **Predict Optimal Price**")
            st.divider()
            st.markdown("**How the model works:**")
            st.markdown("- Algorithm: **Random Forest** (100 Decision Trees)")
            st.markdown("- Training rows: **58,480** (80% of dataset)")
            st.markdown("- Test rows: **14,620** (20% of dataset)")
            st.markdown(f"- R² Score: **{metrics['R2']}** ← 98%+ accuracy")
            st.markdown(f"- Avg Error: **₹{metrics['MAE']}** ← very low error")

# ============================================================
# PAGE 5 — REVENUE SIMULATOR
# ============================================================
elif page == "💰 Revenue Simulator":
    st.title("💰 Revenue Simulator")
    st.caption("Find the price that gives maximum revenue")
    st.divider()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Settings")

        base_price    = st.slider("Current Price (₹)", 10.0, 100.0, 55.0, 1.0)
        base_units    = st.slider("Current Units Sold", 10, 500, 130)
        sim_discount  = st.selectbox("Discount (%)", [0, 5, 10, 15, 20], index=2)
        elasticity    = st.slider("Price Elasticity",
                                  -3.0, -0.1, -1.5, 0.1,
                                  help="How sensitive are customers to price? -1.5 is a good default")

        st.divider()
        st.markdown("**Test a Specific Price**")
        test_price = st.slider("What if I set price to (₹)", 10.0, 120.0, base_price, 1.0)

        # Calculate what-if result
        pct_change   = (test_price - base_price) / max(base_price, 1)
        test_units   = max(0, round(base_units * (1 + elasticity * pct_change)))
        test_eff     = round(test_price * (1 - sim_discount / 100), 2)
        test_revenue = round(test_eff * test_units, 0)

        st.divider()
        metric_card("Est. Units Sold", f"{test_units:,}")
        metric_card("Est. Revenue", f"₹{test_revenue:,.0f}")

    with col2:
        # Simulate revenue across 80 price points
        prices    = np.linspace(max(5, base_price * 0.5), base_price * 1.8, 80)
        revenues  = []
        units_list= []

        for p in prices:
            pct = (p - base_price) / max(base_price, 1)
            u   = max(0, round(base_units * (1 + elasticity * pct)))
            eff = p * (1 - sim_discount / 100)
            rev = round(eff * u, 2)
            revenues.append(rev)
            units_list.append(u)

        sim_df = pd.DataFrame({
            'Price': prices.round(2),
            'Revenue': revenues,
            'Units': units_list
        })

        # Find optimal price
        best_idx = sim_df['Revenue'].idxmax()
        best     = sim_df.loc[best_idx]

        # Draw revenue curve
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(
            x=sim_df['Price'], y=sim_df['Revenue'],
            mode='lines', name='Revenue Curve',
            line=dict(color='#2E75B6', width=2.5)))
        fig_sim.add_trace(go.Scatter(
            x=[best['Price']], y=[best['Revenue']],
            mode='markers+text', name='Optimal Price',
            marker=dict(color='#1D9E75', size=14, symbol='star'),
            text=[f"  ₹{best['Price']:.2f}"],
            textposition='middle right'))
        fig_sim.add_vline(
            x=base_price,
            line_dash='dash', line_color='#D85A30',
            annotation_text=f'Current: ₹{base_price}',
            annotation_position='top right')
        fig_sim.update_layout(
            title='Revenue Simulation Curve',
            xaxis_title='Price (₹)',
            yaxis_title='Estimated Revenue (₹)',
            hovermode='x unified')
        st.plotly_chart(fig_sim, use_container_width=True)

        # 3 optimal stats
        o1, o2, o3 = st.columns(3)
        with o1:
            metric_card("Optimal Price", f"₹{best['Price']:.2f}")
        with o2:
            metric_card("Units at Optimal", f"{int(best['Units']):,}")
        with o3:
            metric_card("Max Revenue", f"₹{best['Revenue']:,.0f}")

        st.divider()

        # Discount × Elasticity Matrix
        st.subheader("Revenue Matrix — Discount × Elasticity")
        discounts    = [0, 5, 10, 15, 20]
        elasticities = [-0.8, -1.2, -1.8, -2.5]
        z = []
        for e in elasticities:
            row = []
            for d in discounts:
                ps  = np.linspace(base_price * 0.7, base_price * 1.3, 30)
                revs= []
                for p in ps:
                    pct = (p - base_price) / max(base_price, 1)
                    u   = max(0, round(base_units * (1 + e * pct)))
                    revs.append(p * (1 - d/100) * u)
                row.append(round(max(revs), 0))
            z.append(row)

        fig_matrix = go.Figure(data=go.Heatmap(
            z=z,
            x=[f'{d}%' for d in discounts],
            y=[f'e={e}' for e in elasticities],
            colorscale='Greens',
            text=[[f'₹{v:,.0f}' for v in row] for row in z],
            texttemplate='%{text}'))
        fig_matrix.update_layout(
            xaxis_title='Discount Level',
            yaxis_title='Elasticity')
        st.plotly_chart(fig_matrix, use_container_width=True)
