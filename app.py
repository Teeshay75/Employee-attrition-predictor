# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="AttritionIQ — Employee Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #080d1a;
    color: #e2e8f0;
}

.stApp { background-color: #080d1a; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #0a1020 100%);
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] .stRadio label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    color: #94a3b8;
}

/* ── Headings ── */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0f1e38 0%, #0d1a30 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 20px;
}
[data-testid="metric-container"] label {
    color: #64748b !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f0f4ff !important;
    font-family: 'Syne', sans-serif;
    font-size: 2rem !important;
    font-weight: 700 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 2rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 0.9rem;
    letter-spacing: 0.03em;
    transition: all 0.2s ease;
    box-shadow: 0 4px 15px rgba(29, 78, 216, 0.3);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(29, 78, 216, 0.4);
}

/* ── Inputs ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background-color: #0f1e38 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
.stSlider > div > div > div { background: #1d4ed8 !important; }

/* ── Divider ── */
hr { border-color: #1e293b; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1526;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #64748b;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
}
.stTabs [aria-selected="true"] {
    background: #1d4ed8 !important;
    color: white !important;
}

/* ── Alerts ── */
.stAlert { border-radius: 12px; }

/* ── DataFrames ── */
.dataframe { border-radius: 12px; overflow: hidden; }
thead tr th {
    background-color: #0f1e38 !important;
    color: #94a3b8 !important;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
tbody tr td { color: #e2e8f0 !important; }
tbody tr:hover td { background-color: #111f38 !important; }

/* ── Custom card ── */
.iq-card {
    background: linear-gradient(135deg, #0f1e38 0%, #0d1a30 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
}
.iq-card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4a7abf;
    margin-bottom: 8px;
}
.iq-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 2px;
}
.badge-high { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
.badge-low  { background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }
.badge-med  { background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }

.risk-factor-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #1e293b;
    font-size: 0.88rem;
}
.risk-factor-row:last-child { border-bottom: none; }
.risk-bar-fill {
    height: 6px;
    border-radius: 3px;
    background: linear-gradient(90deg, #1d4ed8, #f59e0b);
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL FILES
# ─────────────────────────────────────────────

@st.cache_resource
def load_model_files():
    model = pickle.load(open("model/model.pkl", "rb"))
    columns = pickle.load(open("model/columns.pkl", "rb"))
    threshold = pickle.load(open("model/threshold.pkl", "rb"))
    feature_importance = pd.read_csv("model/feature_importance.csv")
    return model, columns, threshold, feature_importance

@st.cache_data
def load_data():
    df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1, inplace=True)
    return df

model, columns, threshold, feature_importance = load_model_files()
df_raw = load_data()

df_encoded = df_raw.copy()
df_encoded['Attrition'] = df_encoded['Attrition'].map({'Yes': 1, 'No': 0})
df_encoded = pd.get_dummies(df_encoded, drop_first=True)
feature_means = df_encoded.mean()

# ─────────────────────────────────────────────
# PLOT THEME HELPER
# ─────────────────────────────────────────────

DARK_BG   = "#080d1a"
CARD_BG   = "#0f1e38"
ACCENT    = "#1d4ed8"
ACCENT2   = "#f59e0b"
TEXT      = "#e2e8f0"
MUTED     = "#475569"
GRID      = "#1e293b"
SUCCESS   = "#10b981"
DANGER    = "#ef4444"

def apply_dark_theme(fig):
    fig.update_layout(
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(family="DM Sans", color=TEXT, size=12),
        title_font=dict(family="Syne", color=TEXT, size=14),
        legend=dict(bgcolor=CARD_BG, bordercolor=GRID),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, color=MUTED),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, color=MUTED),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 30px;'>
        <div style='font-family:Syne; font-size:1.5rem; font-weight:800;
                    background: linear-gradient(135deg, #60a5fa, #f59e0b);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            AttritionIQ
        </div>
        <div style='font-size:0.72rem; color:#475569; letter-spacing:0.15em; margin-top:4px;'>
            EMPLOYEE INTELLIGENCE
        </div>
    </div>
    """, unsafe_allow_html=True)

    menu = st.radio(
        "",
        ["🎯  Predict Risk", "📊  Analytics", "🔍  SHAP Explainability",
         "📈  Feature Importance", "📁  Batch Prediction"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='margin:20px 0; border-color:#1e293b'>", unsafe_allow_html=True)

    # Sidebar KPIs
    total = len(df_raw)
    attrition_rate = round((df_raw['Attrition'] == 'Yes').mean() * 100, 1)
    avg_income = int(df_raw['MonthlyIncome'].mean())
    avg_tenure = round(df_raw['YearsAtCompany'].mean(), 1)

    st.markdown(f"""
    <div style='padding:0 8px'>
        <div class='iq-card-title'>Dataset Overview</div>
        <div style='display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:10px;'>
            <div style='background:#0d1526; border-radius:10px; padding:12px; text-align:center;'>
                <div style='font-family:Syne; font-size:1.2rem; font-weight:700; color:#60a5fa;'>{total:,}</div>
                <div style='font-size:0.68rem; color:#475569; letter-spacing:0.05em;'>EMPLOYEES</div>
            </div>
            <div style='background:#0d1526; border-radius:10px; padding:12px; text-align:center;'>
                <div style='font-family:Syne; font-size:1.2rem; font-weight:700; color:#f87171;'>{attrition_rate}%</div>
                <div style='font-size:0.68rem; color:#475569; letter-spacing:0.05em;'>ATTRITION</div>
            </div>
            <div style='background:#0d1526; border-radius:10px; padding:12px; text-align:center;'>
                <div style='font-family:Syne; font-size:1.2rem; font-weight:700; color:#34d399;'>${avg_income:,}</div>
                <div style='font-size:0.68rem; color:#475569; letter-spacing:0.05em;'>AVG INCOME</div>
            </div>
            <div style='background:#0d1526; border-radius:10px; padding:12px; text-align:center;'>
                <div style='font-family:Syne; font-size:1.2rem; font-weight:700; color:#fbbf24;'>{avg_tenure}y</div>
                <div style='font-size:0.68rem; color:#475569; letter-spacing:0.05em;'>AVG TENURE</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style='padding: 20px 8px 0; font-size:0.72rem; color:#1e3a5f; text-align:center;'>
        Model threshold · <b style='color:#4a7abf;'>{round(threshold,2)}</b>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: PREDICT RISK
# ─────────────────────────────────────────────

if "🎯" in menu:

    st.markdown("""
    <div style='margin-bottom:28px;'>
        <div style='font-family:Syne; font-size:2rem; font-weight:800; color:#f0f4ff;'>
            Employee Risk Predictor
        </div>
        <div style='color:#475569; font-size:0.9rem; margin-top:4px;'>
            Enter employee details to generate an attrition probability score
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([1.1, 0.9], gap="large")

    with col_form:
        st.markdown("<div class='iq-card'>", unsafe_allow_html=True)
        st.markdown("<div class='iq-card-title'>Personal & Role</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Age", 18, 60, 32)
            monthly_income = st.number_input("Monthly Income ($)", 1000, 50000, 5000, step=500)
            job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
            years_at_company = st.slider("Years at Company", 0, 40, 3)
        with c2:
            distance = st.slider("Distance From Home (km)", 1, 30, 8)
            num_companies = st.slider("Num Companies Worked", 0, 10, 2)
            total_working_years = st.slider("Total Working Years", 0, 40, 7)
            years_since_promotion = st.slider("Years Since Promotion", 0, 15, 2)

        st.markdown("<div class='iq-card-title' style='margin-top:16px;'>Satisfaction & Environment</div>", unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            job_satisfaction = st.slider("Job Satisfaction (1–4)", 1, 4, 3)
            env_satisfaction = st.slider("Environment Satisfaction (1–4)", 1, 4, 3)
            work_life_balance = st.slider("Work-Life Balance (1–4)", 1, 4, 3)
        with c4:
            job_involvement = st.slider("Job Involvement (1–4)", 1, 4, 3)
            relationship_sat = st.slider("Relationship Satisfaction (1–4)", 1, 4, 3)
            stock_option = st.selectbox("Stock Option Level", [0, 1, 2, 3])

        st.markdown("<div class='iq-card-title' style='margin-top:16px;'>Work Conditions</div>", unsafe_allow_html=True)
        c5, c6 = st.columns(2)
        with c5:
            overtime = st.selectbox("OverTime", ["Yes", "No"])
            business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
            department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        with c6:
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            education = st.selectbox("Education Level", [1, 2, 3, 4, 5],
                                     format_func=lambda x: {1:"Below College",2:"College",3:"Bachelor",4:"Master",5:"Doctor"}[x])
            training_times = st.slider("Training Times Last Year", 0, 6, 2)

        st.markdown("</div>", unsafe_allow_html=True)
        predict_btn = st.button("🔮  Generate Attrition Score", use_container_width=True)

    with col_result:
        if predict_btn:
            input_dict = feature_means.to_dict()

            # Numeric overrides
            overrides = {
                "Age": age, "MonthlyIncome": monthly_income, "DistanceFromHome": distance,
                "YearsAtCompany": years_at_company, "JobSatisfaction": job_satisfaction,
                "EnvironmentSatisfaction": env_satisfaction, "WorkLifeBalance": work_life_balance,
                "JobInvolvement": job_involvement, "RelationshipSatisfaction": relationship_sat,
                "StockOptionLevel": stock_option, "JobLevel": job_level,
                "NumCompaniesWorked": num_companies, "TotalWorkingYears": total_working_years,
                "YearsSinceLastPromotion": years_since_promotion, "Education": education,
                "TrainingTimesLastYear": training_times
            }
            input_dict.update(overrides)

            cat_map = {
                "OverTime_": overtime, "BusinessTravel_": business_travel,
                "Department_": department, "MaritalStatus_": marital_status
            }
            for prefix, value in cat_map.items():
                for col in columns:
                    if col.startswith(prefix):
                        input_dict[col] = 1 if col == f"{prefix}{value}" else 0

            input_df = pd.DataFrame([input_dict])[columns]
            probability = model.predict_proba(input_df)[0][1]
            prediction = 1 if probability >= threshold else 0
            pct = round(probability * 100, 1)

            # Risk label
            if pct < 35:
                risk_label, risk_color, badge_cls = "LOW RISK", SUCCESS, "badge-low"
            elif pct < 65:
                risk_label, risk_color, badge_cls = "MEDIUM RISK", ACCENT2, "badge-med"
            else:
                risk_label, risk_color, badge_cls = "HIGH RISK", DANGER, "badge-high"

            # Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pct,
                number={'suffix': '%', 'font': {'family': 'Syne', 'size': 40, 'color': risk_color}},
                title={'text': "ATTRITION PROBABILITY",
                       'font': {'family': 'DM Sans', 'size': 11, 'color': MUTED}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': MUTED, 'tickfont': {'color': MUTED}},
                    'bar': {'color': risk_color, 'thickness': 0.25},
                    'bgcolor': CARD_BG,
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 35],  'color': 'rgba(16,185,129,0.08)'},
                        {'range': [35, 65], 'color': 'rgba(245,158,11,0.08)'},
                        {'range': [65, 100],'color': 'rgba(239,68,68,0.08)'}
                    ],
                    'threshold': {
                        'line': {'color': '#60a5fa', 'width': 2},
                        'thickness': 0.75,
                        'value': threshold * 100
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
                height=280, margin=dict(l=30, r=30, t=60, b=10)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Risk verdict
            st.markdown(f"""
            <div class='iq-card' style='text-align:center; border-color:{risk_color}40;'>
                <span class='iq-badge {badge_cls}' style='font-size:0.9rem; padding:6px 20px;'>
                    {risk_label}
                </span>
                <div style='font-family:Syne; font-size:1.6rem; font-weight:700;
                            color:{risk_color}; margin:10px 0 4px;'>
                    {pct}% probability
                </div>
                <div style='color:#475569; font-size:0.82rem;'>
                    Decision threshold at {round(threshold*100,0):.0f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Key risk drivers
            st.markdown("<div class='iq-card'>", unsafe_allow_html=True)
            st.markdown("<div class='iq-card-title'>Key Risk Drivers</div>", unsafe_allow_html=True)

            top_feats = feature_importance.head(6)
            for _, row in top_feats.iterrows():
                feat_name = row['Feature']
                importance = row['Importance']
                bar_w = int(importance / feature_importance['Importance'].max() * 100)
                raw_name = feat_name.replace("_Yes","").replace("_Single","").replace("_"," ").title()
                st.markdown(f"""
                <div class='risk-factor-row'>
                    <span style='color:#94a3b8; width:170px;'>{raw_name}</span>
                    <div style='flex:1; margin:0 12px;'>
                        <div style='background:#1e293b; height:6px; border-radius:3px;'>
                            <div class='risk-bar-fill' style='width:{bar_w}%;'></div>
                        </div>
                    </div>
                    <span style='color:#60a5fa; font-size:0.78rem; width:40px; text-align:right;'>
                        {round(importance*100,1)}%
                    </span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Retention tips
            if prediction == 1:
                tips = []
                if overtime == "Yes":       tips.append("📌 Reduce overtime burden")
                if job_satisfaction <= 2:   tips.append("📌 Address job satisfaction issues")
                if work_life_balance <= 2:  tips.append("📌 Improve work-life balance programs")
                if stock_option == 0:       tips.append("📌 Consider stock option grants")
                if distance > 15:           tips.append("📌 Offer remote/hybrid work options")
                if years_since_promotion > 3: tips.append("📌 Review promotion eligibility")
                if marital_status == "Single": tips.append("📌 Strengthen team engagement & culture")

                if tips:
                    st.markdown("<div class='iq-card' style='border-color:rgba(245,158,11,0.3);'>", unsafe_allow_html=True)
                    st.markdown("<div class='iq-card-title' style='color:#f59e0b;'>Retention Recommendations</div>", unsafe_allow_html=True)
                    for tip in tips[:4]:
                        st.markdown(f"<div style='color:#94a3b8; font-size:0.84rem; padding:4px 0;'>{tip}</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='display:flex; flex-direction:column; align-items:center;
                        justify-content:center; height:500px; color:#1e3a5f;'>
                <div style='font-size:4rem;'>🧠</div>
                <div style='font-family:Syne; font-size:1.1rem; color:#1e3a5f; margin-top:16px;'>
                    Fill in the form and click predict
                </div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: ANALYTICS DASHBOARD
# ─────────────────────────────────────────────

elif "📊" in menu:

    st.markdown("""
    <div style='margin-bottom:28px;'>
        <div style='font-family:Syne; font-size:2rem; font-weight:800; color:#f0f4ff;'>
            Analytics Dashboard
        </div>
        <div style='color:#475569; font-size:0.9rem; margin-top:4px;'>
            Workforce intelligence across departments, roles, and demographics
        </div>
    </div>
    """, unsafe_allow_html=True)

    df = df_raw.copy()

    # ── KPI Row ──
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Employees",    f"{len(df):,}")
    k2.metric("Attrition Count",    f"{(df['Attrition']=='Yes').sum():,}")
    k3.metric("Attrition Rate",     f"{round((df['Attrition']=='Yes').mean()*100,1)}%")
    k4.metric("Avg Monthly Income", f"${int(df['MonthlyIncome'].mean()):,}")
    k5.metric("Avg Age",            f"{round(df['Age'].mean(),1)}")

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["  Overview  ", "  Workforce Profile  ", "  Satisfaction  ", "  Compensation  "])

    # ── TAB 1: Overview ──
    with tab1:
        c1, c2 = st.columns(2)

        with c1:
            counts = df['Attrition'].value_counts()
            fig_pie = go.Figure(go.Pie(
                labels=["Stayed", "Left"],
                values=[counts.get("No",0), counts.get("Yes",0)],
                hole=0.65,
                marker=dict(colors=[ACCENT, DANGER]),
                textinfo='percent',
                hoverinfo='label+value'
            ))
            fig_pie.add_annotation(text=f"{round(counts.get('Yes',0)/len(df)*100,1)}%<br>attrition",
                                   x=0.5, y=0.5, showarrow=False,
                                   font=dict(size=16, family="Syne", color=TEXT))
            fig_pie.update_layout(title="Overall Attrition Split", paper_bgcolor=CARD_BG,
                                  font=dict(color=TEXT, family="DM Sans"),
                                  margin=dict(l=10,r=10,t=50,b=10), height=320,
                                  legend=dict(bgcolor=CARD_BG))
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            dept_attr = df.groupby('Department')['Attrition'].apply(
                lambda x: (x=='Yes').mean()*100).reset_index()
            dept_attr.columns = ['Department','Rate']
            fig_dept = px.bar(dept_attr, x='Rate', y='Department', orientation='h',
                              color='Rate', color_continuous_scale=['#1d4ed8','#f59e0b','#ef4444'],
                              title='Attrition Rate by Department (%)')
            apply_dark_theme(fig_dept)
            fig_dept.update_layout(coloraxis_showscale=False, height=320)
            st.plotly_chart(fig_dept, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            role_attr = df.groupby('JobRole')['Attrition'].apply(
                lambda x: (x=='Yes').mean()*100).reset_index()
            role_attr.columns = ['Role','Rate']
            role_attr = role_attr.sort_values('Rate', ascending=True)
            fig_role = px.bar(role_attr, x='Rate', y='Role', orientation='h',
                              color='Rate', color_continuous_scale=['#1d4ed8','#ef4444'],
                              title='Attrition Rate by Job Role (%)')
            apply_dark_theme(fig_role)
            fig_role.update_layout(coloraxis_showscale=False, height=380)
            st.plotly_chart(fig_role, use_container_width=True)

        with c4:
            travel_attr = df.groupby('BusinessTravel')['Attrition'].apply(
                lambda x: (x=='Yes').mean()*100).reset_index()
            travel_attr.columns = ['Travel','Rate']
            fig_travel = px.bar(travel_attr, x='Travel', y='Rate',
                                color='Rate', color_continuous_scale=['#10b981','#ef4444'],
                                title='Attrition Rate by Business Travel (%)')
            apply_dark_theme(fig_travel)
            fig_travel.update_layout(coloraxis_showscale=False, height=380)
            st.plotly_chart(fig_travel, use_container_width=True)

    # ── TAB 2: Workforce Profile ──
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig_age = px.histogram(df, x='Age', color='Attrition',
                                   color_discrete_map={'Yes': DANGER, 'No': ACCENT},
                                   nbins=25, barmode='overlay', opacity=0.75,
                                   title='Age Distribution by Attrition')
            apply_dark_theme(fig_age)
            st.plotly_chart(fig_age, use_container_width=True)

        with c2:
            fig_tenure = px.histogram(df, x='YearsAtCompany', color='Attrition',
                                      color_discrete_map={'Yes': DANGER, 'No': ACCENT},
                                      nbins=20, barmode='overlay', opacity=0.75,
                                      title='Company Tenure Distribution')
            apply_dark_theme(fig_tenure)
            st.plotly_chart(fig_tenure, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            ms_attr = df.groupby('MaritalStatus')['Attrition'].apply(
                lambda x: (x=='Yes').mean()*100).reset_index()
            ms_attr.columns = ['Status','Rate']
            fig_ms = px.pie(ms_attr, values='Rate', names='Status',
                            color_discrete_sequence=[ACCENT, ACCENT2, DANGER],
                            hole=0.5, title='Attrition Rate by Marital Status')
            fig_ms.update_layout(paper_bgcolor=CARD_BG, font=dict(color=TEXT, family="DM Sans"),
                                 margin=dict(l=10,r=10,t=50,b=10), legend=dict(bgcolor=CARD_BG))
            st.plotly_chart(fig_ms, use_container_width=True)

        with c4:
            ot_attr = df.groupby('OverTime')['Attrition'].apply(
                lambda x: (x=='Yes').mean()*100).reset_index()
            ot_attr.columns = ['OverTime','Rate']
            fig_ot = px.bar(ot_attr, x='OverTime', y='Rate',
                            color='Rate', color_continuous_scale=[SUCCESS, DANGER],
                            title='Attrition Rate — OverTime vs Not (%)')
            apply_dark_theme(fig_ot)
            fig_ot.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_ot, use_container_width=True)

    # ── TAB 3: Satisfaction ──
    with tab3:
        sat_cols = ['JobSatisfaction','EnvironmentSatisfaction',
                    'WorkLifeBalance','JobInvolvement','RelationshipSatisfaction']
        rows = []
        for col in sat_cols:
            for rating in [1,2,3,4]:
                subset = df[df[col] == rating]
                attr_rate = (subset['Attrition']=='Yes').mean()*100 if len(subset)>0 else 0
                rows.append({'Metric': col.replace('Satisfaction','').replace('WorkLife','Work-Life '), 'Rating': str(rating), 'Rate': attr_rate})
        sat_df = pd.DataFrame(rows)
        fig_sat = px.bar(sat_df, x='Rating', y='Rate', facet_col='Metric',
                         color='Rate', color_continuous_scale=['#10b981','#f59e0b','#ef4444'],
                         title='Attrition Rate by Satisfaction Rating')
        apply_dark_theme(fig_sat)
        fig_sat.update_layout(coloraxis_showscale=False, height=400)
        st.plotly_chart(fig_sat, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig_box = px.box(df, x='Attrition', y='JobSatisfaction', color='Attrition',
                             color_discrete_map={'Yes': DANGER, 'No': ACCENT},
                             title='Job Satisfaction Distribution')
            apply_dark_theme(fig_box)
            st.plotly_chart(fig_box, use_container_width=True)
        with c2:
            fig_box2 = px.box(df, x='Attrition', y='WorkLifeBalance', color='Attrition',
                              color_discrete_map={'Yes': DANGER, 'No': ACCENT},
                              title='Work-Life Balance Distribution')
            apply_dark_theme(fig_box2)
            st.plotly_chart(fig_box2, use_container_width=True)

    # ── TAB 4: Compensation ──
    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            fig_inc = px.box(df, x='Attrition', y='MonthlyIncome', color='Attrition',
                             color_discrete_map={'Yes': DANGER, 'No': ACCENT},
                             title='Monthly Income vs Attrition')
            apply_dark_theme(fig_inc)
            st.plotly_chart(fig_inc, use_container_width=True)

        with c2:
            fig_hike = px.histogram(df, x='PercentSalaryHike', color='Attrition',
                                    color_discrete_map={'Yes': DANGER, 'No': ACCENT},
                                    nbins=15, barmode='overlay', opacity=0.8,
                                    title='Salary Hike % Distribution')
            apply_dark_theme(fig_hike)
            st.plotly_chart(fig_hike, use_container_width=True)

        fig_scatter = px.scatter(df, x='MonthlyIncome', y='YearsAtCompany',
                                 color='Attrition', size='Age',
                                 color_discrete_map={'Yes': DANGER, 'No': ACCENT},
                                 opacity=0.6, title='Income vs Tenure (sized by Age)')
        apply_dark_theme(fig_scatter)
        st.plotly_chart(fig_scatter, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: SHAP EXPLAINABILITY
# ─────────────────────────────────────────────

elif "🔍" in menu:

    st.markdown("""
    <div style='margin-bottom:28px;'>
        <div style='font-family:Syne; font-size:2rem; font-weight:800; color:#f0f4ff;'>
            Model Explainability
        </div>
        <div style='color:#475569; font-size:0.9rem; margin-top:4px;'>
            SHAP values — understand why the model predicts attrition
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Computing SHAP values…"):
        @st.cache_data
        def compute_shap():
            explainer = shap.Explainer(model)
            X_shap = df_encoded.drop("Attrition", axis=1)
            sv = explainer(X_shap)
            return sv, X_shap

        shap_values, X_shap = compute_shap()

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='iq-card-title'>Global Feature Impact (Beeswarm)</div>", unsafe_allow_html=True)
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        fig1.patch.set_facecolor(CARD_BG)
        ax1.set_facecolor(CARD_BG)
        shap.summary_plot(shap_values, X_shap, show=False, plot_size=None,
                          color_bar=True, max_display=12)
        plt.rcParams.update({'text.color': TEXT, 'axes.labelcolor': TEXT,
                             'xtick.color': MUTED, 'ytick.color': MUTED})
        st.pyplot(fig1)
        plt.close()

    with c2:
        st.markdown("<div class='iq-card-title'>Mean |SHAP| Bar Chart</div>", unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        fig2.patch.set_facecolor(CARD_BG)
        ax2.set_facecolor(CARD_BG)
        shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False,
                          plot_size=None, max_display=12, color=ACCENT)
        plt.rcParams.update({'text.color': TEXT, 'axes.labelcolor': TEXT,
                             'xtick.color': MUTED, 'ytick.color': MUTED})
        st.pyplot(fig2)
        plt.close()

# ─────────────────────────────────────────────
# PAGE: FEATURE IMPORTANCE
# ─────────────────────────────────────────────

elif "📈" in menu:

    st.markdown("""
    <div style='margin-bottom:28px;'>
        <div style='font-family:Syne; font-size:2rem; font-weight:800; color:#f0f4ff;'>
            Feature Importance
        </div>
        <div style='color:#475569; font-size:0.9rem; margin-top:4px;'>
            Gradient Boosting model — top predictors ranked by importance score
        </div>
    </div>
    """, unsafe_allow_html=True)

    n = st.slider("Number of features to display", 5, len(feature_importance), 20)
    top = feature_importance.head(n).sort_values("Importance", ascending=True)

    fig_fi = go.Figure(go.Bar(
        x=top["Importance"],
        y=top["Feature"],
        orientation="h",
        marker=dict(
            color=top["Importance"],
            colorscale=[[0, ACCENT], [0.5, ACCENT2], [1, DANGER]],
            showscale=True,
            colorbar=dict(title="Score", tickfont=dict(color=MUTED))
        ),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
    ))
    fig_fi.update_layout(
        title=f"Top {n} Feature Importances",
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(family="DM Sans", color=TEXT),
        title_font=dict(family="Syne", size=16),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, color=MUTED),
        yaxis=dict(color=TEXT, tickfont=dict(size=11)),
        height=max(400, n * 28),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    with st.expander("📋  View Full Feature Table"):
        st.dataframe(
            feature_importance.style.background_gradient(
                subset=["Importance"], cmap="Blues"
            ).format({"Importance": "{:.5f}"}),
            use_container_width=True
        )

# ─────────────────────────────────────────────
# PAGE: BATCH PREDICTION
# ─────────────────────────────────────────────

elif "📁" in menu:

    st.markdown("""
    <div style='margin-bottom:28px;'>
        <div style='font-family:Syne; font-size:2rem; font-weight:800; color:#f0f4ff;'>
            Batch Prediction
        </div>
        <div style='color:#475569; font-size:0.9rem; margin-top:4px;'>
            Upload a CSV of employees and get attrition risk scores for all of them
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='iq-card' style='border-color:rgba(29,78,216,0.3);'>
        <div class='iq-card-title'>Expected Format</div>
        <div style='color:#64748b; font-size:0.83rem; line-height:1.6;'>
            Upload a CSV with the same columns as the original dataset
            (Age, MonthlyIncome, Department, OverTime, etc.).<br>
            The model will score each row and append a risk column.
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload employee CSV", type=["csv"])

    if uploaded:
        df_batch = pd.read_csv(uploaded)
        drop_batch = [c for c in ['EmployeeCount','EmployeeNumber','Over18','StandardHours','Attrition'] if c in df_batch.columns]
        df_batch.drop(columns=drop_batch, inplace=True)

        df_enc = pd.get_dummies(df_batch, drop_first=True)

        # Align columns
        for col in columns:
            if col not in df_enc.columns:
                df_enc[col] = 0
        df_enc = df_enc[columns]

        probs = model.predict_proba(df_enc)[:, 1]
        preds = (probs >= threshold).astype(int)

        df_batch["Attrition_Probability"] = (probs * 100).round(1)
        df_batch["Risk_Label"] = ["HIGH" if p >= 0.65 else "MEDIUM" if p >= 0.35 else "LOW" for p in probs]
        df_batch["Prediction"] = ["Yes" if p == 1 else "No" for p in preds]

        # ── Summary KPIs ──
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Employees Scored",  len(df_batch))
        k2.metric("High Risk",         int((df_batch['Risk_Label']=='HIGH').sum()))
        k3.metric("Medium Risk",       int((df_batch['Risk_Label']=='MEDIUM').sum()))
        k4.metric("Low Risk",          int((df_batch['Risk_Label']=='LOW').sum()))

        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            risk_counts = df_batch['Risk_Label'].value_counts()
            fig_risk = go.Figure(go.Pie(
                labels=risk_counts.index, values=risk_counts.values, hole=0.6,
                marker=dict(colors=[DANGER, ACCENT2, SUCCESS]),
                textinfo='percent+label'
            ))
            fig_risk.update_layout(title="Risk Distribution", paper_bgcolor=CARD_BG,
                                   font=dict(color=TEXT, family="DM Sans"),
                                   margin=dict(l=10,r=10,t=50,b=10),
                                   legend=dict(bgcolor=CARD_BG), height=300)
            st.plotly_chart(fig_risk, use_container_width=True)

        with c2:
            fig_hist = px.histogram(df_batch, x='Attrition_Probability', nbins=20,
                                    color_discrete_sequence=[ACCENT],
                                    title='Score Distribution (%)')
            apply_dark_theme(fig_hist)
            fig_hist.update_layout(height=300)
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("#### 🔴 High Risk Employees")
        high_risk = df_batch[df_batch['Risk_Label']=='HIGH'].sort_values(
            'Attrition_Probability', ascending=False)
        st.dataframe(high_risk, use_container_width=True)

        # Download
        csv_out = df_batch.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️  Download Scored CSV",
            data=csv_out,
            file_name="attrition_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.markdown("""
        <div style='display:flex; flex-direction:column; align-items:center;
                    justify-content:center; height:300px; color:#1e3a5f;'>
            <div style='font-size:3rem;'>📂</div>
            <div style='font-family:Syne; font-size:1rem; color:#1e3a5f; margin-top:12px;'>
                No file uploaded yet
            </div>
        </div>
        """, unsafe_allow_html=True)