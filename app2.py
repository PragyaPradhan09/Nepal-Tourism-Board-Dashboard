"""
Nepal Tourism Forecasting — Streamlit App
==========================================
Run with:
   streamlit run app.py


Place this file in the same folder as Nepal_Tourism_Modeling_Dataset.xlsx
"""


import warnings
warnings.filterwarnings("ignore")


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.outliers_influence import variance_inflation_factor


from pmdarima import auto_arima


from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
   page_title="Nepal Tourism Forecasting",
   page_icon="🏔️",
   layout="wide",
   initial_sidebar_state="expanded"
)


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
   .main-title {
       font-size: 2.2rem; font-weight: 700;
       color: #1A5276; text-align: center;
       padding: 10px 0 5px 0;
   }
   .sub-title {
       font-size: 1rem; color: #5D6D7E;
       text-align: center; margin-bottom: 20px;
   }
   .metric-card {
       background: #EBF5FB; border-radius: 10px;
       padding: 15px; text-align: center;
       border-left: 5px solid #2E86C1;
   }
   .metric-good  { border-left-color: #27AE60; background: #EAFAF1; }
   .metric-warn  { border-left-color: #F39C12; background: #FEF9E7; }
   .metric-bad   { border-left-color: #E74C3C; background: #FDEDEC; }
   .section-header {
       font-size: 1.3rem; font-weight: 600;
       color: #1A5276; border-bottom: 2px solid #2E86C1;
       padding-bottom: 5px; margin: 20px 0 10px 0;
   }
   .info-box {
       background: #EBF5FB; border-radius: 8px;
       padding: 12px 16px; font-size: 0.9rem;
       color: #1A5276; margin: 10px 0;
   }
   .warn-box {
       background: #FEF9E7; border-radius: 8px;
       padding: 12px 16px; font-size: 0.9rem;
       color: #7D6608; margin: 10px 0;
   }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏔️ Nepal Tourism Arrival Forecasting</div>',
           unsafe_allow_html=True)
st.markdown('<div class="sub-title">SARIMA & Ridge Regression Models | DATA 620 | King\'s College · Westcliff University</div>',
           unsafe_allow_html=True)
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
   st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9b/Flag_of_Nepal.svg/200px-Flag_of_Nepal.svg.png",
            width=60)
   st.markdown("### ⚙️ Model Settings")


   data_file = st.text_input("Dataset filename",
                             value="Nepal_Tourism_Modeling_Dataset.xlsx")


   st.markdown("**SARIMA Settings**")
   forecast_months = st.slider("Forecast horizon (months)", 12, 36, 24)


   st.markdown("**Linear Regression Settings**")
   forecast_years = st.slider("Forecast horizon (years)", 1, 5, 3)
   n_test_years   = st.slider("Test set size (years)", 2, 4, 3)


   st.markdown("**Feature Selection — Third Country**")
   use_trekking = st.checkbox("Trekking & Mountaineering Arrivals", value=True)
   use_lag1_3rd = st.checkbox("Lag 1 (prev year arrivals)", value=True)
   use_lag2_3rd = st.checkbox("Lag 2 (2 years prior)", value=True)
   use_income   = st.checkbox("Tourism Income (USD Mn)", value=False)
   use_los      = st.checkbox("Avg Length of Stay", value=False)


   st.markdown("**Feature Selection — Indian**")
   use_pilgrimage = st.checkbox("Pilgrimage Arrivals", value=True)
   use_lag1_ind   = st.checkbox("Lag 1 Indian arrivals", value=True)
   use_holiday    = st.checkbox("Holiday Arrivals", value=False)


   run_button = st.button("🚀 Run Models", type="primary", use_container_width=True)


   st.markdown("---")
   st.markdown("**Authors**")
   st.markdown("Asbina Gurung\nPragya Pradhan\nSneha Sharma")
   st.markdown("*Prof. Pawan Niroula*")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def get_metrics(y_true, y_pred):
   y_true = np.array(y_true, dtype=float)
   y_pred = np.array(y_pred, dtype=float)
   r2   = r2_score(y_true, y_pred)
   mae  = mean_absolute_error(y_true, y_pred)
   rmse = np.sqrt(mean_squared_error(y_true, y_pred))
   mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
   return {"R²": r2, "MAE": mae, "RMSE": rmse, "MAPE (%)": mape}


def metric_color(r2):
   if r2 >= 0.8:  return "metric-good"
   if r2 >= 0.5:  return "metric-warn"
   return "metric-bad"


def show_metrics(metrics, label):
   r2 = metrics["R²"]
   cls = metric_color(r2)
   col1, col2, col3, col4 = st.columns(4)
   col1.markdown(f'<div class="metric-card {cls}"><b>R²</b><br>{r2:.4f}</div>',
                 unsafe_allow_html=True)
   col2.markdown(f'<div class="metric-card {cls}"><b>MAE</b><br>{metrics["MAE"]:,.0f}</div>',
                 unsafe_allow_html=True)
   col3.markdown(f'<div class="metric-card {cls}"><b>RMSE</b><br>{metrics["RMSE"]:,.0f}</div>',
                 unsafe_allow_html=True)
   col4.markdown(f'<div class="metric-card {cls}"><b>MAPE</b><br>{metrics["MAPE (%)"]:.2f}%</div>',
                 unsafe_allow_html=True)


@st.cache_data
def load_data(filepath):
   sarima_df = pd.read_excel(filepath, sheet_name="SARIMA_Monthly", header=2)
   lr_3rd_df = pd.read_excel(filepath, sheet_name="LR_ThirdCountry_Annual", header=3)
   lr_ind_df = pd.read_excel(filepath, sheet_name="LR_Indian_Annual", header=3)


   sarima_df = sarima_df[["Year","Month","Date (YYYY-MM)",
                           "Total_Arrivals","Third_Country_Arrivals",
                           "Indian_Arrivals"]].dropna(subset=["Year"])
   sarima_df["Date"] = pd.to_datetime(sarima_df["Date (YYYY-MM)"])
   sarima_df = sarima_df.set_index("Date").sort_index()


   lr_3rd_df = lr_3rd_df.rename(
       columns={"Third_Country_Arrivals (Y)": "Third_Country_Arrivals"}
   ).dropna(subset=["Year"])
   lr_3rd_df["Year"] = lr_3rd_df["Year"].astype(int)


   lr_ind_df = lr_ind_df.rename(
       columns={"Indian_Arrivals (Y)": "Indian_Arrivals"}
   ).dropna(subset=["Year"])
   lr_ind_df["Year"] = lr_ind_df["Year"].astype(int)


   return sarima_df, lr_3rd_df, lr_ind_df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
try:
   sarima_df, lr_3rd_df, lr_ind_df = load_data(data_file)
   data_loaded = True
except Exception as e:
   st.error(f"❌ Could not load dataset: {e}")
   st.info("Make sure **Nepal_Tourism_Modeling_Dataset.xlsx** is in the same folder as app.py")
   data_loaded = False


if data_loaded:


   # ── TABS ─────────────────────────────────────────────────────────────────
   tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
       "Data Overview",
       "Exploratory Analysis",
       "SARIMA Models",
       "Ridge Regression",
       "Model Comparison",
       "Dashboard"
   ])


   # ─────────────────────────────────────────────────────────────────────────
   # TAB 1: DATA OVERVIEW
   # ─────────────────────────────────────────────────────────────────────────
   with tab1:
       st.markdown('<div class="section-header">Dataset Overview</div>',
                   unsafe_allow_html=True)


       col1, col2, col3 = st.columns(3)
       col1.metric("Monthly Rows (SARIMA)", len(sarima_df),
                   f"{sarima_df.index[0].year}–{sarima_df.index[-1].year}")
       col2.metric("Annual Rows (LR)", len(lr_3rd_df),
                   f"{lr_3rd_df['Year'].min()}–{lr_3rd_df['Year'].max()}")
       col3.metric("Total Variables", "19",
                   "across 3 modeling sheets")


       st.markdown("---")
       col1, col2 = st.columns(2)


       with col1:
           st.markdown("**SARIMA Monthly Dataset (sample)**")
           st.dataframe(sarima_df.reset_index()[
               ["Date","Third_Country_Arrivals","Indian_Arrivals","Total_Arrivals"]
           ].tail(12).style.format({
               "Third_Country_Arrivals": "{:,.0f}",
               "Indian_Arrivals": "{:,.0f}",
               "Total_Arrivals": "{:,.0f}",
           }), use_container_width=True)


       with col2:
           st.markdown("**LR Annual Dataset — Third Country (sample)**")
           display_cols = ["Year","Third_Country_Arrivals",
                           "Lag1_Third_Country","Trekking_Mtn_Arrivals",
                           "COVID_Dummy","Earthquake_Dummy"]
           st.dataframe(lr_3rd_df[display_cols].tail(8).style.format({
               "Third_Country_Arrivals": "{:,.0f}",
               "Lag1_Third_Country": "{:,.0f}",
               "Trekking_Mtn_Arrivals": "{:,.0f}",
           }), use_container_width=True)


       st.markdown("---")
       st.markdown("**Annual Arrival Summary**")
       summary = lr_3rd_df[["Year"]].copy()
       summary["Third_Country"] = lr_3rd_df["Third_Country_Arrivals"].astype(int)
       summary["Indian"]        = lr_ind_df["Indian_Arrivals"].astype(int)
       summary["Total"]         = summary["Third_Country"] + summary["Indian"]
       summary["YoY_Growth_%"]  = summary["Total"].pct_change() * 100
       st.dataframe(summary.style.format({
           "Third_Country": "{:,.0f}",
           "Indian": "{:,.0f}",
           "Total": "{:,.0f}",
           "YoY_Growth_%": "{:.1f}%",
       }).background_gradient(subset=["Total"], cmap="Blues"),
       use_container_width=True)


   # ─────────────────────────────────────────────────────────────────────────
   # TAB 2: EXPLORATORY ANALYSIS
   # ─────────────────────────────────────────────────────────────────────────
   with tab2:
       st.markdown('<div class="section-header">Exploratory Analysis</div>',
                   unsafe_allow_html=True)


       series_3rd = sarima_df["Third_Country_Arrivals"].dropna()
       series_ind = sarima_df["Indian_Arrivals"].dropna()


       # Time series plots
       col1, col2 = st.columns(2)
       with col1:
           fig, ax = plt.subplots(figsize=(7, 3.5))
           ax.plot(series_3rd.index, series_3rd.values, color="#2E86C1", lw=1.8)
           ax.fill_between(series_3rd.index, series_3rd.values, alpha=0.15, color="#2E86C1")
           ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31"),
                      color="#E74C3C", alpha=0.15, label="COVID")
           ax.axvline(pd.Timestamp("2015-04-01"), color="#E67E22",
                      linestyle="--", lw=1.2, label="Earthquake")
           ax.set_title("Third-Country Monthly Arrivals", fontweight="bold")
           ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{int(x/1000)}K"))
           ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
           ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
           plt.tight_layout(); st.pyplot(fig); plt.close()


       with col2:
           fig, ax = plt.subplots(figsize=(7, 3.5))
           ax.plot(series_ind.index, series_ind.values, color="#E67E22", lw=1.8)
           ax.fill_between(series_ind.index, series_ind.values, alpha=0.15, color="#E67E22")
           ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31"),
                      color="#E74C3C", alpha=0.15, label="COVID")
           ax.axvline(pd.Timestamp("2015-04-01"), color="#8E44AD",
                      linestyle="--", lw=1.2, label="Earthquake")
           ax.set_title("Indian Monthly Arrivals", fontweight="bold")
           ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{int(x/1000)}K"))
           ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
           ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
           plt.tight_layout(); st.pyplot(fig); plt.close()


       # Seasonality heatmap
       st.markdown("**Seasonality Heatmap — Third-Country Arrivals**")
       pivot = sarima_df[["Third_Country_Arrivals"]].copy()
       pivot["Year"]  = pivot.index.year
       pivot["Month"] = pivot.index.month
       heat = pivot.pivot_table(values="Third_Country_Arrivals",
                                index="Year", columns="Month",
                                aggfunc="sum").astype(float)
       heat.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
       annot = heat.map(lambda v: f"{int(v/1000)}K" if not pd.isna(v) else "")
       fig, ax = plt.subplots(figsize=(13, 6))
       sns.heatmap(heat, annot=annot, fmt="", cmap="YlOrRd",
                   linewidths=0.4, ax=ax, cbar_kws={"label": "Arrivals"})
       ax.set_title("Monthly Arrival Seasonality by Year", fontweight="bold")
       plt.tight_layout(); st.pyplot(fig); plt.close()


       # ADF stationarity
       st.markdown("**Stationarity Tests (ADF)**")
       adf_3rd = adfuller(series_3rd)
       adf_ind = adfuller(series_ind)
       col1, col2 = st.columns(2)
       col1.metric("Third-Country p-value", f"{adf_3rd[1]:.4f}",
                   "STATIONARY ✓" if adf_3rd[1] < 0.05 else "Non-stationary")
       col2.metric("Indian p-value", f"{adf_ind[1]:.4f}",
                   "STATIONARY ✓" if adf_ind[1] < 0.05 else "Non-stationary → d=1")


       # ACF/PACF
       st.markdown("**ACF & PACF Plots**")
       fig, axes = plt.subplots(2, 2, figsize=(13, 6))
       plot_acf(series_3rd,  lags=36, ax=axes[0][0], title="ACF — Third-Country")
       plot_pacf(series_3rd, lags=36, ax=axes[0][1], title="PACF — Third-Country")
       plot_acf(series_ind,  lags=36, ax=axes[1][0], title="ACF — Indian")
       plot_pacf(series_ind, lags=36, ax=axes[1][1], title="PACF — Indian")
       plt.tight_layout(); st.pyplot(fig); plt.close()


   # ─────────────────────────────────────────────────────────────────────────
   # TAB 3: SARIMA
   # ─────────────────────────────────────────────────────────────────────────
   with tab3:
       st.markdown('<div class="section-header">SARIMA Models</div>',
                   unsafe_allow_html=True)
       st.markdown('<div class="info-box">📌 SARIMA uses monthly time series data. '
                   'auto_arima finds the best (p,d,q)(P,D,Q,12) parameters automatically. '
                   'd=1 forced for both models based on ADF test results.</div>',
                   unsafe_allow_html=True)


       if not run_button:
           st.info("👈 Click **Run Models** in the sidebar to fit SARIMA models.")
       else:
           series_3rd = sarima_df["Third_Country_Arrivals"].dropna()
           series_ind = sarima_df["Indian_Arrivals"].dropna()


           # ── Third Country ─────────────────────────────────────────────────
           st.markdown("### Third-Country Arrivals")
           with st.spinner("Fitting SARIMA for Third-Country..."):
               train_3rd = series_3rd.iloc[:-12]
               test_3rd  = series_3rd.iloc[-12:]
               auto_3rd  = auto_arima(train_3rd, seasonal=True, m=12, d=1, D=1,
                                      stepwise=True, information_criterion="aic",
                                      error_action="ignore", suppress_warnings=True,
                                      trace=False)
               order_3rd   = auto_3rd.order
               s_order_3rd = auto_3rd.seasonal_order
               res_3rd = SARIMAX(train_3rd, order=order_3rd,
                                 seasonal_order=s_order_3rd,
                                 enforce_stationarity=False,
                                 enforce_invertibility=False).fit(disp=False)
               fc_test_3rd = res_3rd.get_forecast(steps=12)
               pred_test_3rd = fc_test_3rd.predicted_mean
               pred_test_3rd.index = test_3rd.index


               res_full_3rd    = SARIMAX(series_3rd, order=order_3rd,
                                         seasonal_order=s_order_3rd,
                                         enforce_stationarity=False,
                                         enforce_invertibility=False).fit(disp=False)
               fc_future_3rd   = res_full_3rd.get_forecast(steps=forecast_months)
               pred_future_3rd = fc_future_3rd.predicted_mean
               ci_future_3rd   = fc_future_3rd.conf_int()


           col1, col2 = st.columns(2)
           col1.success(f"✅ Best order: ARIMA{order_3rd} × {s_order_3rd}")
           col2.info(f"AIC: {res_3rd.aic:.2f}")


           m3rd = get_metrics(test_3rd.values, pred_test_3rd.values)
           st.markdown("**Test Performance (last 12 months)**")
           show_metrics(m3rd, "SARIMA Third-Country Test")


           # Plot
           fig, ax = plt.subplots(figsize=(13, 4.5))
           ax.plot(train_3rd.index, train_3rd.values,
                   color="#2E86C1", lw=1.8, label="Training")
           ax.plot(test_3rd.index, test_3rd.values,
                   color="#1A5276", lw=1.8, linestyle="--", label="Actual (test)")
           ax.plot(pred_future_3rd.index, pred_future_3rd.values,
                   color="#E74C3C", lw=2, label="Forecast")
           ax.fill_between(pred_future_3rd.index,
                           ci_future_3rd.iloc[:,0], ci_future_3rd.iloc[:,1],
                           color="#E74C3C", alpha=0.15, label="95% CI")
           ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31"),
                      color="#E74C3C", alpha=0.08)
           ax.set_title(f"SARIMA — Third-Country Forecast ({forecast_months} months)",
                        fontweight="bold")
           ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{int(x):,}"))
           ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
           ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
           plt.tight_layout(); st.pyplot(fig); plt.close()


           # Forecast table
           st.markdown("**Future Forecast Table**")
           fc_table = pd.DataFrame({
               "Month": pred_future_3rd.index.strftime("%Y-%m"),
               "Forecast": pred_future_3rd.values.round(0).astype(int),
               "Lower 95% CI": ci_future_3rd.iloc[:,0].round(0).astype(int),
               "Upper 95% CI": ci_future_3rd.iloc[:,1].round(0).astype(int),
           })
           st.dataframe(fc_table.style.format({
               "Forecast": "{:,}", "Lower 95% CI": "{:,}", "Upper 95% CI": "{:,}"
           }), use_container_width=True)


           st.markdown("---")


           # ── Indian ────────────────────────────────────────────────────────
           st.markdown("### Indian Arrivals")
           st.markdown('<div class="warn-box">⚠️ COVID years (2020–2021) excluded from '
                       'training. Post-COVID Indian arrivals surged beyond historical '
                       'training range — model accuracy is limited.</div>',
                       unsafe_allow_html=True)


           with st.spinner("Fitting SARIMA for Indian arrivals..."):
               series_ind_clean = series_ind[~series_ind.index.year.isin([2020,2021])]
               train_ind = series_ind_clean.iloc[:-12]
               test_ind  = series_ind_clean.iloc[-12:]
               auto_ind  = auto_arima(train_ind, seasonal=True, m=12, d=1, D=1,
                                      stepwise=True, information_criterion="aic",
                                      error_action="ignore", suppress_warnings=True,
                                      trace=False)
               order_ind   = auto_ind.order
               s_order_ind = auto_ind.seasonal_order
               res_ind = SARIMAX(train_ind, order=order_ind,
                                 seasonal_order=s_order_ind,
                                 enforce_stationarity=False,
                                 enforce_invertibility=False).fit(disp=False)
               fc_test_ind = res_ind.get_forecast(steps=12)
               pred_test_ind = fc_test_ind.predicted_mean
               pred_test_ind.index = test_ind.index


               res_full_ind    = SARIMAX(series_ind, order=order_ind,
                                         seasonal_order=s_order_ind,
                                         enforce_stationarity=False,
                                         enforce_invertibility=False).fit(disp=False)
               fc_future_ind   = res_full_ind.get_forecast(steps=forecast_months)
               pred_future_ind = fc_future_ind.predicted_mean
               ci_future_ind   = fc_future_ind.conf_int()


           col1, col2 = st.columns(2)
           col1.success(f"✅ Best order: ARIMA{order_ind} × {s_order_ind}")
           col2.info(f"AIC: {res_ind.aic:.2f}")


           mind = get_metrics(test_ind.values, pred_test_ind.values)
           st.markdown("**Test Performance (last 12 months, excl. COVID)**")
           show_metrics(mind, "SARIMA Indian Test")


           fig, ax = plt.subplots(figsize=(13, 4.5))
           ax.plot(train_ind.index, train_ind.values,
                   color="#E67E22", lw=1.8, label="Training")
           ax.plot(test_ind.index, test_ind.values,
                   color="#784212", lw=1.8, linestyle="--", label="Actual (test)")
           ax.plot(pred_future_ind.index, pred_future_ind.values,
                   color="#8E44AD", lw=2, label="Forecast")
           ax.fill_between(pred_future_ind.index,
                           ci_future_ind.iloc[:,0], ci_future_ind.iloc[:,1],
                           color="#8E44AD", alpha=0.15, label="95% CI")
           ax.set_title(f"SARIMA — Indian Arrivals Forecast ({forecast_months} months)",
                        fontweight="bold")
           ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{int(x):,}"))
           ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
           ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
           plt.tight_layout(); st.pyplot(fig); plt.close()


           fc_table_ind = pd.DataFrame({
               "Month": pred_future_ind.index.strftime("%Y-%m"),
               "Forecast": pred_future_ind.values.round(0).astype(int),
               "Lower 95% CI": ci_future_ind.iloc[:,0].round(0).astype(int),
               "Upper 95% CI": ci_future_ind.iloc[:,1].round(0).astype(int),
           })
           st.dataframe(fc_table_ind.style.format({
               "Forecast": "{:,}", "Lower 95% CI": "{:,}", "Upper 95% CI": "{:,}"
           }), use_container_width=True)


   # ─────────────────────────────────────────────────────────────────────────
   # TAB 4: RIDGE REGRESSION
   # ─────────────────────────────────────────────────────────────────────────
   with tab4:
       st.markdown('<div class="section-header">Ridge Regression Models</div>',
                   unsafe_allow_html=True)
       st.markdown('<div class="info-box">📌 Ridge Regression is used instead of OLS '
                   'Linear Regression to handle multicollinearity in the small dataset '
                   '(16 annual observations). Features are standardized before fitting. '
                   'Regularization strength (alpha) is selected via cross-validation.</div>',
                   unsafe_allow_html=True)


       if not run_button:
           st.info("👈 Click **Run Models** in the sidebar to fit Ridge Regression models.")
       else:
           # Build feature lists from sidebar checkboxes
           features_3rd = []
           if use_trekking: features_3rd.append("Trekking_Mtn_Arrivals")
           if use_lag1_3rd: features_3rd.append("Lag1_Third_Country")
           if use_lag2_3rd: features_3rd.append("Lag2_Third_Country")
           if use_income:   features_3rd.append("Tourism_Income_USD_Mn")
           if use_los:      features_3rd.append("Avg_Length_of_Stay")
           features_3rd += ["COVID_Dummy", "Earthquake_Dummy"]


           features_ind = []
           if use_pilgrimage: features_ind.append("Pilgrimage_Arrivals")
           if use_lag1_ind:   features_ind.append("Lag1_Indian")
           if use_holiday:    features_ind.append("Holiday_Arrivals")
           features_ind += ["COVID_Dummy", "Earthquake_Dummy"]


           # ── Third Country ──────────────────────────────────────────────────
           st.markdown("### Third-Country Arrivals")
           st.markdown(f"**Features used:** {', '.join(features_3rd)}")


           with st.spinner("Fitting Ridge Regression for Third-Country..."):
               sub_3rd = lr_3rd_df[["Year","Third_Country_Arrivals"] + features_3rd].dropna()
               X_3rd   = sub_3rd[features_3rd].values
               y_3rd   = sub_3rd["Third_Country_Arrivals"].values
               yrs_3rd = sub_3rd["Year"].values
               sp3     = len(sub_3rd) - n_test_years


               sc3  = StandardScaler()
               Xsc3 = sc3.fit_transform(X_3rd)
               ridge3 = RidgeCV(alphas=[0.01,0.1,1,5,10,50,100,500,1000], cv=5)
               ridge3.fit(Xsc3[:sp3], y_3rd[:sp3])


               yptr3 = ridge3.predict(Xsc3[:sp3])
               ypte3 = ridge3.predict(Xsc3[sp3:])
               ypall3 = ridge3.predict(Xsc3)


           col1, col2 = st.columns(2)
           col1.success(f"✅ Best alpha: {ridge3.alpha_:.2f}")
           col2.info(f"Train years: {yrs_3rd[0]}–{yrs_3rd[sp3-1]} | "
                     f"Test years: {yrs_3rd[sp3]}–{yrs_3rd[-1]}")


           m3_train = get_metrics(y_3rd[:sp3], yptr3)
           m3_test  = get_metrics(y_3rd[sp3:], ypte3)


           col1, col2 = st.columns(2)
           with col1:
               st.markdown("**Train Performance**")
               show_metrics(m3_train, "Train")
           with col2:
               st.markdown("**Test Performance**")
               show_metrics(m3_test, "Test")


           # Future forecast
           prev1_3 = float(y_3rd[-1]); prev2_3 = float(y_3rd[-2])
           trek_last = float(sub_3rd["Trekking_Mtn_Arrivals"].iloc[-1]) if use_trekking else 0
           inc_last  = float(sub_3rd["Tourism_Income_USD_Mn"].iloc[-1]) if use_income else 0
           los_last  = float(sub_3rd["Avg_Length_of_Stay"].iloc[-1]) if use_los else 0
           fut_yrs3, fut_preds3 = [], []
           for i in range(1, forecast_years+1):
               row_d = {}
               if use_trekking: row_d["Trekking_Mtn_Arrivals"] = trek_last
               if use_lag1_3rd: row_d["Lag1_Third_Country"]    = prev1_3
               if use_lag2_3rd: row_d["Lag2_Third_Country"]    = prev2_3
               if use_income:   row_d["Tourism_Income_USD_Mn"] = inc_last
               if use_los:      row_d["Avg_Length_of_Stay"]    = los_last
               row_d["COVID_Dummy"] = 0; row_d["Earthquake_Dummy"] = 0
               row_arr = np.array([[row_d[f] for f in features_3rd]])
               pred = ridge3.predict(sc3.transform(row_arr))[0]
               fut_yrs3.append(int(yrs_3rd[-1])+i)
               fut_preds3.append(pred)
               prev2_3 = prev1_3; prev1_3 = pred


           # Plot
           fig, ax = plt.subplots(figsize=(13, 5))
           ax.plot(yrs_3rd, y_3rd, "o-", color="#2E86C1", lw=2,
                   markersize=6, label="Actual", zorder=3)
           ax.plot(yrs_3rd[:sp3], yptr3, "s--", color="#27AE60",
                   lw=1.8, markersize=6, label="Fitted (train)", zorder=3)
           ax.plot(yrs_3rd[sp3:], ypte3, "^--", color="#E74C3C",
                   lw=1.8, markersize=7, label="Predicted (test)", zorder=3)
           ax.plot(fut_yrs3, fut_preds3, "D-", color="#8E44AD",
                   lw=2, markersize=7,
                   label=f"Forecast ({fut_yrs3[0]}–{fut_yrs3[-1]})", zorder=3)
           for yr, pred in zip(fut_yrs3, fut_preds3):
               ax.annotate(f"{int(pred):,}", xy=(yr,pred), xytext=(0,10),
                           textcoords="offset points", ha="center",
                           fontsize=8, color="#8E44AD", fontweight="bold")
           for yr, act, pred in zip(yrs_3rd[sp3:], y_3rd[sp3:], ypte3):
               ax.annotate(f"A:{int(act):,}\nP:{int(pred):,}", xy=(yr,act),
                           xytext=(0,-32), textcoords="offset points",
                           ha="center", fontsize=7.5, color="#1A5276")
           ax.axvspan(2020, 2021.8, color="#E74C3C", alpha=0.08, label="COVID")
           ax.axvline(2015, color="#E67E22", linestyle="--", lw=1.2,
                      label="Earthquake")
           ax.axvline(yrs_3rd[-1]+0.5, color="#8E44AD", linestyle=":", lw=1.2, alpha=0.6)
           ax.set_title("Ridge LR — Third-Country: Actual vs Predicted vs Forecast",
                        fontweight="bold")
           ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{int(x):,}"))
           ax.set_xticks(range(int(yrs_3rd[0]), fut_yrs3[-1]+1))
           ax.tick_params(axis="x", rotation=45)
           ax.legend(fontsize=9, loc="upper left"); ax.grid(axis="y", alpha=0.3)
           plt.tight_layout(); st.pyplot(fig); plt.close()


           # Coefficients
           st.markdown("**Feature Coefficients**")
           coef_df3 = pd.DataFrame({
               "Feature": features_3rd,
               "Coefficient": ridge3.coef_
           }).sort_values("Coefficient", key=abs, ascending=True)
           fig, ax = plt.subplots(figsize=(7, 3))
           colors = ["#E74C3C" if c < 0 else "#27AE60" for c in coef_df3["Coefficient"]]
           ax.barh(coef_df3["Feature"], coef_df3["Coefficient"], color=colors)
           ax.axvline(0, color="black", lw=0.8)
           ax.set_title("Feature Coefficients (Ridge LR — Third-Country)",
                        fontweight="bold")
           plt.tight_layout(); st.pyplot(fig); plt.close()


           st.markdown("---")


           # ── Indian ─────────────────────────────────────────────────────────
           st.markdown("### Indian Arrivals")
           st.markdown('<div class="warn-box">⚠️ Post-COVID Indian arrivals '
                       '(2023–2025: 319K–438K) exceed training maximum (254K in 2019). '
                       'Model is retained for structural coefficient interpretation only. '
                       'Use SARIMA for Indian forecasting.</div>',
                       unsafe_allow_html=True)
           st.markdown(f"**Features used:** {', '.join(features_ind)}")


           with st.spinner("Fitting Ridge Regression for Indian..."):
               sub_ind = lr_ind_df[["Year","Indian_Arrivals"] + features_ind].dropna()
               X_ind   = sub_ind[features_ind].values
               y_ind   = sub_ind["Indian_Arrivals"].values
               yrs_ind = sub_ind["Year"].values
               spi     = len(sub_ind) - n_test_years


               sci   = StandardScaler()
               Xsci  = sci.fit_transform(X_ind)
               ridgei = RidgeCV(alphas=[0.01,0.1,1,5,10,50,100,500,1000], cv=5)
               ridgei.fit(Xsci[:spi], y_ind[:spi])


               yptri = ridgei.predict(Xsci[:spi])
               yptei = ridgei.predict(Xsci[spi:])


           mi_train = get_metrics(y_ind[:spi], yptri)
           mi_test  = get_metrics(y_ind[spi:], yptei)


           col1, col2 = st.columns(2)
           with col1:
               st.markdown("**Train Performance**")
               show_metrics(mi_train, "Train")
           with col2:
               st.markdown("**Test Performance**")
               show_metrics(mi_test, "Test")


           coef_dfi = pd.DataFrame({
               "Feature": features_ind,
               "Coefficient": ridgei.coef_
           }).sort_values("Coefficient", key=abs, ascending=True)
           fig, ax = plt.subplots(figsize=(7, 3))
           colors = ["#E74C3C" if c < 0 else "#27AE60" for c in coef_dfi["Coefficient"]]
           ax.barh(coef_dfi["Feature"], coef_dfi["Coefficient"], color=colors)
           ax.axvline(0, color="black", lw=0.8)
           ax.set_title("Feature Coefficients (Ridge LR — Indian)", fontweight="bold")
           plt.tight_layout(); st.pyplot(fig); plt.close()


   # ─────────────────────────────────────────────────────────────────────────
   # TAB 5: MODEL COMPARISON
   # ─────────────────────────────────────────────────────────────────────────
   with tab5:
       st.markdown('<div class="section-header">Model Comparison & Summary</div>',
                   unsafe_allow_html=True)


       if not run_button:
           st.info("👈 Click **Run Models** in the sidebar to see comparison.")
       else:
           summary_data = {
               "Model":     ["SARIMA Third-Country","SARIMA Indian",
                              "Ridge LR Third-Country","Ridge LR Indian"],
               "Test R²":   [m3rd["R²"], mind["R²"], m3_test["R²"], mi_test["R²"]],
               "Test MAE":  [m3rd["MAE"], mind["MAE"], m3_test["MAE"], mi_test["MAE"]],
               "Test MAPE": [m3rd["MAPE (%)"], mind["MAPE (%)"],
                             m3_test["MAPE (%)"], mi_test["MAPE (%)"]],
               "Status":    [
                   "✅ Good" if m3rd["R²"] >= 0.8 else "⚠️ Moderate",
                   "✅ Good" if mind["R²"] >= 0.8 else "⚠️ Moderate",
                   "✅ Good" if m3_test["R²"] >= 0.8 else "⚠️ Moderate",
                   "❌ Structural limitation",
               ]
           }
           summary_df = pd.DataFrame(summary_data)


           st.markdown("**Model Performance Summary**")
           st.dataframe(summary_df.style.format({
               "Test R²":   "{:.4f}",
               "Test MAE":  "{:,.0f}",
               "Test MAPE": "{:.2f}%",
           }).background_gradient(subset=["Test R²"], cmap="RdYlGn"),
           use_container_width=True)


           # Average metrics (exclude Indian LR)
           reliable = [m3rd, mind, m3_test]
           avg_r2   = np.mean([m["R²"]       for m in reliable])
           avg_mape = np.mean([m["MAPE (%)"] for m in reliable])


           col1, col2, col3 = st.columns(3)
           col1.metric("Avg R² (3 reliable models)", f"{avg_r2:.4f}",
                       "Target: ≥ 0.80")
           col2.metric("Avg MAPE (3 reliable models)", f"{avg_mape:.2f}%",
                       "Lower is better")
           col3.metric("Indian LR Status", "Structural Break",
                       "Post-COVID surge out-of-range")


           # Bar chart
           fig, axes = plt.subplots(1, 2, figsize=(13, 4))
           colors = ["#27AE60" if r >= 0.8 else "#F39C12" if r >= 0.5
                     else "#E74C3C" for r in summary_df["Test R²"]]
           axes[0].bar(summary_df["Model"], summary_df["Test R²"], color=colors)
           axes[0].axhline(0.8, color="green", linestyle="--", lw=1.2, label="80% target")
           axes[0].set_title("Test R² by Model", fontweight="bold")
           axes[0].set_ylim(-5, 1.1)
           axes[0].tick_params(axis="x", rotation=15)
           axes[0].legend()


           mape_colors = ["#27AE60" if m <= 10 else "#F39C12" if m <= 20
                          else "#E74C3C" for m in summary_df["Test MAPE"]]
           axes[1].bar(summary_df["Model"], summary_df["Test MAPE"], color=mape_colors)
           axes[1].axhline(10, color="green", linestyle="--", lw=1.2, label="10% target")
           axes[1].set_title("Test MAPE (%) by Model", fontweight="bold")
           axes[1].tick_params(axis="x", rotation=15)
           axes[1].legend()


           plt.tight_layout(); st.pyplot(fig); plt.close()


           st.markdown("---")
           st.markdown("**Key Findings**")
           findings = (
               "- **SARIMA Third-Country** performs well: R2=0.90, MAPE=9.22%, "
               "capturing Oct-Nov and Mar-Apr seasonal peaks\n"
               "- **Ridge LR Third-Country** performs excellently: R2=0.97, MAPE=1.35%, "
               "with Trekking arrivals and Lag1 as strongest predictors\n"
               "- **SARIMA Indian** is moderate: R2=0.43 after excluding COVID training years\n"
               "- **Ridge LR Indian** has a structural limitation: post-COVID Indian surge "
               "(319K-438K) far exceeds training maximum (254K), making regression unreliable\n"
               "- **Recommendation**: Use SARIMA for monthly forecasting of both segments; "
               "use Ridge LR for annual structural analysis of third-country arrivals only"
           )
           st.markdown(findings)






   # ─────────────────────────────────────────────────────────────────────────
   # TAB 6: DASHBOARD
   # ─────────────────────────────────────────────────────────────────────────
   with tab6:
       st.markdown('<div class="section-header">Tourism Insights Dashboard</div>',
                   unsafe_allow_html=True)
       st.markdown('<div class="info-box">Interactive overview of Nepal tourism trends '
                   '(2010-2025). Use this to explain key patterns to your professor.</div>',
                   unsafe_allow_html=True)


       import pandas as pd
       import numpy as np
       import matplotlib.pyplot as plt
       import matplotlib.ticker as mticker


       # Load data
       sarima_d = sarima_df.copy()
       sarima_d["Year"]  = sarima_d.index.year
       sarima_d["Month"] = sarima_d.index.month


       lr3 = lr_3rd_df.copy()
       lri = lr_ind_df.copy()


       # ── Row 1: KPI Cards ─────────────────────────────────────────────────
       st.markdown("### Key Performance Indicators (2025 vs 2019 Pre-COVID)")
       total_2019 = int(lr3[lr3["Year"]==2019]["Total_Arrivals"].values[0])
       total_2025 = int(lr3[lr3["Year"]==2025]["Total_Arrivals"].values[0])
       third_2025 = int(lr3[lr3["Year"]==2025]["Third_Country_Arrivals"].values[0])
       indian_2025 = int(lri[lri["Year"]==2025]["Indian_Arrivals"].values[0])
       income_2025 = float(lr3[lr3["Year"]==2025]["Tourism_Income_USD_Mn"].values[0])
       income_2019 = float(lr3[lr3["Year"]==2019]["Tourism_Income_USD_Mn"].values[0])
       recovery_pct = (total_2025 / total_2019) * 100


       col1, col2, col3, col4, col5 = st.columns(5)
       col1.metric("Total Arrivals 2025",    f"{total_2025:,}",
                   f"{((total_2025-total_2019)/total_2019*100):+.1f}% vs 2019")
       col2.metric("Third-Country 2025",     f"{third_2025:,}",
                   f"{(third_2025/total_2025*100):.1f}% of total")
       col3.metric("Indian Arrivals 2025",   f"{indian_2025:,}",
                   f"{(indian_2025/total_2025*100):.1f}% of total")
       col4.metric("Tourism Income 2025",    f"USD {income_2025:.0f}M",
                   f"{((income_2025-income_2019)/income_2019*100):+.1f}% vs 2019")
       col5.metric("COVID Recovery",         f"{recovery_pct:.1f}%",
                   "of 2019 levels reached")


       st.markdown("---")


       # ── Row 2: Total arrivals trend + composition ─────────────────────────
       col1, col2 = st.columns(2)


       with col1:
           st.markdown("**Total Arrivals Trend (2010-2025)**")
           fig, ax = plt.subplots(figsize=(6.5, 3.5))
           ax.fill_between(lr3["Year"], lr3["Third_Country_Arrivals"],
                           color="#2E86C1", alpha=0.7, label="Third-Country")
           ax.fill_between(lr3["Year"],
                           lr3["Third_Country_Arrivals"],
                           lr3["Total_Arrivals"],
                           color="#E67E22", alpha=0.7, label="Indian")
           ax.axvspan(2020, 2021.8, color="#E74C3C", alpha=0.12, label="COVID")
           ax.axvline(2015, color="#8E44AD", linestyle="--", lw=1.2, label="Earthquake")
           ax.set_title("Arrival Composition by Year", fontweight="bold")
           ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x/1000)}K"))
           ax.set_xticks(range(2010, 2026, 2))
           ax.tick_params(axis="x", rotation=45)
           ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
           plt.tight_layout(); st.pyplot(fig); plt.close()


       with col2:
           st.markdown("**Third-Country vs Indian Share Over Time**")
           fig, ax = plt.subplots(figsize=(6.5, 3.5))
           third_pct  = lr3["Third_Country_Arrivals"] / lr3["Total_Arrivals"] * 100
           indian_pct = lri["Indian_Arrivals"] / lr3["Total_Arrivals"] * 100
           ax.stackplot(lr3["Year"], third_pct, indian_pct,
                        labels=["Third-Country %","Indian %"],
                        colors=["#2E86C1","#E67E22"], alpha=0.8)
           ax.set_ylim(0, 100)
           ax.set_ylabel("Share (%)")
           ax.set_title("Visitor Mix: Third-Country vs Indian (%)", fontweight="bold")
           ax.set_xticks(range(2010, 2026, 2))
           ax.tick_params(axis="x", rotation=45)
           ax.legend(fontsize=8, loc="upper left"); ax.grid(axis="y", alpha=0.3)
           plt.tight_layout(); st.pyplot(fig); plt.close()


       # ── Row 3: Purpose of visit + seasonality ─────────────────────────────
       col1, col2 = st.columns(2)


       with col1:
           st.markdown("**Purpose of Visit Trend (2010-2025)**")
           fig, ax = plt.subplots(figsize=(6.5, 3.5))
           ax.plot(lr3["Year"], lr3["Holiday_Arrivals"],
                   "o-", color="#2E86C1", lw=1.8, label="Holiday")
           ax.plot(lr3["Year"], lr3["Trekking_Mtn_Arrivals"],
                   "s-", color="#27AE60", lw=1.8, label="Trekking & Mtn")
           ax.plot(lr3["Year"], lr3["Pilgrimage_Arrivals"],
                   "^-", color="#E67E22", lw=1.8, label="Pilgrimage")
           ax.axvspan(2020, 2021.8, color="#E74C3C", alpha=0.1)
           ax.axvline(2015, color="#8E44AD", linestyle="--", lw=1.2)
           ax.set_title("Arrivals by Purpose of Visit", fontweight="bold")
           ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x/1000)}K"))
           ax.set_xticks(range(2010, 2026, 2))
           ax.tick_params(axis="x", rotation=45)
           ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
           plt.tight_layout(); st.pyplot(fig); plt.close()


       with col2:
           st.markdown("**Average Monthly Seasonality (excl. COVID years)**")
           sea = sarima_d[~sarima_d["Year"].isin([2020,2021])]
           avg_3rd = sea.groupby("Month")["Third_Country_Arrivals"].mean()
           avg_ind = sea.groupby("Month")["Indian_Arrivals"].mean()
           months_lbl = ["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"]
           x = np.arange(12)
           fig, ax = plt.subplots(figsize=(6.5, 3.5))
           ax.bar(x - 0.2, avg_3rd.values, 0.4, color="#2E86C1",
                  alpha=0.8, label="Third-Country")
           ax.bar(x + 0.2, avg_ind.values,  0.4, color="#E67E22",
                  alpha=0.8, label="Indian")
           ax.set_xticks(x); ax.set_xticklabels(months_lbl, rotation=45)
           ax.set_title("Avg Monthly Arrivals by Segment", fontweight="bold")
           ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x/1000)}K"))
           ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
           plt.tight_layout(); st.pyplot(fig); plt.close()


       # ── Row 4: Hotel supply + income ──────────────────────────────────────
       col1, col2 = st.columns(2)


       with col1:
           st.markdown("**Hotel Supply Growth vs Arrivals**")
           fig, ax1 = plt.subplots(figsize=(6.5, 3.5))
           ax2 = ax1.twinx()
           ax1.bar(lr3["Year"], lr3["Total_Beds"],
                   color="#AED6F1", alpha=0.7, label="Total Beds")
           ax2.plot(lr3["Year"], lr3["Total_Arrivals"],
                    "o-", color="#E74C3C", lw=2, label="Total Arrivals")
           ax1.set_ylabel("Total Beds", color="#2E86C1")
           ax2.set_ylabel("Total Arrivals", color="#E74C3C")
           ax1.set_title("Hotel Capacity vs Arrivals", fontweight="bold")
           ax1.set_xticks(range(2010, 2026, 2))
           ax1.tick_params(axis="x", rotation=45)
           ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x/1000)}K"))
           ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x/1000)}K"))
           lines1, labels1 = ax1.get_legend_handles_labels()
           lines2, labels2 = ax2.get_legend_handles_labels()
           ax1.legend(lines1+lines2, labels1+labels2, fontsize=8)
           ax1.grid(axis="y", alpha=0.3)
           plt.tight_layout(); st.pyplot(fig); plt.close()


       with col2:
           st.markdown("**Tourism Income (USD Million) vs Arrivals**")
           lr3_clean = lr3.dropna(subset=["Tourism_Income_USD_Mn"])
           fig, ax1 = plt.subplots(figsize=(6.5, 3.5))
           ax2 = ax1.twinx()
           ax1.bar(lr3_clean["Year"], lr3_clean["Tourism_Income_USD_Mn"],
                   color="#A9DFBF", alpha=0.8, label="Income (USD Mn)")
           ax2.plot(lr3_clean["Year"], lr3_clean["Total_Arrivals"],
                    "o-", color="#E74C3C", lw=2, label="Total Arrivals")
           ax1.set_ylabel("Income (USD Mn)", color="#27AE60")
           ax2.set_ylabel("Total Arrivals", color="#E74C3C")
           ax1.set_title("Tourism Income vs Arrivals", fontweight="bold")
           ax1.set_xticks(range(2010, 2026, 2))
           ax1.tick_params(axis="x", rotation=45)
           ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x/1000)}K"))
           lines1, labels1 = ax1.get_legend_handles_labels()
           lines2, labels2 = ax2.get_legend_handles_labels()
           ax1.legend(lines1+lines2, labels1+labels2, fontsize=8)
           ax1.grid(axis="y", alpha=0.3)
           plt.tight_layout(); st.pyplot(fig); plt.close()


       # ── Row 5: YoY growth + avg length of stay ────────────────────────────
       col1, col2 = st.columns(2)


       with col1:
           st.markdown("**Year-on-Year Growth Rate (%)**")
           yoy = lr3["Total_Arrivals"].pct_change() * 100
           colors_yoy = ["#27AE60" if v >= 0 else "#E74C3C" for v in yoy]
           fig, ax = plt.subplots(figsize=(6.5, 3.5))
           ax.bar(lr3["Year"], yoy, color=colors_yoy, alpha=0.85)
           ax.axhline(0, color="black", lw=0.8)
           ax.axvspan(2020, 2021.8, color="#E74C3C", alpha=0.08, label="COVID")
           ax.axvline(2015, color="#8E44AD", linestyle="--", lw=1.2, label="Earthquake")
           for yr, val in zip(lr3["Year"], yoy):
               if not pd.isna(val):
                   ax.text(yr, val + (1.5 if val >= 0 else -3.5),
                           f"{val:.0f}%", ha="center", fontsize=7, fontweight="bold")
           ax.set_title("Annual Growth Rate of Total Arrivals", fontweight="bold")
           ax.set_ylabel("Growth (%)")
           ax.set_xticks(range(2010, 2026, 2))
           ax.tick_params(axis="x", rotation=45)
           ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
           plt.tight_layout(); st.pyplot(fig); plt.close()


       with col2:
           st.markdown("**Average Length of Stay (days)**")
           fig, ax = plt.subplots(figsize=(6.5, 3.5))
           ax.plot(lr3["Year"], lr3["Avg_Length_of_Stay"],
                   "o-", color="#8E44AD", lw=2, markersize=7)
           ax.fill_between(lr3["Year"], lr3["Avg_Length_of_Stay"],
                           alpha=0.15, color="#8E44AD")
           for yr, val in zip(lr3["Year"], lr3["Avg_Length_of_Stay"]):
               ax.text(yr, val+0.15, f"{val:.1f}", ha="center",
                       fontsize=7.5, color="#6C3483")
           ax.axvspan(2020, 2021.8, color="#E74C3C", alpha=0.08, label="COVID")
           ax.axvline(2015, color="#E67E22", linestyle="--", lw=1.2, label="Earthquake")
           ax.set_title("Avg Length of Stay Per Tourist (days)", fontweight="bold")
           ax.set_ylabel("Days")
           ax.set_ylim(10, 18)
           ax.set_xticks(range(2010, 2026, 2))
           ax.tick_params(axis="x", rotation=45)
           ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
           plt.tight_layout(); st.pyplot(fig); plt.close()


       # ── Row 6: Correlation heatmap ─────────────────────────────────────────
       st.markdown("**Correlation Heatmap — All Variables vs Third-Country Arrivals**")
       corr_cols = ["Third_Country_Arrivals","Trekking_Mtn_Arrivals",
                    "Holiday_Arrivals","Pilgrimage_Arrivals",
                    "Avg_Length_of_Stay","Tourism_Income_USD_Mn",
                    "Total_Hotels","Total_Beds","Intl_Flights_Total"]
       corr_df = lr3[corr_cols].dropna().corr()
       fig, ax = plt.subplots(figsize=(10, 6))
       mask = np.zeros_like(corr_df, dtype=bool)
       mask[np.triu_indices_from(mask)] = True
       sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="RdYlGn",
                   center=0, linewidths=0.5, ax=ax, mask=mask,
                   cbar_kws={"label": "Correlation"})
       ax.set_title("Variable Correlation Matrix", fontweight="bold", pad=10)
       plt.tight_layout(); st.pyplot(fig); plt.close()


       st.markdown("---")
       st.markdown("**What These Charts Tell You**")
       st.markdown(
           "- **Arrival Composition**: Indian share has grown post-COVID from ~20%% to ~33%%\n"
           "- **Seasonality**: Third-country peaks Oct-Nov and Mar-Apr (trekking seasons); "
           "Indian peaks May-Jun (pilgrimage/summer)\n"
           "- **Purpose**: Holiday is the largest category but Trekking drives third-country uniqueness\n"
           "- **Hotel supply** has grown steadily (744 hotels in 2010 to 1,578 in 2025) "
           "but arrivals are more volatile\n"
           "- **COVID impact**: 2020 saw -83%% drop in arrivals and -88%% drop in income\n"
           "- **Length of stay** increased during COVID (fewer but longer-staying tourists) "
           "and again in 2025 (16.3 days)\n"
           "- **Strong correlations**: Income, Hotels, Beds all highly correlated with arrivals "
           "(multicollinearity risk in regression)"
       )



