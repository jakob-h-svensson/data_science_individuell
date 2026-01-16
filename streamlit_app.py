import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px
import datetime as dt
from xgboost import XGBClassifier
import os
import json
from datetime import datetime, date, timedelta

# --- Setting visuals --- #
st.set_page_config(page_title="Data Science Project: Ekonomiresan", layout="wide")

# --- Settings and data --- #

# --- Pathing --- #

DATA_PATH = "data_science_project/data"
RAW_DATA_PATH = "data_science_project/raw_data"
PROCESSED_DATA_PATH = "data_science_project/processed_data"


# Defining tickers/stocks
omxs30_stocks = {
    "ALFA.ST": "Alfa Laval",
    "ASSA-B.ST": "Assa Abloy",
    "ATCO-A.ST": "Atlas Copco A",
    "ATCO-B.ST": "Atlas Copco B",
    "AZN.ST": "AstraZeneca",
    "BOL.ST": "Boliden",
    "ELUX-B.ST": "Electrolux",
    "ERIC-B.ST": "Ericsson",
    "ESSITY-B.ST": "Essity",
    "GETI-B.ST": "Getinge",
    "HEXA-B.ST": "Hexagon",
    "HM-B.ST": "H&M",
    "INVE-B.ST": "Investor B",
    "NDA-SE.ST": "Nordea",
    "SAND.ST": "Sandvik",
    "SCA-B.ST": "SCA",
    "SEB-A.ST": "SEB A",
    "SHB-A.ST": "Handelsbanken A",
    "SKF-B.ST": "SKF B",
    "SSAB-A.ST": "SSAB A",
    "SWED-A.ST": "Swedbank A",
    "TELIA.ST": "Telia Company",
    "VOLV-B.ST": "Volvo B",
    "KINV-B.ST": "Kinnevik B",
    "LATO-B.ST": "Latour B",
    "NIBE-B.ST": "Nibe Industrier",
    "SAAB-B.ST": "Saab B",
    "LIFCO-B.ST": "Lifco B",
    "EVO.ST": "Evolution",
    "SINCH.ST": "Sinch"
}

nasdaq_stocks = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "META": "Meta Platforms",
    "NVDA": "Nvidia",
    "TSLA": "Tesla",
    "AVGO": "Broadcom",
    "INTC": "Intel",
    "AMD": "Advanced Micro Devices",
    "CSCO": "Cisco Systems",
    "PEP": "PepsiCo",
    "COST": "Costco Wholesale",
    "ADBE": "Adobe",
    "NFLX": "Netflix",
    "TXN": "Texas Instruments",
    "QCOM": "Qualcomm",
    "AMGN": "Amgen",
    "HON": "Honeywell",
    "INTU": "Intuit",
    "MDLZ": "Mondelez International",
    "SBUX": "Starbucks",
    "BKNG": "Booking Holdings",
    "GILD": "Gilead Sciences",
    "ISRG": "Intuitive Surgical",
    "LRCX": "Lam Research",
    "MU": "Micron Technology",
    "REGN": "Regeneron Pharmaceuticals",
    "VRTX": "Vertex Pharmaceuticals",
    "ADP": "Automatic Data Processing"
}

# Journey start and end dates
start = "2014-01-01"
end = "2025-12-31"


# Fetching data from yfinance
def fetch_data(tickers, start, end):
    # Ev lägga in status-text och progress-bar
    #prog_bar = st.progress(0)
    #status_text = st.empty()
    list_tickers = []
    
    for ticker, name in tickers.items():
        list_tickers.append(ticker)

    df = yf.download(
            list_tickers,
            start=start,
            end=end,
            progress=False
        )

    return df

# Returning both lists in the same function (for progress display purposes)
# Using st.cache_data to cache the data (faster slider loading)
@st.cache_data
def get_data():
    raw_omxs30 = fetch_data(omxs30_stocks, start, end)
    raw_nasdaq = fetch_data(nasdaq_stocks, start, end)

    return raw_omxs30, raw_nasdaq

@st.cache_data
def close_and_volume(df):
    """Extracts the close and volume columns from the dataframe 
    and returns them as two separate dataframes."""
    
    df_close = df['Close']
    df_volume = df['Volume']

    # Moving the index to a column
    df_close = df_close.reset_index()
    df_volume = df_volume.reset_index()

    # Renaming the index to Date
    df_close = df_close.rename(columns={'index': 'Date'})
    df_volume = df_volume.rename(columns={'index': 'Date'})

    return df_close, df_volume

@st.cache_data
def last_30_days_change(df):
    """Calculates the percentage change of the close column over the last 30 days."""
    df_pct_last_30 = df['Close']/df['Close'].shift(30) - 1
    df_pct_last_30 = df_pct_last_30.reset_index()
    df_pct_last_30 = df_pct_last_30.rename(columns={'index': 'Date'})

    return df_pct_last_30

def get_nearest_trading_day(df, selected_date):
    return (
        df
        .set_index("Date")
        .sort_index()
        .loc[:selected_date]
        .iloc[-1]
    )



#--- Processing the data ---#
raw_omxs30, raw_nasdaq = get_data()
omx_close, omx_volume = close_and_volume(raw_omxs30)
nasdaq_close, nasdaq_volume = close_and_volume(raw_nasdaq)
omx_last_30_days_change = last_30_days_change(raw_omxs30)
nasdaq_last_30_days_change = last_30_days_change(raw_nasdaq)
#------#

#--- Creating weighted index ---#
omx_index = raw_omxs30
nasdaq_index = raw_nasdaq

def extract_close_for_index(df, market):
    df_close = df["Close"].reset_index()

    df_long = df_close.melt(
        id_vars="Date",
        var_name="ticker",
        value_name="close"
    )

    df_long["market"] = market
    df_long = df_long.dropna()
    df_long = df_long.sort_values(["ticker", "Date"])

    return df_long

omx_prices = extract_close_for_index(raw_omxs30, "Sverige")
nasdaq_prices = extract_close_for_index(raw_nasdaq, "USA")

prices = pd.concat([omx_prices, nasdaq_prices], ignore_index=True)

def build_index(df):
    df = df.copy()
    df["norm_close"] = df.groupby("ticker")["close"].transform(
        lambda x: x / x.iloc[0]
    )
    return (
        df.groupby("Date")["norm_close"]
        .mean()
        .reset_index(name="index_value")
    )

omx_index = build_index(omx_prices)
nasdaq_index = build_index(nasdaq_prices)

omx_index.to_csv(f"{PROCESSED_DATA_PATH}/omx_index.csv", index=False)
nasdaq_index.to_csv(f"{PROCESSED_DATA_PATH}/nasdaq_index.csv", index=False)

omx_index['Market'] = 'Sverige'
nasdaq_index['Market'] = 'USA'

def last_30_days_change(df):
    """Calculates the percentage change of the close column over the last 30 days."""
    df['pct_change'] = df['index_value']/df['index_value'].shift(30) - 1

    return df

omx_index_last_30_days_change = last_30_days_change(omx_index)
nasdaq_index_last_30_days_change = last_30_days_change(nasdaq_index)


#--- Get events ---#

events_df = pd.read_csv(f"{PROCESSED_DATA_PATH}/events.csv")
events_df["date"] = pd.to_datetime(events_df["date"])
events_df = events_df.sort_values("date", ascending=False)



tab1_landing, tab2_visuals = st.tabs(["Information", "Visualiseringar"])
with tab1_landing:
    st.title("Ekonomiresan: 10 år av turbulens och tillväxt")
    st.markdown("""
    ### En interaktiv visualisering av marknaden 2015–2025
    
    Välkommen till **Ekonomiresan**. De senaste tio åren har varit en av de mest händelserika perioderna i modern ekonomisk historia. 
    Vi har rört oss från en era av nollräntor och låg inflation, genom en global pandemi och leveranskedjekriser, till krig i Europa, energichocker och en explosiv AI-boom.

    Denna applikation syftar till att besvara en central fråga:
    > *Hur reagerar egentligen börsen på stora geopolitiska och makroekonomiska händelser?*
    
    Genom att kombinera finansiell data med en tidslinje av nyhetshändelser kan du här utforska sambanden mellan rubrikerna och kursrörelserna.
    """)

    st.divider()

    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.subheader("Marknaderna vi följer")
        st.markdown("""
        Analys görs av två distinkta marknader genom att jämföra de 30 mest tongivande bolagen från respektive marknad.
        
        **Urvalsindex Sverige**
        *Det svenska industriundret.*
        Listan innehåller de 30 mest omsatta aktierna på Stockholmsbörsen. Här dominerar verkstad (t.ex. Atlas Copco, Volvo), bank och traditionell industri, kompletterat med moderna tillväxtbolag som Evolution och Sinch.
        
        **Urvalsindex USA**
        *Den globala tillväxtmotorn.*
        Jag har valt ut 30 av de största och mest inflytelserika bolagen på Nasdaq-börsen. Här ser vi tydliga effekter av digitalisering och AI-utveckling genom jättar som Apple, Nvidia och Microsoft, men även bioteknik och modern handel.
        """)

    with col_info2:
        st.subheader("Metodik & Data")
        st.markdown("""
        För att möjliggöra en rättvis jämförelse över tid har jag konstruerat egna index baserade på aktuella aktiekorgar:
        
        * **Normerad utveckling:** Alla aktiekurser är normerade från startdatumet. Indexet du ser i grafen är ett genomsnitt av den procentuella utvecklingen för bolagen i korgen.
        * **Data:** Datan hämtas via `yfinance`.
        * **Machine Learning:** I bakgrunden används en XGBoost-modell (Extreme Gradient Boosting) för att analysera mönster och samband mellan historisk data och marknadsrörelser.
        """)

    st.info("**Gå till fliken 'Visualiseringar'** för att starta tidsresan. Använd tidsreglaget för att se hur världen – och din portfölj – förändrades, månad för månad.")

with tab2_visuals:
    if "selected_date" not in st.session_state:
        st.session_state.selected_date = dt.date(2015, 1, 1)
    col1, col2, col3 = st.columns(3)
    with col1:
        subcol1, subcol2, subcol3 = st.columns(3)
        with subcol1:
            if st.button("Hoppa 10 dagar"):
                st.session_state.selected_date += dt.timedelta(days=10)
        with subcol2:
            if st.button("Hoppa 30 dagar"):
                st.session_state.selected_date += dt.timedelta(days=30)

    col1, col2 = st.columns([5, 2])
    with col1:
        selected_date = st.slider(
            "Välj ett datum",
            min_value=dt.date(2015, 1, 1),
            max_value=dt.date(2025, 12, 31),
            value=st.session_state.selected_date,
            format="YYYY-MM-DD"
        )

        selected_ts = pd.Timestamp(selected_date)

        # Filtering the frames
        omx_index_selected = omx_index[(omx_index['Date'] <= selected_ts) & (omx_index['Date'] >= "2015-01-01")]
        nasdaq_index_selected = nasdaq_index[(nasdaq_index['Date'] <= selected_ts) & (nasdaq_index['Date'] >= "2015-01-01")]
              
        fig = px.line(pd.concat([omx_index_selected, nasdaq_index_selected]), x="Date", y="index_value", color='Market', title="Index Sverige vs USA")

        for _, row in events_df.iterrows():
            if row['date'] <= selected_ts:

                fig.add_vline(
                    x=row["date"],
                    line_width=1,
                    line_dash="dot",
                    line_color="red",
                    opacity=0.3,
                )
                if row["date"] <= selected_ts and row["date"] >= selected_ts - dt.timedelta(days=182):
                    fig.add_annotation(
                        x=row["date"],
                        y=1.02,
                        yref="paper",
                        text=row["event"],
                        showarrow=False,
                        textangle=-90,
                        font=dict(size=15, color="red"),
                        align="left"
                    )

                
        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:
        st.subheader("30 dagars marknadsändringar")

        # Filtering the frames
        omx_row = get_nearest_trading_day(omx_last_30_days_change, selected_ts)
        nasdaq_row = get_nearest_trading_day(nasdaq_last_30_days_change, selected_ts)

        omx_row = omx_row.dropna()
        nasdaq_row = nasdaq_row.dropna()

        omx_index_row = get_nearest_trading_day(omx_index_last_30_days_change, selected_ts)
        nasdaq_index_row = get_nearest_trading_day(nasdaq_index_last_30_days_change, selected_ts)

        # Winner/Loser
        sorted_omx = omx_row.sort_values(ascending=False)
        sorted_nasdaq = nasdaq_row.sort_values(ascending=False)

        omx_index_value = omx_index_row['pct_change']*100
        winner_omx_name = omxs30_stocks[sorted_omx.index[0]]
        winner_omx_value = sorted_omx.values[0]*100
        loser_omx_name = omxs30_stocks[sorted_omx.index[-1]]
        loser_omx_value = sorted_omx.values[-1]*100

        nasdaq_index_value = nasdaq_index_row['pct_change']*100
        winner_nasdaq_name = nasdaq_stocks[sorted_nasdaq.index[0]]
        winner_nasdaq_value = sorted_nasdaq.values[0]*100
        loser_nasdaq_name = nasdaq_stocks[sorted_nasdaq.index[-1]]
        loser_nasdaq_value = sorted_nasdaq.values[-1]*100

        col1, col2 = st.columns(2)
        with col1:
            col1.subheader("Sverige")
            st.metric(label="Index", value=f"{omx_index_value:.2f}%")
            st.metric(label=f"Vinnare: {winner_omx_name}", value=f"{winner_omx_value:.2f}%")
            st.metric(label=f"Förlorare: {loser_omx_name}", value=f"{loser_omx_value:.2f}%")
            
        with col2:
            col2.subheader("USA")
            st.metric(label="Index", value=f"{nasdaq_index_value:.2f}%")
            st.metric(label=f"Vinnare: {winner_nasdaq_name}", value=f"{winner_nasdaq_value:.2f}%")
            st.metric(label=f"Förlorare: {loser_nasdaq_name}", value=f"{loser_nasdaq_value:.2f}%")
        
    for _, row in events_df.iterrows():
        if row['date'] <= selected_ts:
            st.markdown(f"""
            #### :blue-background[{row['date'].date()} - {row['event']}]
            {row['description']}
            """)
            

    
    


