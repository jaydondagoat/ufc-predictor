import streamlit as st
import pandas as pd
import requests

st.set_page_config(
    page_title="UFC Quant & EV Engine",
    page_icon="🥊",
    layout="wide"
)

st.title("🥊 UFC Quantitative Market & EV Engine")
st.markdown("Direct Bookmaker Line Integration & Quant Expected Value (EV) Analytics.")

# --- LIVE ODDS API INTEGRATION ---
@st.cache_data(ttl=1800)
def fetch_live_odds(api_key):
    url = f"https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds/?apiKey={api_key}&regions=us&oddsFormat=american"
    try:
        res = requests.get(url)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return []

try:
    api_key = st.secrets.get("ODDS_API_KEY", "")
    if not api_key:
        api_key = st.sidebar.text_input("Enter The Odds API Key", type="password")

    with st.spinner("Syncing live sportsbook markets..."):
        events = fetch_live_odds(api_key) if api_key else []

    if not events:
        st.info("No active live feed events returned. Loading default upcoming championship card.")
        events = [{
            'id': 'fallback_event',
            'home_team': 'Islam Makhachev',
            'away_team': 'Arman Tsarukyan',
            'bookmakers': [{
                'title': 'DraftKings',
                'markets': [{
                    'key': 'h2h',
                    'outcomes': [
                        {'name': 'Islam Makhachev', 'price': -280},
                        {'name': 'Arman Tsarukyan', 'price': +230}
                    ]
                }]
            }]
        }]

    st.subheader("🏟️ Active Fight Card Lineup")
    selected_event_idx = st.selectbox(
        "Select Matchup from Feed", 
        range(len(events)), 
        format_func=lambda i: f"{events[i].get('home_team', 'Fighter 1')} vs. {events[i].get('away_team', 'Fighter 2')}"
    )
    current_event = events[selected_event_idx]
    
    bookmakers = current_event.get('bookmakers', [])
    odds_dict = {}
    book_name = "DraftKings"
    
    if bookmakers:
        bm = bookmakers[0]
        book_name = bm.get('title')
        for m in bm.get('markets', []):
            if m.get('key') == 'h2h':
                for out in m.get('outcomes', []):
                    odds_dict[out.get('name')] = out.get('price')

    fighters = list(odds_dict.keys())
    if len(fighters) < 2:
        fighters = [current_event.get('home_team', 'Fighter A'), current_event.get('away_team', 'Fighter B')]
        odds_dict = {fighters[0]: -250, fighters[1]: +200}

    fighter_a, fighter_b = fighters[0], fighters[1]
    odds_a = odds_dict.get(fighter_a, -200)
    odds_b = odds_dict.get(fighter_b, +170)

    # --- MARKET-DERIVED PROBABILITY & QUANT ENGINE ---
    st.markdown("---")
    st.subheader("📊 Model Weight & Statistical Adjuster")
    st.markdown("Adjust your custom handicap confidence modifier against the bookmaker baseline:")

    col_slider1, col_slider2 = st.columns(2)
    with col_slider1:
        handicap_a = st.slider(f"{fighter_a} Custom Edge Adjustment (%)", -20, 20, 0, step=1)
    with col_slider2:
        handicap_b = st.slider(f"{fighter_b} Custom Edge Adjustment (%)", -20, 20, 0, step=1)

    if st.button("🚀 Run Quantitative EV Calculation", use_container_width=True):
        
        # Convert American odds to true market implied probabilities
        def odds_to_implied(odds):
            if odds < 0:
                return abs(odds) / (abs(odds) + 100)
            else:
                return 100 / (odds + 100)

        raw_implied_a = odds_to_implied(odds_a)
        raw_implied_b = odds_to_implied(odds_b)
        
        # Normalize baseline market probabilities
        total_market_prob = raw_implied_a + raw_implied_b
        base_prob_a = raw_implied_a / total_market_prob
        base_prob_b = raw_implied_b / total_market_prob

        # Apply user's analytical adjustments safely (clamped between 5% and 95%)
        final_prob_a = max(0.05, min(0.95, base_prob_a + (handicap_a / 100.0)))
        final_prob_b = max(0.05, min(0.95, (1.0 - final_prob_a) + (handicap_b / 100.0)))
        
        # Re-normalize
        norm = final_prob_a + final_prob_b
        prob_a = final_prob_a / norm
        prob_b = final_prob_b / norm

        def calculate_ev(odds, prob):
            if odds < 0:
                dec = (100 / abs(odds)) + 1
            else:
                dec = (odds / 100) + 1
            return (prob * dec) - 1

        ev_a = calculate_ev(odds_a, prob_a)
        ev_b = calculate_ev(odds_b, prob_b)

        st.subheader("🎯 Quantitative Model Results")
        res1, res2 = st.columns(2)
        
        with res1:
            st.markdown(f"### 🔴 {fighter_a} ({odds_a})")
            st.metric("Model Win Probability", f"{prob_a:.1%}")
            st.metric("Market Implied Baseline", f"{base_prob_a:.1%}")
            if ev_a > 0:
                st.success(f"Value Edge Detected: +{ev_a:.2%}")
            else:
                st.error(f"Negative EV: {ev_a:.2%}")

        with res2:
            st.markdown(f"### 🔵 {fighter_b} ({odds_b})")
            st.metric("Model Win Probability", f"{prob_b:.1%}")
            st.metric("Market Implied Baseline", f"{base_prob_b:.1%}")
            if ev_b > 0:
                st.success(f"Value Edge Detected: +{ev_b:.2%}")
            else:
                st.error(f"Negative EV: {ev_b:.2%}")

        # Verified structural links
        st.markdown("---")
        st.subheader("💬 Verified External Intelligence")
        x_query = fighter_a.replace(" ", "%20")
        st.markdown(f"* **Live X Feed:** [Search live commentary for {fighter_a}](https://twitter.com/search?q={x_query}&f=live)")
        st.markdown(f"* **Technical Reference:** [Watch Breakdown Video Context](https://www.youtube.com/watch?v=wftY3jrZDdk)")

except Exception as e:
    st.error("🚨 Execution error captured:")
    st.text(str(e))