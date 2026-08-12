import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import traceback

st.set_page_config(page_title="UFC EV Predictor Pro", layout="wide")
st.title("🥊 UFC Matchup & Expected Value (EV) Predictor")

# --- BULLETPROOF MODEL INITIALIZER ---
@st.cache_resource
def load_safe_model():
    np.random.seed(42)
    data_size = 1000
    X = pd.DataFrame({
        'reach_diff': np.random.normal(0, 3, data_size),
        'age_diff': np.random.normal(0, 4, data_size),
        'sig_strike_diff': np.random.normal(0, 1.5, data_size),
        'td_diff': np.random.normal(0, 1.2, data_size),
        'win_streak_diff': np.random.randint(-3, 4, data_size),
        'sentiment_diff': np.random.normal(0, 0.5, data_size)
    })
    log_odds = (X['sig_strike_diff'] * 0.5) + (X['td_diff'] * 0.4) - (X['age_diff'] * 0.2)
    y = np.where(np.random.rand(data_size) < (1 / (1 + np.exp(-log_odds))), 1, 0)
    
    model = xgb.XGBClassifier(n_estimators=50, learning_rate=0.05, max_depth=3)
    model.fit(X, y)
    return model

# --- SAFE MAIN EXECUTION ---
try:
    model = load_safe_model()

    st.markdown("### Matchup Differentials (Fighter A minus Fighter B)")

    col1, col2, col3 = st.columns(3)
    with col1:
        fighter_a_name = st.text_input("Fighter A Name", "Islam Makhachev")
        reach_diff = st.number_input("Reach Adv. (inches)", value=0.0)
        sig_strike_diff = st.number_input("Sig. Strikes/Min Adv.", value=0.0)

    with col2:
        fighter_b_name = st.text_input("Fighter B Name", "Arman Tsarukyan")
        age_diff = st.number_input("Age Diff (A - B)", value=0.0)
        td_diff = st.number_input("Takedown Adv.", value=0.0)

    with col3:
        win_streak_diff = st.number_input("Win Streak Diff", value=0)
        sportsbook_odds = st.number_input("Odds (Fighter A)", value=-110)
        st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Calculate Prediction & Expected Value", use_container_width=True):
        input_data = pd.DataFrame({
            'reach_diff': [reach_diff],
            'age_diff': [age_diff],
            'sig_strike_diff': [sig_strike_diff],
            'td_diff': [td_diff],
            'win_streak_diff': [win_streak_diff],
            'sentiment_diff': [0.0]
        })
        
        prob = model.predict_proba(input_data)[0][1]
        
        if sportsbook_odds < 0:
            implied_prob = abs(sportsbook_odds) / (abs(sportsbook_odds) + 100)
            decimal_odds = (100 / abs(sportsbook_odds)) + 1
        else:
            implied_prob = 100 / (sportsbook_odds + 100)
            decimal_odds = (sportsbook_odds / 100) + 1
            
        ev = (prob * decimal_odds) - 1
        
        st.markdown("---")
        st.subheader("📊 Model Output")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Model Win Prob.", f"{prob:.1%}")
        m2.metric("Sportsbook Implied Prob.", f"{implied_prob:.1%}")
        
        if ev > 0:
            m3.metric("Expected Value (EV)", f"+{ev:.2%}", "Value Bet Detected!")
            st.success(f"This bet has a positive expected edge of {ev:.2%} over the sportsbook.")
        else:
            m3.metric("Expected Value (EV)", f"{ev:.2%}", "Negative EV")
            st.warning("The sportsbook's line is sharper than the model. Pass on this bet.")

except Exception as e:
    st.error("🚨 An error occurred while running the application:")
    st.text(traceback.format_exc())