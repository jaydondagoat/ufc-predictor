import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import cloudscraper
from bs4 import BeautifulSoup
import tweepy
from textblob import TextBlob
from youtubesearchpython import VideosSearch
from youtube_transcript_api import YouTubeTranscriptApi
import traceback

st.set_page_config(page_title="UFC EV Predictor Pro", layout="wide")
st.title("🥊 UFC Matchup & EV Predictor (Multi-File Data Engine)")

# --- 1. SCRAPING, NLP & YOUTUBE FUNCTIONS ---

def get_tapology_record(fighter_name):
    scraper = cloudscraper.create_scraper() 
    formatted_name = fighter_name.lower().replace(" ", "-")
    url = f"https://www.tapology.com/fightcenter/fighters/{formatted_name}"
    
    try:
        response = scraper.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        record_div = soup.find('div', class_='record')
        if record_div:
            return record_div.text.strip()
        return "Record not found."
    except Exception:
        return "Tapology block / Failed."

def get_twitter_sentiment(fighter_name, api_key, api_secret, access_token, access_secret):
    try:
        auth = tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_secret)
        api = tweepy.API(auth)
        query = f"{fighter_name} (injury OR weight OR camp OR staph) -is:retweet"
        tweets = api.search_tweets(q=query, lang="en", count=15)
        
        if not tweets:
            return 0.0
            
        sentiment_score = 0
        for tweet in tweets:
            analysis = TextBlob(tweet.text)
            sentiment_score += analysis.sentiment.polarity
            
        return sentiment_score / len(tweets)
    except Exception:
        return 0.0

def get_youtube_predictions(fighter_a, fighter_b):
    query = f"{fighter_a} vs {fighter_b} UFC prediction"
    try:
        search = VideosSearch(query, limit=3)
        results = search.result()['result']
        
        videos_data = []
        for vid in results:
            vid_id = vid['id']
            title = vid['title']
            channel = vid['channel']['name']
            link = vid['link']
            
            try:
                transcript = YouTubeTranscriptApi.get_transcript(vid_id)
                start_idx = int(len(transcript) * 0.75)
                closing_statements = " ".join([t['text'].replace('\n', ' ') for t in transcript[start_idx:start_idx+20]])
                
                videos_data.append({
                    'channel': channel,
                    'title': title,
                    'link': link,
                    'quote': closing_statements + "..."
                })
            except Exception:
                videos_data.append({
                    'channel': channel,
                    'title': title,
                    'link': link,
                    'quote': "Transcript disabled or auto-generated subs unavailable."
                })
        return videos_data
    except Exception as e:
        return [{"error": str(e)}]

# --- 2. MULTI-FILE ML ENGINE ---
@st.cache_resource
def load_or_train_model():
    model_file = 'ufc_multi_file_model.json'
    model = xgb.XGBClassifier()
    
    if os.path.exists(model_file):
        try:
            model.load_model(model_file)
            return model
        except Exception:
            pass # Retrain if model file is corrupted
        
    with st.spinner("Merging your CSV files and training AI model..."):
        fights_path = 'fights.csv'
        fighters_path = 'fighters.csv'
        events_path = 'events.csv'
        
        if os.path.exists(fights_path) and os.path.exists(fighters_path):
            df_fights = pd.read_csv(fights_path)
            df_fighters = pd.read_csv(fighters_path)
            
            if os.path.exists(events_path):
                df_events = pd.read_csv(events_path)
            
            if 'reach_diff' not in df_fights.columns:
                df_fights['reach_diff'] = df_fights.get('R_reach', 0) - df_fights.get('B_reach', 0)
                df_fights['age_diff'] = df_fights.get('R_age', 0) - df_fights.get('B_age', 0)
                df_fights['sig_strike_diff'] = df_fights.get('R_SIG_STR_landed', 0) - df_fights.get('B_SIG_STR_landed', 0)
                df_fights['td_diff'] = df_fights.get('R_TD_landed', 0) - df_fights.get('B_TD_landed', 0)
                df_fights['win_streak_diff'] = df_fights.get('R_current_win_streak', 0) - df_fights.get('B_current_win_streak', 0)
                df_fights['sentiment_diff'] = 0.0
                
                if 'Winner' in df_fights.columns:
                    df_fights['fighter_a_win'] = df_fights['Winner'].apply(lambda x: 1 if str(x).strip().lower() in ['red', 'fighter 1'] else 0)
                else:
                    df_fights['fighter_a_win'] = 1
            
            features = ['reach_diff', 'age_diff', 'sig_strike_diff', 'td_diff', 'win_streak_diff', 'sentiment_diff']
            for col in features:
                if col not in df_fights.columns:
                    df_fights[col] = 0.0
                    
            X = df_fights[features].fillna(0)
            y = df_fights.get('fighter_a_win', pd.Series([1]*len(df_fights))).fillna(0)
            
        else:
            np.random.seed(42)
            data_size = 5000
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
        
        model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4)
        model.fit(X, y)
        try:
            model.save_model(model_file)
        except Exception:
            pass
        
    return model

# --- SAFE EXECUTION WRAPPER ---
try:
    model = load_or_train_model()

    # --- 3. DASHBOARD UI ---
    st.sidebar.header("🔑 API Keys")
    st.sidebar.markdown("Twitter keys are required for live camp sentiment. YouTube does not require keys.")
    api_key = st.sidebar.text_input("X API Key", type="password")
    api_secret = st.sidebar.text_input("X API Secret", type="password")
    access_token = st.sidebar.text_input("X Access Token", type="password")
    access_secret = st.sidebar.text_input("X Access Secret", type="password")

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
        fetch_live_data = st.checkbox("Fetch Live Web & YouTube Stats", value=True)

    if st.button("Calculate Prediction & Expected Value", use_container_width=True):
        sentiment_diff = 0.0
        
        if fetch_live_data:
            st.markdown("---")
            st.subheader("📡 Live Intelligence Feed")
            
            with st.spinner("Scraping Tapology, Twitter, and YouTube..."):
                a_record = get_tapology_record(fighter_a_name)
                b_record = get_tapology_record(fighter_b_name)
                
                if api_key:
                    a_sentiment = get_twitter_sentiment(fighter_a_name, api_key, api_secret, access_token, access_secret)
                    b_sentiment = get_twitter_sentiment(fighter_b_name, api_key, api_secret, access_token, access_secret)
                    sentiment_diff = a_sentiment - b_sentiment
                    st.info(f"**{fighter_a_name}** Tapology: {a_record} | Twitter Sentiment: {a_sentiment:.2f}")
                    st.info(f"**{fighter_b_name}** Tapology: {b_record} | Twitter Sentiment: {b_sentiment:.2f}")
                else:
                    st.warning("⚠️ No Twitter API keys found. Skipping X sentiment analysis.")
                    st.info(f"**{fighter_a_name}** Tapology: {a_record}")
                    st.info(f"**{fighter_b_name}** Tapology: {b_record}")

                yt_data = get_youtube_predictions(fighter_a_name, fighter_b_name)
                
                st.markdown("#### 📺 YouTube Analyst Transcripts (Closing Thoughts)")
                if yt_data and "error" not in yt_data[0]:
                    for video in yt_data:
                        with st.expander(f"🎥 {video['channel']}: {video['title']}"):
                            st.markdown(f"**Extracted Quote:** _{video['quote']}_")
                            st.markdown(f"[Watch Full Video]({video['link']})")
                else:
                    st.error("Could not fetch YouTube predictions.")
        
        input_data = pd.DataFrame({
            'reach_diff': [reach_diff],
            'age_diff': [age_diff],
            'sig_strike_diff': [sig_strike_diff],
            'td_diff': [td_diff],
            'win_streak_diff': [win_streak_diff],
            'sentiment_diff': [sentiment_diff]
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