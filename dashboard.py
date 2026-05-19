import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch

# --- CORE MATH & LOGIC ---
SAMPLING_RATE = 250 
DURATION = 5 
NUM_CHANNELS = 3 # UPGRADE: We are now scanning 3 brain regions!

def clean_signal(raw_data, fs):
    nyquist = 0.5 * fs
    b, a = butter(4, [8.0 / nyquist, 30.0 / nyquist], btype='band')
    return filtfilt(b, a, raw_data)

def extract_brain_hash(clean_data, fs):
    freqs, psd = welch(clean_data, fs, nperseg=fs*2)
    alpha_power = np.sum(psd[np.logical_and(freqs >= 8, freqs <= 13)])
    beta_power = np.sum(psd[np.logical_and(freqs >= 14, freqs <= 30)])
    total_power = np.sum(psd)
    return [alpha_power / total_power, beta_power / total_power]

def check_liveness(raw_chunk):
    """Checks if the signal is coming from a living human."""
    signal_variance = np.var(raw_chunk)
    
    if signal_variance < 1.0: # Too perfectly flat
        return False, "🔴 LIVENESS ERROR: Flatline detected. Is the headset on a human?"
    elif signal_variance > 5000.0: # Too noisy
        return False, "🔴 LIVENESS ERROR: Extreme muscle/movement noise detected."
    
    return True, "🟢 Liveness Confirmed: Biological signal detected."

# --- UI VISUALIZATION FUNCTIONS ---
def plot_multi_channel_wave(df_chunk, title):
    """Plots 3 channels overlapping to show the network topology"""
    fig, ax = plt.subplots(figsize=(10, 2))
    colors = ['#8A2BE2', '#32CD32', '#FF4500']
    
    for i in range(NUM_CHANNELS):
        # Plot just 1 second to keep the graph readable
        ax.plot(df_chunk.iloc[:250, i], color=colors[i], alpha=0.7, label=f'Channel {i+1}')
        
    ax.set_title(title)
    ax.axis('off')
    ax.legend(loc='upper right')
    st.pyplot(fig)

def plot_pie_chart(alpha, beta):
    """Creates a pie chart of the Average Relative Power across the network"""
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.pie([alpha, beta], labels=['Alpha', 'Beta'], autopct='%1.1f%%', colors=['#FF9999', '#66B2FF'])
    st.pyplot(fig)

# --- STREAMLIT WEB DASHBOARD ---
st.set_page_config(page_title="EEG Biometric Lock", layout="wide")

st.title("🧠 3D Spatial EEG Authentication Dashboard")
st.write(f"Scanning **{NUM_CHANNELS} distinct brain regions** simultaneously for maximum anti-spoofing security.")
st.markdown("---")

col1, col2 = st.columns(2)

# COLUMN 1: System Enrollment (The Baseline)
with col1:
    st.subheader("1. System Enrollment")
    st.info("Upload the Owner's baseline EEG data.")
    
    owner_file = st.file_uploader("Upload owner_sample.csv", type=['csv'], key="owner")
    baseline_key = None
    
    if owner_file is not None:
        df = pd.read_csv(owner_file)
        chunk_size = SAMPLING_RATE * DURATION
        owner_hashes = []
        
        for i in range(10):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            
            if end_idx <= len(df):
                chunk_fingerprint = []
                
                # Extract features for all 3 channels
                for ch in range(NUM_CHANNELS):
                    raw_chunk = df.iloc[start_idx:end_idx, ch].values
                    clean_wave = clean_signal(raw_chunk, SAMPLING_RATE)
                    ch_alpha, ch_beta = extract_brain_hash(clean_wave, SAMPLING_RATE)
                    chunk_fingerprint.extend([ch_alpha, ch_beta])
                    
                owner_hashes.append(chunk_fingerprint)
                
        # Calculate the baseline across all 6 variables (3 channels x 2 bands)
        baseline_key = np.mean(owner_hashes, axis=0)
        
        st.success("🟢 3D Baseline Enrolled Successfully!")
        
        # Calculate average alpha/beta just for the visual pie chart
        avg_alpha_visual = np.mean(baseline_key[0::2])
        avg_beta_visual = np.mean(baseline_key[1::2])
        
        plot_multi_channel_wave(df.iloc[0:chunk_size], "Multi-Channel Baseline Signal")
        plot_pie_chart(avg_alpha_visual, avg_beta_visual)


# COLUMN 2: Login Attempt
with col2:
    st.subheader("2. Live Login Attempt")
    st.warning("Upload a new brainwave to test access.")
    
    attempt_file = st.file_uploader("Upload test_sample.csv or imposter.csv", type=['csv'], key="attempt")
    
    if attempt_file is not None and baseline_key is not None:
        df_attempt = pd.read_csv(attempt_file)
        chunk_size = SAMPLING_RATE * DURATION
        
        # --- PRE-CHECK: LIVENESS DETECTION ---
        # We test the variance of Channel 1 to ensure the user is alive and present
        ch1_raw = df_attempt.iloc[0:chunk_size, 0].values
        is_alive, liveness_msg = check_liveness(ch1_raw)
        
        if not is_alive:
            st.error(liveness_msg) # Stops the login process if it's a fake/dead signal
        else:
            st.success(liveness_msg)
            
            attempt_fingerprint = []
            
            for ch in range(NUM_CHANNELS):
                raw_chunk = df_attempt.iloc[0:chunk_size, ch].values
                clean_attempt = clean_signal(raw_chunk, SAMPLING_RATE)
                ch_alpha, ch_beta = extract_brain_hash(clean_attempt, SAMPLING_RATE)
                attempt_fingerprint.extend([ch_alpha, ch_beta])
            
            # Show visuals
            plot_multi_channel_wave(df_attempt.iloc[0:chunk_size], "Multi-Channel Login Attempt")
            
            # Calculate Math across all 3 channels
            total_difference = sum(abs(a - b) for a, b in zip(attempt_fingerprint, baseline_key))
            threshold = 0.05 * NUM_CHANNELS # Scales threshold to 0.15 for 3 channels
            
            st.metric(label="3D Spatial Difference Score", value=f"{total_difference:.4f}", delta=f"Must be < {threshold:.2f}", delta_color="inverse")
            
            if total_difference < threshold:
                st.success("🟢 ACCESS GRANTED: Spatial Signature Matched. Welcome back.")
                st.balloons()
            else:
                st.error("🔴 ACCESS DENIED: Spatial Signature Mismatch.")

    elif attempt_file is not None and baseline_key is None:
        st.error("Please enroll the Owner's baseline first!")