import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch

SAMPLING_RATE = 250 
DURATION = 5 

def clean_signal(raw_data, fs):
    nyquist = 0.5 * fs
    low = 8.0 / nyquist
    high = 30.0 / nyquist
    b, a = butter(4, [low, high], btype='band')
    clean_data = filtfilt(b, a, raw_data)
    return clean_data

def extract_brain_hash(clean_data, fs):
    freqs, psd = welch(clean_data, fs, nperseg=fs*2)
    alpha_idx = np.logical_and(freqs >= 8, freqs <= 13)
    beta_idx = np.logical_and(freqs >= 14, freqs <= 30)
    
    alpha_power = np.sum(psd[alpha_idx])
    beta_power = np.sum(psd[beta_idx])
    total_power = np.sum(psd)
    
    rel_alpha = alpha_power / total_power
    rel_beta = beta_power / total_power
    return [rel_alpha, rel_beta]

def train_lock():
    print("Initializing Biometric Lock...")
    print("Enrolling Owner...")
    
    try:
        df = pd.read_csv('owner_data/owner_sample.csv')
        col_name = df.columns[0] 
        real_wave = df[col_name].values
        
        chunk_size = SAMPLING_RATE * DURATION
        owner_hashes = []
        
        for i in range(10):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            if end_idx <= len(real_wave):
                chunk = real_wave[start_idx:end_idx]
                clean_wave = clean_signal(chunk, SAMPLING_RATE)
                owner_hashes.append(extract_brain_hash(clean_wave, SAMPLING_RATE))
        
        if len(owner_hashes) == 0:
            print("🔴 ERROR: Not enough data.")
            return None
            
        avg_alpha = np.mean([h[0] for h in owner_hashes])
        avg_beta = np.mean([h[1] for h in owner_hashes])
        
        print(f"📊 OWNER'S BASELINE -> Alpha: {avg_alpha:.4f} | Beta: {avg_beta:.4f}")
        print("🟢 Lock successfully trained on Owner's neural signature.\n")
        
        return [avg_alpha, avg_beta]
        
    except FileNotFoundError:
        print("🔴 ERROR: Could not find 'owner_data/owner_sample.csv'.")
        return None

def attempt_login(baseline_key, file_path):
    print(f"--- NEW LOGIN ATTEMPT: {file_path} ---")
    
    try:
        df = pd.read_csv(file_path)
        col_name = df.columns[0]
        attempt_wave = df[col_name].values
        
        chunk_size = SAMPLING_RATE * DURATION
        if len(attempt_wave) < chunk_size:
            print("🔴 ERROR: File too short.\n")
            return
            
        chunk = attempt_wave[0 : chunk_size]
        clean_attempt = clean_signal(chunk, SAMPLING_RATE)
        attempt_hash = extract_brain_hash(clean_attempt, SAMPLING_RATE)
        
        print(f"🔍 ATTEMPT HASH -> Alpha: {attempt_hash[0]:.4f} | Beta: {attempt_hash[1]:.4f}")
        
        alpha_diff = abs(attempt_hash[0] - baseline_key[0])
        beta_diff = abs(attempt_hash[1] - baseline_key[1])
        total_difference = alpha_diff + beta_diff
        
        print(f"📐 DIFFERENCE SCORE -> {total_difference:.4f} (Must be < 0.05 to unlock)")
        
        if total_difference < 0.05:
            print("🟢 STATUS: ACCESS GRANTED. Welcome back.\n")
        else:
            print("🔴 STATUS: ACCESS DENIED. Unrecognized Biometric Signature.\n")
            
    except FileNotFoundError:
        print(f"🔴 ERROR: Could not find '{file_path}'.\n")

if __name__ == "__main__":
    my_lock = train_lock()
    
    if my_lock is not None:
        print("\n=== TESTING SCENARIO 1: The Owner Returns ===")
        attempt_login(my_lock, 'owner_data/owner_test_sample.csv')
        
        print("\n=== TESTING SCENARIO 2: A Stranger Attacks ===")
        attempt_login(my_lock, 'login_attempts/imposter_sample.csv')