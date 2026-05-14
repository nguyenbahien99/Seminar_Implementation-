import os
import librosa
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings

warnings.filterwarnings('ignore')

print("="*65)
print("     ASVSPOOF 2019 - GMM BASELINE IMPLEMENTATION")
print("="*65)

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
        return mfcc.T
    except Exception as e:
        print(f"Lỗi đọc file {file_path}: {e}")
        return None

def load_data_from_folder(folder_path):
    features_list = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.wav', '.mp3', '.flac')):
            file_path = os.path.join(folder_path, file_name)
            features = extract_features(file_path)
            if features is not None:
                features_list.append(features)
    return np.vstack(features_list) if features_list else np.array([])

print("[1/3] Đang đọc dữ liệu huấn luyện...")
real_train_path = "data/Real_train"
spoof_train_path = "data/spoof_train"

real_train_data = load_data_from_folder(real_train_path)
spoof_train_data = load_data_from_folder(spoof_train_path)

print("[2/3] Đang huấn luyện mô hình GMM Baseline...")
gmm_real = GaussianMixture(n_components=16, covariance_type='diag', random_state=42)
gmm_spoof = GaussianMixture(n_components=16, covariance_type='diag', random_state=42)

gmm_real.fit(real_train_data)
gmm_spoof.fit(spoof_train_data)
print("=> Huấn luyện hoàn tất!\n")

print("[3/3] Đang kiểm tra trên tập Test...\n")

def evaluate_folder(folder_path, true_label):
    y_true = []
    y_pred = []
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.wav', '.mp3', '.flac')):
            file_path = os.path.join(folder_path, file_name)
            features = extract_features(file_path)
            if features is None: continue
            
            score_real = gmm_real.score(features)
            score_spoof = gmm_spoof.score(features)
            
            prediction = "Real" if score_real > score_spoof else "Spoof"
            
            y_true.append(true_label)
            y_pred.append(prediction)
            
            result_text = "ĐÚNG" if prediction == true_label else "SAI"
            print(f"File: {file_name:15} | Dự đoán: {prediction:5} | Kết quả: {result_text}")
            
    return y_true, y_pred

print("--- TEST GIỌNG THẬT (REAL) ---")
y_true_real, y_pred_real = evaluate_folder("data/test/Real_voice", "Real")

print("\n--- TEST GIỌNG GIẢ MẠO (SPOOF) ---")
y_true_spoof, y_pred_spoof = evaluate_folder("data/test/Spoof_voice", "Spoof")

y_true_all = y_true_real + y_true_spoof
y_pred_all = y_pred_real + y_pred_spoof

print("\n" + "="*65)
print("                BÁO CÁO ĐÁNH GIÁ MÔ HÌNH")
print("="*65)

acc = accuracy_score(y_true_all, y_pred_all) * 100
print(f"[*] TỔNG KẾT ĐỘ CHÍNH XÁC (ACCURACY): {acc:.2f}%\n")

labels = ["Real", "Spoof"]
cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)

print("1. MA TRẬN NHẦM LẪN (CONFUSION MATRIX):")
print(f"                 Dự đoán Real    Dự đoán Spoof")
print(f"Thực tế Real  |      {cm[0][0]:<10} |      {cm[0][1]:<10}")
print(f"Thực tế Spoof |      {cm[1][0]:<10} |      {cm[1][1]:<10}")
print("-" * 65)

print("2. CÁC CHỈ SỐ CHI TIẾT (PRECISION, RECALL, F1-SCORE):")
report = classification_report(y_true_all, y_pred_all, target_names=labels)
print(report)
print("="*65)