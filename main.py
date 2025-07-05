import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# --- 1. Membaca Data dari File CSV ---
print("Memuat data dari file CSV...")

csv_files = {
    'item_metadata': 'dataset/item_metadata.csv',
    'transaction_history': 'dataset/transaction_history.csv',
    'market_listings_snapshot': 'dataset/market_listings_snapshot.csv',
    'game_events': 'dataset/game_events.csv'
}

dataframes = {}
for name, filename in csv_files.items():
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' tidak ditemukan di direktori saat ini.")
        print("Pastikan Anda memiliki file CSV yang benar atau sesuaikan path.")
        exit()

try:
    item_metadata_df = pd.read_csv(csv_files['item_metadata'])
    print(f"- '{csv_files['item_metadata']}' berhasil dimuat.")
    
    transaction_history_df = pd.read_csv(csv_files['transaction_history'], parse_dates=['timestamp'])
    print(f"- '{csv_files['transaction_history']}' berhasil dimuat.")
    
    market_listings_snapshot_df = pd.read_csv(csv_files['market_listings_snapshot'], parse_dates=['timestamp'])
    print(f"- '{csv_files['market_listings_snapshot']}' berhasil dimuat.")
    
    game_events_df = pd.read_csv(csv_files['game_events'], parse_dates=['start_date', 'end_date'])
    game_events_df['affected_items'] = game_events_df['affected_items'].apply(eval)
    print(f"- '{csv_files['game_events']}' berhasil dimuat.")

except Exception as e:
    print(f"Terjadi kesalahan saat memuat file CSV: {e}")
    print("Kemungkinan besar ada masalah format data di salah satu file atau 'affected_items' di game_events.csv.")
    exit()

print("Semua DataFrame berhasil dimuat dari file CSV.\n")

# --- 2. Pra-pemrosesan Data dan Rekayasa Fitur ---

# Gabungkan transaction_history dengan item_metadata
df = pd.merge(transaction_history_df, item_metadata_df, on='item_id', how='left')

# Simpan timestamp asli untuk pembagian data time-series nantinya
df['timestamp_orig'] = df['timestamp']

# --- Rekayasa Fitur ---

# Fitur Waktu
df['hour_of_day'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['day_of_month'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month
df['year'] = df['timestamp'].dt.year

# Fitur Rarity (Label Encoding)
rarity_mapping = {'Common': 1, 'Uncommon': 2, 'Rare': 3, 'Epic': 4, 'Legendary': 5}
df['rarity_encoded'] = df['rarity'].map(rarity_mapping)

# Fitur Item Type (One-Hot Encoding)
df = pd.get_dummies(df, columns=['item_type'], prefix='type', drop_first=True)

# Fitur craftable (ubah ke int)
df['craftable_int'] = df['craftable'].astype(int)

print("Menghitung fitur historis (rolling averages & std dev)... Ini mungkin butuh waktu.")
# Fitur Harga Historis (Moving Averages & Volatility)
# Fungsi ini akan menerima df_item yang MASIH memiliki kolom 'timestamp'
def calculate_rolling_features(df_item):
    # Urutkan berdasarkan timestamp untuk rolling window
    df_item = df_item.sort_values(by='timestamp')
    
    # Set 'timestamp' sebagai indeks untuk operasi rolling
    # Ini adalah perbaikan utama!
    df_item_indexed = df_item.set_index('timestamp')

    df_item['price_ma_1h'] = df_item_indexed['price_per_unit'].rolling(window=pd.Timedelta(hours=1), closed='left').mean().values
    df_item['price_ma_6h'] = df_item_indexed['price_per_unit'].rolling(window=pd.Timedelta(hours=6), closed='left').mean().values
    df_item['price_std_1h'] = df_item_indexed['price_per_unit'].rolling(window=pd.Timedelta(hours=1), closed='left').std().values
    df_item['price_std_6h'] = df_item_indexed['price_per_unit'].rolling(window=pd.Timedelta(hours=6), closed='left').std().values
    
    return df_item

# Pastikan `df` yang dilewatkan ke `groupby().apply()` memiliki kolom `timestamp`
df = df.groupby('item_id', group_keys=False).apply(calculate_rolling_features)

# Mengisi NaN yang mungkin muncul karena tidak ada data historis sebelumnya
df['price_ma_1h'] = df['price_ma_1h'].fillna(df['base_value'] * df['rarity_encoded'])
df['price_ma_6h'] = df['price_ma_6h'].fillna(df['base_value'] * df['rarity_encoded'])
df['price_std_1h'] = df['price_std_1h'].fillna(0)
df['price_std_6h'] = df['price_std_6h'].fillna(0)
print("Fitur historis selesai dihitung.\n")


# Fitur Event Game
def is_event_active(row_timestamp, item_id, events_df):
    for idx, event in events_df.iterrows():
        start = event['start_date']
        # Handle cases where end_date might be NaT (Not a Time) or None
        end = event['end_date'] if pd.notna(event['end_date']) else datetime(2099, 12, 31)
        affected_items = event['affected_items']
        
        if start <= row_timestamp <= end and item_id in affected_items:
            return 1
    return 0

print("Menambahkan fitur event game...")
# Gunakan timestamp_orig untuk ini, agar tidak terpengaruh oleh drop kolom 'timestamp' yang belum dilakukan
# Metode apply() ini bisa sangat lambat. Untuk dataset yang sangat besar, pertimbangkan cara lain.
df['is_event_active'] = df.apply(lambda row: is_event_active(row['timestamp_orig'], row['item_id'], game_events_df), axis=1)
print("Fitur event game selesai ditambahkan.\n")


# Hapus kolom yang tidak lagi diperlukan atau akan menyebabkan kebocoran data
# Kolom 'timestamp' (asli) baru di-drop di sini setelah semua perhitungan yang membutuhkannya selesai
df_processed = df.drop(columns=[
    'transaction_id', 'item_name', 'rarity', 'source_method', 'usage_effect',
    'craftable', 'total_price', 'buyer_id', 'seller_id', 'timestamp' # Sekarang aman untuk drop 'timestamp'
])

# Pastikan tidak ada NaN yang tersisa di fitur yang akan digunakan untuk training
# Ini akan menghapus baris di mana base_value atau rarity_encoded mungkin NaN dari merge.
df_processed = df_processed.dropna()

print("DataFrame setelah pra-pemrosesan dan rekayasa fitur (5 baris pertama):")
print(df_processed.head())
print(f"Jumlah baris setelah dropna: {len(df_processed)}\n")

# --- 3. Pembagian Dataset (Training dan Testing) ---
# Gunakan 'timestamp_orig' yang sudah kita simpan untuk pembagian berdasarkan waktu.

df_processed = df_processed.sort_values(by='timestamp_orig').reset_index(drop=True)

split_date_index = int(len(df_processed) * 0.8)
split_date = df_processed['timestamp_orig'].iloc[split_date_index]

train_df = df_processed[df_processed['timestamp_orig'] < split_date]
test_df = df_processed[df_processed['timestamp_orig'] >= split_date]

# Base value tidak seharusnya menjadi fitur input untuk model yang memprediksi harga
# karena base_value adalah nilai statis dan bukan indikator pasar dinamis
X_train = train_df.drop(columns=['price_per_unit', 'item_id', 'timestamp_orig', 'base_value'])
y_train = train_df['price_per_unit']
X_test = test_df.drop(columns=['price_per_unit', 'item_id', 'timestamp_orig', 'base_value'])
y_test = test_df['price_per_unit']

# Lakukan One-Hot Encoding untuk item_id pada X_train dan X_test
X_train_item_id_encoded = pd.get_dummies(train_df['item_id'], prefix='item')
X_test_item_id_encoded = pd.get_dummies(test_df['item_id'], prefix='item')

X_train = pd.concat([X_train, X_train_item_id_encoded], axis=1)
X_test = pd.concat([X_test, X_test_item_id_encoded], axis=1)

# Pastikan kolom X_train dan X_test sama
common_cols = list(set(X_train.columns) & set(X_test.columns))
X_train = X_train[common_cols]
X_test = X_test[common_cols]

missing_cols_in_test = set(X_train.columns) - set(X_test.columns)
for c in missing_cols_in_test:
    X_test[c] = 0

missing_cols_in_train = set(X_test.columns) - set(X_train.columns)
for c in missing_cols_in_train:
    X_train[c] = 0

X_test = X_test[X_train.columns]


print("Bentuk X_train:", X_train.shape)
print("Bentuk y_train:", y_train.shape)
print("Bentuk X_test:", X_test.shape)
print("Bentuk y_test:", y_test.shape)
print(f"Dataset dibagi. Data training hingga tanggal {split_date.strftime('%Y-%m-%d %H:%M:%S')}, data testing setelah itu.\n")

# --- 4. Pilih dan Latih Model AI (Regressor) ---

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

print("Melatih model RandomForestRegressor...")
model.fit(X_train, y_train)
print("Model selesai dilatih.\n")

# Prediksi pada test set
y_pred = model.predict(X_test)

# Evaluasi model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"--- Hasil Evaluasi Model ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2 Score): {r2:.4f}\n")

# --- 5. Visualisasi Hasil dan Pentingnya Fitur ---

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Harga Aktual (per unit)")
plt.ylabel("Harga Prediksi (per unit)")
plt.title("Prediksi Harga vs Harga Aktual")
plt.grid(True)
plt.show()

if not X_train.empty:
    feature_importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print("Top 10 Pentingnya Fitur:")
    print(feature_importances.head(10))

    plt.figure(figsize=(12, 7))
    sns.barplot(x=feature_importances.head(10), y=feature_importances.head(10).index)
    plt.title("Top 10 Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()
else:
    print("Tidak dapat menampilkan Feature Importance karena X_train kosong.")

print("\nProses modeling selesai.")
print("Untuk implementasi nyata, pastikan Anda menggunakan data riil dan metode validasi time-series yang tepat.")

# --- Menyimpan Model ke File ---
model_filename = 'ingame_price_predictor_model.joblib'
joblib.dump(model, model_filename)

print(f"\nModel berhasil disimpan sebagai '{model_filename}'")
print("Anda sekarang dapat memuat ulang model ini kapan saja untuk membuat prediksi.")