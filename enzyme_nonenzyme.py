# ============================================================
#  ISPLESTAS ENZYME KLASIFIKATORIUS SU NCBI DUOMENIMIS
#  + Data Augmentation
#  + Multi-Modal Features (BIOCHEMINES SAVYBES - tik 5)
#  + Ensemble Methods
#  + Feature Importance
# ============================================================

import os, re, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, callbacks, optimizers, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    confusion_matrix, precision_recall_curve,
    ConfusionMatrixDisplay
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from collections import Counter
import json
from datetime import datetime

# ------------------------------------------------------------
# BIOPYTHON – BIOCHEMINIŲ SAVYBIŲ SKAIČIAVIMUI
# ------------------------------------------------------------
try:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    BIOPYTHON_AVAILABLE = True
except ImportError:
    print("WARNING: Biopython nerastas. Biochemines savybes (išskyrus length) bus 0.0.")
    BIOPYTHON_AVAILABLE = False

# ------------------------------------------------------------
# ATSITIKTINIŲ SKAIČIŲ SĖKLOS (REPRODUKUOJAMUMUI)
# ------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------------------------------------------------
# PAGRINDINIAI MODELIO PARAMETRAI
# ------------------------------------------------------------
MAX_LEN = 512
EMB_DIM = 64
BATCH = 64
EPOCHS = 20
LR = 1e-3

# ------------------------------------------------------------
# FAILŲ IR REZULTATŲ KELIAI
# ------------------------------------------------------------
SNAP_CSV = "ncbi_enzyme_dataset.csv"
RESULTS_DIR = "results_enhanced"

Path(RESULTS_DIR).mkdir(exist_ok=True)
for subdir in ['images', 'models', 'reports', 'augmented_data']:
    Path(f"{RESULTS_DIR}/{subdir}").mkdir(exist_ok=True)

print("="*70)
print("ISPLESTAS ENZYME KLASIFIKATORIUS")
print("Su NCBI Duomenimis + Patobulinimai")
print("="*70)
print("\nNaujos funkcijos:")
print("  1. Data Augmentation - 2x daugiau duomenų")
print("  2. Biocheminės savybės - 5 papildomi požymiai")
print("  3. Multi-Modal Model - sekos + savybės")
print("  4. Ensemble - CNN+BiLSTM + Random Forest")
print("  5. Feature Importance - kas svarbiausia?")
print("  6. Train/Val/Test padalinimas (70/15/15)")
print("="*70)

# ============================================================
# 1) SEKŲ KODAVIMAS
# ============================================================
AA = list("ACDEFGHIKLMNPQRSTVWY")
AA2IDX = {a: i+1 for i, a in enumerate(AA)}
AA2IDX["X"] = len(AA2IDX) + 1
VOCAB_SIZE = len(AA2IDX) + 1

def clean_aa(seq: str) -> str:
    s = re.sub(r'[^A-Za-z]', '', str(seq)).upper()
    s = s.replace("U","C").replace("O","K").replace("B","D") \
         .replace("Z","E").replace("J","I")
    return "".join(ch if ch in AA2IDX else "X" for ch in s)

def encode_seq(seq: str, max_len: int = MAX_LEN) -> np.ndarray:
    ids = [AA2IDX.get(ch, AA2IDX["X"]) for ch in str(seq)[:max_len]]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return np.array(ids, dtype=np.int32)

# ============================================================
# 2) DATA AUGMENTATION
# ============================================================
print("\n" + "="*70)
print("DATA AUGMENTATION - Duomenu Isplėtimas")
print("="*70)

def mutate_sequence(seq: str, mutation_rate: float = 0.02) -> str:
    if len(seq) < 5:
        return seq
    seq_list = list(seq)
    n_mutations = max(1, int(len(seq) * mutation_rate))
    positions = random.sample(range(len(seq)), min(n_mutations, len(seq)))
    for pos in positions:
        seq_list[pos] = random.choice(AA)
    return ''.join(seq_list)

def insert_residues(seq: str, max_insertions: int = 3) -> str:
    if len(seq) > MAX_LEN - 10:
        return seq
    seq_list = list(seq)
    n_insertions = random.randint(1, max_insertions)
    for _ in range(n_insertions):
        pos = random.randint(0, len(seq_list))
        aa = random.choice(AA)
        seq_list.insert(pos, aa)
    return ''.join(seq_list)

def delete_residues(seq: str, max_deletions: int = 3) -> str:
    if len(seq) < 20:
        return seq
    seq_list = list(seq)
    n_deletions = random.randint(1, min(max_deletions, len(seq_list) - 10))
    for _ in range(n_deletions):
        if len(seq_list) > 10:
            pos = random.randint(0, len(seq_list) - 1)
            del seq_list[pos]
    return ''.join(seq_list)

def shuffle_segment(seq: str, segment_length: int = 10) -> str:
    if len(seq) < segment_length * 2:
        return seq
    seq_list = list(seq)
    start = random.randint(0, len(seq) - segment_length)
    segment = seq_list[start:start+segment_length]
    random.shuffle(segment)
    seq_list[start:start+segment_length] = segment
    return ''.join(seq_list)

def augment_sequence(seq: str, n_augmentations: int = 1) -> list:
    augmented = []
    for _ in range(n_augmentations):
        aug_type = random.choice(['mutate', 'insert', 'delete', 'shuffle', 'combo'])
        if aug_type == 'mutate':
            new_seq = mutate_sequence(seq, mutation_rate=random.uniform(0.01, 0.05))
        elif aug_type == 'insert':
            new_seq = insert_residues(seq, max_insertions=random.randint(1, 4))
        elif aug_type == 'delete':
            new_seq = delete_residues(seq, max_deletions=random.randint(1, 4))
        elif aug_type == 'shuffle':
            new_seq = shuffle_segment(seq, segment_length=random.randint(5, 12))
        else:
            new_seq = mutate_sequence(seq, mutation_rate=0.02)
            if random.random() > 0.5:
                new_seq = shuffle_segment(new_seq, segment_length=8)
        augmented.append(new_seq)
    return augmented

print("OK Augmentation funkcijos sukurtos.")

# ============================================================
# 3) BIOCHEMINIU SAVYBIU SKAICIAVIMAS (TIK 5 SAVYBES)
# ============================================================
print("\n" + "="*70)
print("BIOCHEMINIU SAVYBIU SKAICIAVIMAS (TIK 5 SAVYBES)")
print("="*70)

SELECTED_BIO_FEATURES = [
    "length",
    "molecular_weight",
    "isoelectric_point",
    "gravy",
    "aromaticity",
]

def calculate_biochemical_features(seq: str) -> dict:
    """
    Gražina TIK 5 savybes:
    length, molecular_weight, isoelectric_point, gravy, aromaticity
    """
    # Be Biopython – paliekam tik length, kitus nuliais, kad sistema nesugriūtų
    if not BIOPYTHON_AVAILABLE:
        return {
            "length": float(len(seq)),
            "molecular_weight": 0.0,
            "isoelectric_point": 0.0,
            "gravy": 0.0,
            "aromaticity": 0.0,
        }

    try:
        clean_seq = ''.join([aa for aa in seq if aa in AA])
        if len(clean_seq) < 5:
            return None

        analyzer = ProteinAnalysis(clean_seq)

        features = {
            "length": float(len(clean_seq)),
            "molecular_weight": float(analyzer.molecular_weight()),
            "isoelectric_point": float(analyzer.isoelectric_point()),
            "gravy": float(analyzer.gravy()),
            "aromaticity": float(analyzer.aromaticity()),
        }

        # garantuojam, kad grąžinam tik pasirinktus ir ta pačia tvarka
        return {k: float(features.get(k, 0.0)) for k in SELECTED_BIO_FEATURES}

    except Exception:
        return None

if BIOPYTHON_AVAILABLE:
    print("OK Naudojamos 5 biocheminės savybės (su Biopython):")
    print("  1) length")
    print("  2) molecular_weight")
    print("  3) isoelectric_point (pI)")
    print("  4) gravy (angl. GRAVY)")
    print("  5) aromaticity")
else:
    print("ĮSPĖJIMAS: Biopython nėra – bus tik length, kiti 0.0 (įdiek: pip install biopython)")

# ============================================================
# 4) DUOMENU UZKROVIMAS
# ============================================================
print("\n" + "="*70)
print("NCBI DUOMENŲ UŽKROVIMAS")
print("="*70)

assert os.path.exists(SNAP_CSV), f"ERROR: Nerastas {SNAP_CSV}. Įkelk CSV faila!"
df_original = pd.read_csv(SNAP_CSV)
assert {"sequence","label"}.issubset(df_original.columns), "CSV turi 'sequence' ir 'label'"

print(f"NCBI duomenys užkrauti: {len(df_original)} sekų")
print(f"  Klasių balansas: {Counter(df_original['label'])}")
print(f"  Sekų ilgiai: min={df_original['sequence'].str.len().min()}, "
      f"max={df_original['sequence'].str.len().max()}, "
      f"mean={df_original['sequence'].str.len().mean():.0f}")

# ============================================================
# 5) DATA AUGMENTATION TAIKYMAS
# ============================================================
print("\n" + "="*70)
print("DATA AUGMENTATION - Kuriami papildomi duomenys")
print("="*70)

augmented_data = []
for idx, row in df_original.iterrows():
    augmented_data.append({
        'sequence': row['sequence'],
        'label': row['label'],
        'augmented': False,
        'original_idx': idx
    })
    aug_seqs = augment_sequence(row['sequence'], n_augmentations=1)
    for aug_seq in aug_seqs:
        augmented_data.append({
            'sequence': aug_seq,
            'label': row['label'],
            'augmented': True,
            'original_idx': idx
        })

df_augmented = pd.DataFrame(augmented_data)

print(f"\nPo augmentation:")
print(f"  Originalus (NCBI): {(~df_augmented['augmented']).sum()}")
print(f"  Augmented: {df_augmented['augmented'].sum()}")
print(f"  VISO: {len(df_augmented)} sekų")
print(f"  Klasių balansas: {Counter(df_augmented['label'])}")

df_augmented.to_csv(f"{RESULTS_DIR}/augmented_data/augmented_dataset.csv", index=False)
print(f"OK Išsaugota: augmented_dataset.csv")

# ============================================================
# 6) BIOCHEMINIU SAVYBIU SKAICIAVIMAS VISIEMS DUOMENIMS
# ============================================================
print("\n" + "="*70)
print("BIOCHEMINIŲ SAVYBIŲ SKAIČIAVIMAS (TIK 5)")
print("="*70)
print("INFO: Skaičiuojama... (gali užtrukti kelias minutes)")

bio_features_list = []
valid_indices = []

for idx, row in df_augmented.iterrows():
    seq = clean_aa(row['sequence'])
    features = calculate_biochemical_features(seq)
    if features is not None:
        bio_features_list.append(features)
        valid_indices.append(idx)

    if (idx + 1) % 500 == 0:
        print(f"  Apdorota: {idx + 1}/{len(df_augmented)}")

df_augmented = df_augmented.iloc[valid_indices].reset_index(drop=True)
bio_features_df = pd.DataFrame(bio_features_list, columns=SELECTED_BIO_FEATURES)

print(f"\nBiocheminės savybės apskaičiuotos: {len(bio_features_df)} sekų")
print(f"  Požymių skaičius: {bio_features_df.shape[1]}")
print(f"  Požymiai: {list(bio_features_df.columns)}")

scaler = StandardScaler()
bio_features_scaled = scaler.fit_transform(bio_features_df)
print("Savybės normalizuotos (mean=0, std=1)")

# ============================================================
# 7) DUOMENŲ PADALINIMAS
# ============================================================
print("\n" + "="*70)
print("DUOMENŲ PADALINIMAS")
print("="*70)

idx_all = np.arange(len(df_augmented))

idx_train, idx_tmp = train_test_split(
    idx_all, test_size=0.30,
    stratify=df_augmented["label"],
    random_state=SEED
)
idx_val, idx_test = train_test_split(
    idx_tmp, test_size=0.50,
    stratify=df_augmented["label"].iloc[idx_tmp],
    random_state=SEED
)

X_seq_all = np.vstack([encode_seq(clean_aa(s)) for s in df_augmented["sequence"]])
X_bio_all = bio_features_scaled
y_all = df_augmented["label"].astype(int).to_numpy()

X_seq_train, y_train = X_seq_all[idx_train], y_all[idx_train]
X_seq_val, y_val     = X_seq_all[idx_val],   y_all[idx_val]
X_seq_test, y_test   = X_seq_all[idx_test],  y_all[idx_test]

X_bio_train = X_bio_all[idx_train]
X_bio_val   = X_bio_all[idx_val]
X_bio_test  = X_bio_all[idx_test]

print("Padalinimas (70/15/15):")
print(f"  Train: {len(y_train)} ({len(y_train)/len(df_augmented)*100:.1f}%)")
print(f"  Val:   {len(y_val)} ({len(y_val)/len(df_augmented)*100:.1f}%)")
print(f"  Test:  {len(y_test)} ({len(y_test)/len(df_augmented)*100:.1f}%)")
print(f"\n  Sekų forma: {X_seq_train.shape}")
print(f"  Bio forma:  {X_bio_train.shape}")

classes = np.unique(y_train)
cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weights = {int(c): float(w) for c, w in zip(classes, cw)}
print(f"\nClass weights: {class_weights}")

# ============================================================
# 8) MULTI-MODAL MODELIS
# ============================================================
print("\n" + "="*70)
print("MULTI-MODAL MODELIS")
print("="*70)

def build_multimodal_model(vocab_size=VOCAB_SIZE, max_len=MAX_LEN,
                           emb_dim=EMB_DIM, n_bio_features=5, lr=LR):
    seq_input = layers.Input(shape=(max_len,), dtype="int32", name="sequence_input")
    x1 = layers.Embedding(input_dim=vocab_size, output_dim=emb_dim)(seq_input)
    x1 = layers.Conv1D(128, kernel_size=9, padding="same", activation="relu")(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.MaxPooling1D(pool_size=2)(x1)
    x1 = layers.Bidirectional(layers.LSTM(96, return_sequences=False))(x1)
    x1 = layers.Dropout(0.5)(x1)
    x1 = layers.Dense(128, activation="relu")(x1)
    x1 = layers.Dropout(0.3)(x1)

    bio_input = layers.Input(shape=(n_bio_features,), name="bio_input")
    x2 = layers.Dense(64, activation="relu")(bio_input)
    x2 = layers.Dropout(0.3)(x2)
    x2 = layers.Dense(32, activation="relu")(x2)
    x2 = layers.Dropout(0.2)(x2)

    merged = layers.concatenate([x1, x2])
    x = layers.Dense(128, activation="relu")(merged)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=[seq_input, bio_input], outputs=out, name="MultiModal_Enzyme")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model

model_mm = build_multimodal_model(n_bio_features=X_bio_train.shape[1])
print("\nOK Multi-modal modelis sukurtas!")
print(f"  Parametrų: {model_mm.count_params():,}")
print(f"  2 įvestys: Sekos ({MAX_LEN}) + Bio savybės ({X_bio_train.shape[1]})")

# ============================================================
# 9) TRENIRAVIMAS
# ============================================================
print("\n" + "="*70)
print("MODELIO TRENIRAVIMAS")
print("="*70)

cb = [
    callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5,
                            restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                               min_lr=1e-6, verbose=1),
    callbacks.ModelCheckpoint(f"{RESULTS_DIR}/models/best_multimodal.keras",
                              monitor="val_auc", mode="max",
                              save_best_only=True, verbose=0)
]

history = model_mm.fit(
    [X_seq_train, X_bio_train], y_train,
    validation_data=([X_seq_val, X_bio_val], y_val),
    epochs=EPOCHS,
    batch_size=BATCH,
    class_weight=class_weights,
    verbose=2,
    callbacks=cb
)
print("\nTreniravimas baigtas!")

# ============================================================
# 10) ĮVERTINIMAS
# ============================================================
print("\n" + "="*70)
print("TESTAVIMO REZULTATAI")
print("="*70)

probs_mm = model_mm.predict([X_seq_test, X_bio_test], verbose=0).ravel()

val_probs = model_mm.predict([X_seq_val, X_bio_val], verbose=0).ravel()
p, r, th = precision_recall_curve(y_val, val_probs)
f1 = 2 * p * r / (p + r + 1e-9)
best_tau = float(th[np.argmax(f1[:-1])]) if len(th) > 0 else 0.5

preds_mm = (probs_mm >= best_tau).astype(int)
acc_mm = (preds_mm == y_test).mean()
auc_mm = roc_auc_score(y_test, probs_mm)

print("\nMulti-Modal Rezultatai:")
print(f"  Slenkstis: {best_tau:.3f}")
print(f"  Accuracy: {acc_mm:.4f} ({acc_mm*100:.2f}%)")
print(f"  ROC-AUC: {auc_mm:.4f}")

print("\n" + "="*40)
print(classification_report(
    y_test, preds_mm,
    digits=4,
    target_names=['Non-Enzyme', 'Enzyme']
))

cm_mm = confusion_matrix(y_test, preds_mm)
print("\nKlaidų matrica:")
print(cm_mm)

# ============================================================
# 11) ENSEMBLE
# ============================================================
print("\n" + "="*70)
print("ENSEMBLE MODELIS")
print("="*70)

train_preds_mm = model_mm.predict([X_seq_train, X_bio_train], verbose=0).ravel()
test_preds_mm = probs_mm

X_ensemble_train = np.column_stack([X_bio_train, train_preds_mm])
X_ensemble_test = np.column_stack([X_bio_test, test_preds_mm])

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=SEED,
    n_jobs=-1
)
rf.fit(X_ensemble_train, y_train)

ensemble_probs = rf.predict_proba(X_ensemble_test)[:, 1]
ensemble_preds = (ensemble_probs >= 0.5).astype(int)

acc_ens = (ensemble_preds == y_test).mean()
auc_ens = roc_auc_score(y_test, ensemble_probs)

print("\nEnsemble Rezultatai:")
print(f"  Accuracy: {acc_ens:.4f} ({acc_ens*100:.2f}%)")
print(f"  ROC-AUC: {auc_ens:.4f}")

# ============================================================
# 12) FEATURE IMPORTANCE (tik 5)
# ============================================================
print("\n" + "="*70)
print("BIOCHEMINIŲ SAVYBIŲ SVARBA (TIK 5)")
print("="*70)

rf_bio = RandomForestClassifier(
    n_estimators=100,
    random_state=SEED,
    n_jobs=-1
)
rf_bio.fit(X_bio_train, y_train)

feature_importance = pd.DataFrame({
    'feature': bio_features_df.columns,
    'importance': rf_bio.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTOP biocheminės savybės:")
for _, row in feature_importance.iterrows():
    print(f"  {row['feature']:20s}: {row['importance']:.4f}")

# ============================================================
# 13) VIZUALIZACIJOS (plt.show pašalinta)
# ============================================================
print("\n" + "="*70)
print("VIZUALIZACIJŲ GENERAVIMAS")
print("="*70)

plt.style.use('seaborn-v0_8-whitegrid')

# 1) Mokymo eiga
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Multi-Modal Modelio Mokymas', fontsize=16, fontweight='bold')

axes[0].plot(history.history['loss'], label='Train', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Val', linewidth=2)
axes[0].set_title('Nuostolis (Loss)', fontweight='bold')
axes[0].set_xlabel('Epocha'); axes[0].legend()

axes[1].plot(history.history['accuracy'], label='Train', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Val', linewidth=2)
axes[1].set_title('Tikslumas (Accuracy)', fontweight='bold')
axes[1].set_xlabel('Epocha'); axes[1].legend()

axes[2].plot(history.history['auc'], label='Train', linewidth=2)
axes[2].plot(history.history['val_auc'], label='Val', linewidth=2)
axes[2].set_title('AUC', fontweight='bold')
axes[2].set_xlabel('Epocha'); axes[2].legend()

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/images/01_training.png', dpi=300)
plt.close()

# 2) ROC kreivės
fig, ax = plt.subplots(figsize=(10, 8))
fpr_mm, tpr_mm, _ = roc_curve(y_test, probs_mm)
fpr_ens, tpr_ens, _ = roc_curve(y_test, ensemble_probs)

ax.plot(fpr_mm, tpr_mm, label='Multi-Modal', linewidth=2.5)
ax.plot(fpr_ens, tpr_ens, label='Ensemble', linewidth=2.5)
ax.plot([0, 1], [0, 1], 'k--', label='Atsitiktinis')

ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Kreivių Palyginimas')
ax.legend()

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/images/02_roc.png', dpi=300)
plt.close()

# 3) Feature importance (tik 5)
fig, ax = plt.subplots(figsize=(10, 5))
top_features = feature_importance.head(len(SELECTED_BIO_FEATURES))
ax.barh(top_features['feature'], top_features['importance'])
ax.set_xlabel('Svarba (Importance)')
ax.set_title('Biocheminių Savybių Svarba (5 požymiai)')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/images/03_feature_importance.png', dpi=300)
plt.close()

# 4) Confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Klaidu Matricos', fontsize=16)

ConfusionMatrixDisplay(cm_mm, display_labels=['Non-Enzyme', 'Enzyme']).plot(ax=axes[0])
ConfusionMatrixDisplay(
    confusion_matrix(y_test, ensemble_preds),
    display_labels=['Non-Enzyme', 'Enzyme']
).plot(ax=axes[1])

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/images/04_confusion.png', dpi=300)
plt.close()

# 5) Precision–Recall
fig, ax = plt.subplots(figsize=(10, 8))
prec_mm, rec_mm, _ = precision_recall_curve(y_test, probs_mm)
prec_ens, rec_ens, _ = precision_recall_curve(y_test, ensemble_probs)

ax.plot(rec_mm, prec_mm, label='Multi-Modal', linewidth=2.5)
ax.plot(rec_ens, prec_ens, label='Ensemble', linewidth=2.5)

ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision–Recall Kreivės')
ax.legend()

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/images/05_pr_curves.png', dpi=300)
plt.close()

# 6) Modelių palyginimas
fig, ax = plt.subplots(figsize=(10, 6))
models_ = ['Multi-Modal', 'Ensemble']
accuracies = [acc_mm, acc_ens]
aucs = [auc_mm, auc_ens]

x = np.arange(len(models_))
width = 0.35

ax.bar(x - width/2, accuracies, width, label='Accuracy')
ax.bar(x + width/2, aucs, width, label='ROC-AUC')

ax.set_xticks(x)
ax.set_xticklabels(models_)
ax.set_ylabel('Reikšmė')
ax.set_title('Modelių Palyginimas')
ax.legend()

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/images/06_comparison.png', dpi=300)
plt.close()

print("\nVisi grafikai sugeneruoti!")

# ============================================================
# 14) EKSPORTAS
# ============================================================
print("\n" + "="*70)
print("REZULTATŲ EKSPORTAVIMAS")
print("="*70)

df_results = df_augmented.iloc[idx_test].reset_index(drop=True)
df_results['multimodal_prob'] = probs_mm
df_results['multimodal_pred'] = preds_mm
df_results['ensemble_prob'] = ensemble_probs
df_results['ensemble_pred'] = ensemble_preds
df_results['correct_mm'] = (preds_mm == y_test)
df_results['correct_ens'] = (ensemble_preds == y_test)
df_results['true_label'] = y_test

for col in bio_features_df.columns:
    df_results[f'bio_{col}'] = bio_features_df.iloc[idx_test][col].values

df_results.to_csv(f'{RESULTS_DIR}/reports/all_predictions.csv', index=False)
print(f"Išsaugota: all_predictions.csv ({len(df_results)} įrašų)")

errors_mm = df_results[~df_results['correct_mm']].copy()
errors_ens = df_results[~df_results['correct_ens']].copy()
errors_mm.to_csv(f'{RESULTS_DIR}/reports/errors_multimodal.csv', index=False)
errors_ens.to_csv(f'{RESULTS_DIR}/reports/errors_ensemble.csv', index=False)
print(f"Išsaugotos klaidos: {len(errors_mm)} (MM), {len(errors_ens)} (Ensemble)")

feature_importance.to_csv(f'{RESULTS_DIR}/reports/feature_importance.csv', index=False)
print("Išsaugota: feature_importance.csv")

report = {
    'metadata': {
        'timestamp': datetime.now().isoformat(),
        'original_data': SNAP_CSV,
        'original_samples': int(len(df_original)),
        'augmented_samples': int(len(df_augmented)),
        'augmentation_ratio': float(len(df_augmented) / max(len(df_original), 1)),
        'seed': SEED
    },
    'data': {
        'train_size': int(len(y_train)),
        'val_size': int(len(y_val)),
        'test_size': int(len(y_test)),
        'n_bio_features': int(X_bio_train.shape[1]),
        'bio_features': list(bio_features_df.columns)
    },
    'multimodal_results': {
        'threshold': float(best_tau),
        'accuracy': float(acc_mm),
        'roc_auc': float(auc_mm),
        'errors': int((preds_mm != y_test).sum())
    },
    'ensemble_results': {
        'accuracy': float(acc_ens),
        'roc_auc': float(auc_ens),
        'errors': int((ensemble_preds != y_test).sum())
    },
    'improvement': {
        'accuracy_gain': float(acc_ens - acc_mm),
        'auc_gain': float(auc_ens - auc_mm)
    },
    'top_features': feature_importance.to_dict('records')
}

with open(f'{RESULTS_DIR}/reports/final_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print("Išsaugota: final_report.json")

# ============================================================
# 15) GALUTINĖ SANTRAUKA
# ============================================================
print("\n" + "="*70)
print("GALUTINĖ SANTRAUKA")
print("="*70)

print("\nDuomenys:")
print(f"   NCBI originalūs: {len(df_original)}")
print(f"   Po augmentation: {len(df_augmented)} (x{len(df_augmented)/len(df_original):.1f})")
print(f"   Biocheminių savybių: {X_bio_train.shape[1]} ({list(bio_features_df.columns)})")

print("\nMulti-Modal Modelis:")
print(f"   Accuracy: {acc_mm:.4f} ({acc_mm*100:.2f}%)")
print(f"   ROC-AUC:  {auc_mm:.4f}")
print(f"   Klaidos:  {(preds_mm != y_test).sum()}")

print("\nEnsemble Modelis:")
print(f"   Accuracy: {acc_ens:.4f} ({acc_ens*100:.2f}%)")
print(f"   ROC-AUC:  {auc_ens:.4f}")
print(f"   Klaidos:  {(ensemble_preds != y_test).sum()}")

print("\nPatobulinimas:")
print(f"   Accuracy: +{(acc_ens - acc_mm)*100:.2f}%")
print(f"   ROC-AUC:  +{(auc_ens - auc_mm):.4f}")

print("\nTOP biocheminės savybės:")
for i, (_, row) in enumerate(feature_importance.iterrows(), start=1):
    print(f"   {i}. {row['feature']:20s} ({row['importance']:.4f})")

print("\nIšsaugoti failai:")
print(f"   {RESULTS_DIR}/models/best_multimodal.keras")
print(f"   {RESULTS_DIR}/images/ (6 grafikai)")
print(f"   {RESULTS_DIR}/reports/ (CSV + JSON)")

print("\n" + "="*70)
print("PROJEKTAS BAIGTAS!")
print("="*70)
