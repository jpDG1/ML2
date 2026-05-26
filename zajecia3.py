import matplotlib
matplotlib.use('Agg')  # Wyłącza GUI, wymusza zapis do plików PNG
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════
# A) KLASYFIKACJA — Breast Cancer Wisconsin
# ════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  A) KLASYFIKACJA — Breast Cancer Wisconsin")
print("═"*60)

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# ── 1. Dane ──────────────────────────────────────────────────────
data = load_breast_cancer()
X_cls, y_cls = data.data, data.target

# ── 2. Podział: 60% train | 20% val | 20% test ──────────────────
X_train_full, X_test_cls, y_train_full, y_test_cls = train_test_split(
    X_cls, y_cls, test_size=0.20, random_state=222
)
X_train_cls, X_val_cls, y_train_cls, y_val_cls = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42
)

# ── 3. Pipeline (Imputacja + Standaryzacja + Model) ──────────────
#
# Każdy model owinięty w sklearn Pipeline — gwarantuje brak
# data leakage: scaler i imputer fitowane tylko na train.
#
cls_pipelines = {
    "Regresja Logistyczna": Pipeline([
        ('imp', SimpleImputer(strategy='mean')),
        ('scl', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "K-NN": Pipeline([
        ('imp', SimpleImputer(strategy='mean')),
        ('scl', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=7))
    ]),
    "SVM": Pipeline([
        ('imp', SimpleImputer(strategy='mean')),
        ('scl', StandardScaler()),
        ('clf', SVC(kernel='rbf', C=10, probability=True, random_state=42))
    ]),
    "Drzewo Decyzyjne": Pipeline([
        ('imp', SimpleImputer(strategy='mean')),
        ('scl', StandardScaler()),
        ('clf', DecisionTreeClassifier(max_depth=5, random_state=42))
    ]),
    "Las Losowy": Pipeline([
        ('imp', SimpleImputer(strategy='mean')),
        ('scl', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=42))
    ]),
    "Naiwny Bayes": Pipeline([
        ('imp', SimpleImputer(strategy='mean')),
        ('scl', StandardScaler()),
        ('clf', GaussianNB())
    ]),
    "MLP": Pipeline([
        ('imp', SimpleImputer(strategy='mean')),
        ('scl', StandardScaler()),
        ('clf', MLPClassifier(hidden_layer_sizes=(100, 50),
                              max_iter=1000, random_state=42))
    ]),
}

# ── 4. Metoda holdout: ranking na zbiorze walidacyjnym ───────────
print("\n── Metoda Holdout ──────────────────────────────────────")
print(f"{'Model':22} | {'Val Acc':>9}")
print("-" * 36)

best_cls_name, best_cls_val, best_cls_pipe = "", 0, None

for name, pipe in cls_pipelines.items():
    pipe.fit(X_train_cls, y_train_cls)
    val_acc = accuracy_score(y_val_cls, pipe.predict(X_val_cls))
    if val_acc > best_cls_val:
        best_cls_val, best_cls_name, best_cls_pipe = val_acc, name, pipe
    print(f"{name:22} | {val_acc:9.4f}")

test_acc_holdout = best_cls_pipe.score(X_test_cls, y_test_cls)
print("-" * 36)
print(f"Zwycięzca (holdout): {best_cls_name} ({best_cls_val:.4f})")
print(f"Wynik na teście:     {test_acc_holdout:.4f}")

# ── 5. Walidacja krzyżowa (5-fold) — stabilniejsza ocena ────────
print("\n── Walidacja Krzyżowa 5-Fold ───────────────────────────")
print(f"{'Model':22} | {'Średni Acc':>10} | {'Std +/-':>8}")
print("-" * 46)

kf = KFold(n_splits=5, shuffle=True, random_state=222)
X_cv_cls, _, y_cv_cls, _ = train_test_split(
    X_cls, y_cls, test_size=0.15, random_state=42
)

cv_results = {}
for name, pipe in cls_pipelines.items():
    scores = cross_val_score(pipe, X_cv_cls, y_cv_cls, cv=kf, scoring='accuracy')
    cv_results[name] = scores.mean()
    print(f"{name:22} | {scores.mean():10.4f} | {scores.std():8.4f}")

best_cv_name = max(cv_results, key=cv_results.get)
print("-" * 46)
print(f"Zwycięzca (CV):      {best_cv_name} ({cv_results[best_cv_name]:.4f})")

# Finalny model CV trenujemy na całym zbiorze treningowym
best_cv_pipe = cls_pipelines[best_cv_name]
best_cv_pipe.fit(X_cv_cls, y_cv_cls)

# ── 6. Raport klasyfikacji zwycięzcy ─────────────────────────────
print(f"\n── Raport klasyfikacji: {best_cv_name} ─────────────")
print(classification_report(y_test_cls,
                             best_cv_pipe.predict(X_test_cls),
                             target_names=data.target_names))

# ── 7. Wykres porównawczy ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
names = list(cv_results.keys())
scores = list(cv_results.values())
colors = ['#2ecc71' if n == best_cv_name else '#3498db' for n in names]
bars = ax.barh(names, scores, color=colors, edgecolor='white')
ax.set_xlim(0.85, 1.01)
ax.set_xlabel('Średnia Accuracy (5-Fold CV)')
ax.set_title('Porównanie modeli klasyfikacji — Breast Cancer Wisconsin')
for bar, score in zip(bars, scores):
    ax.text(score + 0.001, bar.get_y() + bar.get_height()/2,
            f'{score:.4f}', va='center', fontsize=9)
ax.axvline(x=max(scores), color='green', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('cls_porownanie.png', dpi=100, bbox_inches='tight')
plt.show()
print("Zapisano: cls_porownanie.png")


# ════════════════════════════════════════════════════════════════
# B) REGRESJA — Auto MPG
# ════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  B) REGRESJA — Auto MPG (UCI)")
print("═"*60)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ── 1. Wczytanie danych ──────────────────────────────────────────
url_mpg = ("http://archive.ics.uci.edu/ml/machine-learning-databases"
           "/auto-mpg/auto-mpg.data")
col_names = ['MPG','Cylinders','Displacement','Horsepower',
             'Weight','Acceleration','Model Year','Origin']
try:
    df_mpg = pd.read_csv(url_mpg, names=col_names, na_values='?',
                         comment='\t', sep=' ', skipinitialspace=True)
    df_mpg = df_mpg.dropna()
    print(f"Wczytano dane Auto MPG: {df_mpg.shape}")
except Exception:
    # Fallback — dane syntetyczne jeśli UCI niedostępne
    print("UCI niedostępne — generuję dane syntetyczne.")
    np.random.seed(42)
    n = 392
    df_mpg = pd.DataFrame({
        'MPG':          np.random.normal(23, 7, n).clip(9, 47),
        'Cylinders':    np.random.choice([4,6,8], n, p=[0.5,0.3,0.2]),
        'Displacement': np.random.normal(195, 105, n).clip(68, 455),
        'Horsepower':   np.random.normal(105, 40, n).clip(46, 230),
        'Weight':       np.random.normal(2977, 850, n).clip(1613, 5140),
        'Acceleration': np.random.normal(15.5, 2.8, n).clip(8, 25),
        'Model Year':   np.random.randint(70, 83, n),
        'Origin':       np.random.choice([1,2,3], n),
    })

X_mpg = df_mpg.drop('MPG', axis=1)
y_mpg = df_mpg['MPG']

X_train_mpg, X_test_mpg, y_train_mpg, y_test_mpg = train_test_split(
    X_mpg, y_mpg, test_size=0.2, random_state=222
)

# ── 2. Pipeline modeli regresji ──────────────────────────────────
reg_models = {
    "Regresja Liniowa": Pipeline([
        ('scl', StandardScaler()),
        ('reg', LinearRegression())
    ]),
    "Regresja Wielomianowa (d=2)": Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scl',  StandardScaler()),
        ('reg',  LinearRegression())
    ]),
    "Drzewo Regresyjne": Pipeline([
        ('scl', StandardScaler()),
        ('reg', DecisionTreeRegressor(max_depth=5, random_state=222))
    ]),
    "Las Regresyjny": Pipeline([
        ('scl', StandardScaler()),
        ('reg', RandomForestRegressor(n_estimators=200, random_state=222))
    ]),
    "SVR (RBF)": Pipeline([
        ('scl', StandardScaler()),
        ('reg', SVR(kernel='rbf', C=10, epsilon=0.1))
    ]),
}

# ── 3. Trening i ewaluacja ───────────────────────────────────────
print(f"\n{'Model':26} | {'R²':>7} | {'MAE':>6} | {'RMSE':>6}")
print("-" * 55)

reg_results = {}
all_preds   = {}

for name, pipe in reg_models.items():
    pipe.fit(X_train_mpg, y_train_mpg)
    y_pred = pipe.predict(X_test_mpg)
    all_preds[name] = y_pred

    r2   = r2_score(y_test_mpg, y_pred)
    mae  = mean_absolute_error(y_test_mpg, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_mpg, y_pred))
    reg_results[name] = {'R2': r2, 'MAE': mae, 'RMSE': rmse}

    print(f"{name:26} | {r2:7.4f} | {mae:6.2f} | {rmse:6.2f}")

best_reg = max(reg_results, key=lambda k: reg_results[k]['R2'])
print("-" * 55)
print(f"Najlepszy model (R²): {best_reg}  "
      f"R²={reg_results[best_reg]['R2']:.4f}")

# ── 4. Walidacja krzyżowa regresji ───────────────────────────────
print("\n── Walidacja Krzyżowa 5-Fold (R²) ─────────────────────")
print(f"{'Model':26} | {'Średni R²':>9} | {'Std +/-':>8}")
print("-" * 50)

kf_reg = KFold(n_splits=5, shuffle=True, random_state=222)
for name, pipe in reg_models.items():
    scores = cross_val_score(pipe, X_mpg, y_mpg,
                             cv=kf_reg, scoring='r2')
    print(f"{name:26} | {scores.mean():9.4f} | {scores.std():8.4f}")

# ── 5. Wykresy dopasowania i reszt ───────────────────────────────
fig, axes = plt.subplots(len(reg_models), 2,
                         figsize=(14, 5 * len(reg_models)))

for i, (name, y_pred) in enumerate(all_preds.items()):
    # Dopasowanie
    ax1 = axes[i, 0]
    ax1.scatter(y_test_mpg, y_pred, alpha=0.6,
                color='teal', edgecolors='w', s=40)
    lims = [min(y_test_mpg.min(), y_pred.min()),
            max(y_test_mpg.max(), y_pred.max())]
    ax1.plot(lims, lims, 'r--', alpha=0.7)
    ax1.set_title(f'{name}: Dopasowanie  '
                  f'(R²={reg_results[name]["R2"]:.3f})')
    ax1.set_xlabel('Rzeczywiste MPG')
    ax1.set_ylabel('Przewidziane MPG')

    # Reszty
    ax2 = axes[i, 1]
    residuals = y_test_mpg.values - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6,
                color='indianred', edgecolors='w', s=40)
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title(f'{name}: Wykres reszt')
    ax2.set_xlabel('Przewidziane MPG')
    ax2.set_ylabel('Residuum')

plt.suptitle('Porównanie modeli regresji — Auto MPG', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('reg_wykresy.png', dpi=80, bbox_inches='tight')
plt.show()
print("Zapisano: reg_wykresy.png")


# ════════════════════════════════════════════════════════════════
# C) SYSTEM REKOMENDACJI — MovieLens ml-latest-small
# ════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  C) SYSTEM REKOMENDACJI — MovieLens")
print("═"*60)

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── 1. Pobieranie danych ─────────────────────────────────────────
ML_PATH = 'ml-latest-small'
ZIP     = 'ml-latest-small.zip'
ML_URL  = ('https://files.grouplens.org/datasets'
           '/movielens/ml-latest-small.zip')

if not os.path.exists(ML_PATH):
    print("Pobieram zbiór MovieLens...")
    try:
        import urllib.request
        urllib.request.urlretrieve(ML_URL, ZIP)
        import zipfile
        with zipfile.ZipFile(ZIP, 'r') as z:
            z.extractall('.')
        print("Pobrano i rozpakowano.")
    except Exception as e:
        print(f"Błąd pobierania: {e}")
        print("Uruchom ręcznie:")
        print(f"  Pobierz {ML_URL}")
        print(f"  Rozpakuj do folderu ml-latest-small/")
        # Generujemy minimalny zbiór syntetyczny do demonstracji
        os.makedirs(ML_PATH, exist_ok=True)
        genres_list = [
            "Action|Sci-Fi", "Action|Adventure", "Comedy|Romance",
            "Drama", "Action|Thriller", "Sci-Fi|Thriller",
            "Comedy", "Drama|Romance", "Action|Crime", "Sci-Fi|Action"
        ]
        titles = [
            "Matrix, The (1999)", "Speed (1994)", "When Harry Met Sally (1989)",
            "Shawshank Redemption, The (1994)", "Die Hard (1988)",
            "Terminator 2 (1991)", "Home Alone (1990)", "Notebook, The (2004)",
            "Heat (1995)", "Interstellar (2014)"
        ]
        movies_syn = pd.DataFrame({
            'movieId': range(1, 11),
            'title':   titles,
            'genres':  genres_list
        })
        users  = list(range(1, 51)) * 4
        items  = list(range(1, 11)) * 20
        np.random.seed(42)
        ratings_syn = pd.DataFrame({
            'userId':  users[:200],
            'movieId': items[:200],
            'rating':  np.random.choice([2,3,4,5], 200),
            'timestamp': 0
        }).drop_duplicates(subset=['userId','movieId'])
        movies_syn.to_csv(f'{ML_PATH}/movies.csv', index=False)
        ratings_syn.to_csv(f'{ML_PATH}/ratings.csv', index=False)
        print("Wygenerowano minimalny zbiór syntetyczny.")

movies  = pd.read_csv(f'{ML_PATH}/movies.csv')
ratings = pd.read_csv(f'{ML_PATH}/ratings.csv')
print(f"Filmy: {len(movies):,}   Oceny: {len(ratings):,}")

# ── 2. Filtracja popularnych filmów (min. 50 ocen) ───────────────
df_combined   = pd.merge(ratings, movies, on='movieId')
movie_counts  = df_combined.groupby('title')['rating'].count()
popular       = movie_counts[movie_counts >= 50].index
df_filtered   = df_combined[df_combined['title'].isin(popular)]
movies_filt   = (movies[movies['title'].isin(popular)]
                 .reset_index(drop=True))

print(f"Popularne filmy (≥50 ocen): {len(movies_filt)}")

# ── 3. Algorytm 1: Collaborative Filtering (SVD) ────────────────
def get_svd_recs(movie_title, top_n=5):
    matrix = (df_filtered
              .pivot_table(index='userId', columns='title',
                           values='rating')
              .fillna(0))
    svd = TruncatedSVD(n_components=12, random_state=42)
    reduced   = svd.fit_transform(matrix.values.T)
    corr_mat  = np.corrcoef(reduced)
    titles    = list(matrix.columns)

    if movie_title not in titles:
        return None

    idx    = titles.index(movie_title)
    scores = corr_mat[idx]
    res    = (pd.DataFrame({'Tytuł': titles, 'Score': scores})
              .query("Tytuł != @movie_title")
              .sort_values('Score', ascending=False)
              .head(top_n)
              .reset_index(drop=True))
    res.index += 1
    return res

# ── 4. Algorytm 2: Content-Based (gatunki + TF-IDF) ─────────────
def get_content_recs(movie_title, top_n=5):
    df = movies_filt.copy()
    df['genres_str'] = df['genres'].str.replace('|', ' ', regex=False)
    tfidf      = TfidfVectorizer(stop_words='english')
    tfidf_mat  = tfidf.fit_transform(df['genres_str'])
    cos_sim    = cosine_similarity(tfidf_mat, tfidf_mat)

    if movie_title not in df['title'].values:
        return None

    idx    = df[df['title'] == movie_title].index[0]
    res    = (pd.DataFrame({
                  'Tytuł':   df['title'],
                  'Gatunki': df['genres'],
                  'Score':   cos_sim[idx]})
              .query("Tytuł != @movie_title")
              .sort_values('Score', ascending=False)
              .head(top_n)
              .reset_index(drop=True))
    res.index += 1
    return res

# ── 5. Algorytm 3: Hybryda (SVD + Content, wagi 0.5/0.5) ────────
def get_hybrid_recs(movie_title, w_svd=0.5, w_content=0.5, top_n=5):
    svd_all = get_svd_recs(movie_title, top_n=9999)
    if svd_all is None:
        return None

    # Normalizacja Min-Max wyników SVD
    mn, mx = svd_all['Score'].min(), svd_all['Score'].max()
    svd_all['Score'] = (svd_all['Score'] - mn) / (mx - mn + 1e-9)

    # Content scores
    df = movies_filt.copy()
    df['genres_str'] = df['genres'].str.replace('|', ' ', regex=False)
    tfidf     = TfidfVectorizer(stop_words='english')
    tfidf_mat = tfidf.fit_transform(df['genres_str'])
    cos_sim   = cosine_similarity(tfidf_mat, tfidf_mat)
    idx       = df[df['title'] == movie_title].index[0]
    content   = pd.DataFrame({'Tytuł': df['title'].values,
                               'Content_Score': cos_sim[idx]})

    hybrid = pd.merge(svd_all, content, on='Tytuł')
    hybrid['Hybrid_Score'] = (hybrid['Score'] * w_svd +
                               hybrid['Content_Score'] * w_content)
    res = (hybrid.sort_values('Hybrid_Score', ascending=False)
           .head(top_n)[['Tytuł', 'Hybrid_Score']]
           .reset_index(drop=True))
    res.index += 1
    return res

# ── 6. Testowanie ────────────────────────────────────────────────
# Wybieramy film z dostępnej listy
available = list(df_filtered['title'].unique())
test_movie = 'Matrix, The (1999)' if 'Matrix, The (1999)' in available \
             else available[0]

print(f"\n{'='*20} RAPORT: {test_movie} {'='*20}")

svd_res = get_svd_recs(test_movie)
if svd_res is not None:
    print("\n1. COLLABORATIVE (SVD) — Co oglądali inni fani:")
    print(svd_res.to_string())
else:
    print("\n1. SVD: film niedostępny w macierzy")

cb_res = get_content_recs(test_movie)
if cb_res is not None:
    print("\n2. CONTENT-BASED — Te same gatunki:")
    print(cb_res.to_string())
else:
    print("\n2. Content-Based: film niedostępny")

hy_res = get_hybrid_recs(test_movie)
if hy_res is not None:
    print("\n3. HYBRYDA (SVD + Content, wagi 50/50):")
    print(hy_res.to_string())
else:
    print("\n3. Hybryda: film niedostępny")

# ── 7. Ewaluacja systemu rekomendacji (RMSE na podziale) ─────────
print("\n── Ewaluacja RMSE (SVD na podziale train/test) ─────────")

from sklearn.model_selection import train_test_split as tts

ratings_eval = df_filtered[['userId','title','rating']].copy()
train_rat, test_rat = tts(ratings_eval, test_size=0.2, random_state=42)

matrix_train = (train_rat.pivot_table(index='userId',
                                       columns='title',
                                       values='rating')
                .fillna(0))

svd_eval = TruncatedSVD(n_components=12, random_state=42)
U  = svd_eval.fit_transform(matrix_train.values)
Vt = svd_eval.components_
reconstructed = pd.DataFrame(
    np.dot(U, Vt),
    index=matrix_train.index,
    columns=matrix_train.columns
)

errors = []
for _, row in test_rat.iterrows():
    uid, title, true_r = row['userId'], row['title'], row['rating']
    if uid in reconstructed.index and title in reconstructed.columns:
        pred_r = reconstructed.loc[uid, title]
        errors.append((true_r - pred_r) ** 2)

rmse_svd = np.sqrt(np.mean(errors)) if errors else float('nan')
print(f"SVD RMSE na zbiorze testowym: {rmse_svd:.4f}")
print("(im bliżej 0, tym lepsze dopasowanie do rzeczywistych ocen)")


# ════════════════════════════════════════════════════════════════
# PODSUMOWANIE KOŃCOWE
# ════════════════════════════════════════════════════════════════
print("  PODSUMOWANIE — Zajęcia 3")
print("═"*60)
print(f"\nA) Klasyfikacja")
print(f"   Zwycięzca (CV 5-fold): {best_cv_name}")
print(f"   Accuracy CV:           {cv_results[best_cv_name]:.4f}")
print(f"\nB) Regresja")
print(f"   Najlepszy model (R²):  {best_reg}")
print(f"   R²:  {reg_results[best_reg]['R2']:.4f}")
print(f"   MAE: {reg_results[best_reg]['MAE']:.2f}  "
      f"RMSE: {reg_results[best_reg]['RMSE']:.2f}")
print(f"\nC) Rekomendacje")
print(f"   SVD RMSE (eval):       {rmse_svd:.4f}")
print(f"   Metody: SVD · Content-Based · Hybryda (50/50)")
print("\n=== Zajęcia 3 zakończone ===")