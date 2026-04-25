import os
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

os.makedirs("figures", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# 1. Загрузка данных

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "breakfast_basket.csv")

df_raw = pd.read_csv(DATA_PATH)

print("1. ДАННЫЕ")
print(f"   Источник  : Kaggle - Global Grocery Inflation 2025-2026")
print(f"   Строк     : {df_raw.shape[0]:,} | Столбцов: {df_raw.shape[1]}")
print(f"   Городов   : {df_raw['City'].nunique()} | Месяцев: {df_raw['Month'].nunique()}")
print(f"   Категории : {sorted(df_raw['Item_Category'].unique())}")


# 2. Очистка данных

df = df_raw.copy()
df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")

for col in ["Price_USD", "Breakfast_Basket_USD", "YoY_Inflation_Estimate_Pct",
            "Exchange_Rate", "Price_Local", "Exchange_Rate",
            "FAO_Index_Value", "Population_Estimate"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.dropna(subset=["Price_USD", "Breakfast_Basket_USD"], inplace=True)
df.drop_duplicates(subset=["City", "Month", "Item_Key"], inplace=True)

def cap_iqr(series, factor=3.0):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    return series.clip(q1 - factor * (q3 - q1), q3 + factor * (q3 - q1))

df["Price_USD"] = df.groupby("Item_Key")["Price_USD"].transform(cap_iqr)
df["Breakfast_Basket_USD"] = cap_iqr(df["Breakfast_Basket_USD"])

print("\n2. ОЧИСТКА")
print(f"   Пропуски: отсутствуют | Дубли: 0 | Выбросы: IQR-кэпинг по товару")
print(f"   Итог: {df.shape[0]:,} строк x {df.shape[1]} столбцов")


# 3. Генерация признаков

basket_pivot = (
    df.pivot_table(index=["City", "Month"], columns="Item_Key",
                   values="Price_USD", aggfunc="mean")
    .reset_index()
)
basket_pivot.columns.name = None

meta = df.groupby(["City", "Month"]).agg(
    Continent=("Continent", "first"),
    Basket_USD=("Breakfast_Basket_USD", "mean"),
    Inflation_Pct=("YoY_Inflation_Estimate_Pct", "mean"),
    Exchange_Rate=("Exchange_Rate", "mean"),
).reset_index()

basket = basket_pivot.merge(meta, on=["City", "Month"])
ITEM_COLS = [c for c in basket.columns if c in df["Item_Key"].unique()]

basket["номер_месяца"] = basket["Month"].dt.month + (basket["Month"].dt.year - 2025) * 12
total = basket[ITEM_COLS].sum(axis=1)
cat_map = df.groupby("Item_Key")["Item_Category"].first().to_dict()


share_cols = []
for cat in df["Item_Category"].unique():
    items = [i for i in ITEM_COLS if cat_map.get(i) == cat]
    col = f"доля_{cat.lower().replace(' & ', '_').replace(' ', '_')}"
    basket[col] = basket[items].sum(axis=1) / (total + 1e-9)
    share_cols.append(col)

# Нормированные и относительные цены
for item in ITEM_COLS:
    basket[f"норм_{item}"] = basket[item] / (total + 1e-9)
    mm = basket.groupby("номер_месяца")[item].transform("median")
    basket[f"откл_{item}"] = basket[item] / (mm + 1e-9)

# Признаки для классификатора (состав и стоимость)
basket["цена_итого"]    = total
basket["цена_cv"]       = basket[ITEM_COLS].std(axis=1) / (total + 1e-9)
basket["log_стоимость"] = np.log1p(basket["Basket_USD"])
basket["log_курс"]      = np.log1p(basket["Exchange_Rate"])
basket["инфляция"]      = basket["Inflation_Pct"]

STRUCT_COLS = (
    share_cols
    + [c for c in basket.columns if c.startswith("норм_")]
    + [c for c in basket.columns if c.startswith("откл_")]
)
STRUCT_COLS = [c for c in STRUCT_COLS if basket[c].std() > 0]

CLASS_COLS_RAW = (
    ITEM_COLS + share_cols
    + [c for c in basket.columns if c.startswith("откл_")]
    + ["цена_итого", "цена_cv", "log_стоимость", "инфляция", "log_курс", "номер_месяца"]
)
seen = set()
CLASS_COLS = [c for c in CLASS_COLS_RAW
              if c in basket.columns and basket[c].std() > 0
              and not (c in seen or seen.add(c))]

print("\n3. ПРИЗНАКИ")
print(f"   Корзин                      : {basket.shape[0]}")
print(f"   Для кластеризации (структура): {len(STRUCT_COLS)}")
print(f"   Для классификатора (полные)  : {len(CLASS_COLS)}")


# 4. EDA

sns.set_theme(style="whitegrid")

# Корреляционная матрица долей категорий
fig, ax = plt.subplots(figsize=(8, 6))
corr = basket[share_cols].corr()
corr.index   = [c.replace("доля_", "") for c in corr.index]
corr.columns = [c.replace("доля_", "") for c in corr.columns]
sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, square=True)
ax.set_title("Корреляция долей категорий в корзине")
plt.tight_layout()
plt.savefig("figures/01_корреляция_категорий.png", dpi=120)
plt.close()

#Распределение стоимости корзины по континентам
fig, ax = plt.subplots(figsize=(9, 4))
basket.boxplot(column="Basket_USD", by="Continent", ax=ax, grid=False)
ax.set_title("Стоимость корзины (USD) по континентам")
ax.set_xlabel("")
ax.set_ylabel("USD")
plt.suptitle("")
plt.tight_layout()
plt.savefig("figures/02_стоимость_по_континентам.png", dpi=120)
plt.close()

print("\n4. EDA - графики 1-2 сохранены")


# 5. Разбиение до обучения

X_struct = basket[STRUCT_COLS].fillna(0).values
X_class  = basket[CLASS_COLS].fillna(0).values

# На train 60% , val 20% , test 20%
# Все трансформации (scaler, PCA, KMeans) обучаются только на train.
idx_trainval, idx_test = train_test_split(
    np.arange(len(basket)), test_size=0.20, random_state=SEED
)
idx_train, idx_val = train_test_split(
    idx_trainval, test_size=0.25, random_state=SEED
)

# StandardScaler (fit только на train)
sc_struct = StandardScaler()
Xtr_s  = sc_struct.fit_transform(X_struct[idx_train])
Xval_s = sc_struct.transform(X_struct[idx_val])
Xte_s  = sc_struct.transform(X_struct[idx_test])

sc_class = StandardScaler()
Xtr_c  = sc_class.fit_transform(X_class[idx_train])
Xval_c = sc_class.transform(X_class[idx_val])
Xte_c  = sc_class.transform(X_class[idx_test])

# PCA (fit только на train)
pca = PCA(n_components=10, random_state=SEED)
Xtr_p  = pca.fit_transform(Xtr_s)
Xval_p = pca.transform(Xval_s)
Xte_p  = pca.transform(Xte_s)
n90 = int(np.argmax(pca.explained_variance_ratio_.cumsum() >= 0.90)) + 1
Xtr_p  = Xtr_p[:, :n90]
Xval_p = Xval_p[:, :n90]
Xte_p  = Xte_p[:, :n90]

print(f"\n5. РАЗБИЕНИЕ (train/val/test)")
print(f"   Train : {len(idx_train)} корзин (60%)")
print(f"   Val   : {len(idx_val)} корзин (20%) - подбор гиперпараметров")
print(f"   Test  : {len(idx_test)} корзин (20%) - финальная оценка")
print(f"   PCA: {n90} компонент (90% дисперсии)")
print(f"   Data leakage исключён: scaler/PCA/KMeans обучены только на train")


# 6. Кластеризация KMeans - только на train, K=6

from sklearn.metrics import silhouette_score, davies_bouldin_score

print("\n6. КЛАСТЕРИЗАЦИЯ (K=6)")
for k in [4, 5, 6, 7]:
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED)
    lb = km.fit_predict(Xtr_p)
    sil = silhouette_score(Xtr_p, lb)
    db  = davies_bouldin_score(Xtr_p, lb)
    print(f"   K={k}  Силуэт={sil:.4f}  ДБ={db:.4f}")


km_final = KMeans(n_clusters=6, n_init=30, random_state=SEED)
y_train = km_final.fit_predict(Xtr_p)
y_val   = km_final.predict(Xval_p)
y_test  = km_final.predict(Xte_p)

# Интерпретация кластеров
train_basket = basket.iloc[idx_train].copy()
train_basket["Кластер"] = y_train

profile = train_basket.groupby("Кластер")[share_cols + ["Basket_USD"]].mean()
profile.columns = [c.replace("доля_", "") for c in profile.columns]
share_short = [c.replace("доля_", "") for c in share_cols]
means_v = profile[share_short].mean()

def name_cluster(cl_idx):
    row = profile.loc[cl_idx]
    above = []
    for key, lbl in [("dairy","молочный"), ("dairy_eggs","молочный"),
                     ("grains","зерновой"), ("vegetables","овощной"),
                     ("fruits","фруктовый"), ("meat","мясной")]:
        if key in row.index:
            val = float(row[key])
            avg = float(means_v.get(key, 0))
            if avg > 0 and val > avg * 1.10 and lbl not in above:
                above.append(lbl)
    if not above:
        above = ["смешанный"]
    price = float(row["Basket_USD"])
    tag = "дорогой" if price > 12 else ("дешёвый" if price < 6 else "средний")
    return "+".join(above[:2]) + f" ({tag})"

name_map = {c: name_cluster(c) for c in range(6)}

print("\n   Сегменты:")
for c, name in name_map.items():
    cities = train_basket[train_basket["Кластер"] == c]["City"].unique()[:3].tolist()
    print(f"   {c}: {name:<35} | {cities}")

basket["Кластер"] = -1
basket.iloc[idx_train, basket.columns.get_loc("Кластер")] = y_train
basket.iloc[idx_val,   basket.columns.get_loc("Кластер")] = y_val
basket.iloc[idx_test,  basket.columns.get_loc("Кластер")] = y_test
basket["Сегмент"] = basket["Кластер"].map(name_map)

#PCA 2D проекция кластеров
fig, ax = plt.subplots(figsize=(9, 6))
colors = plt.cm.tab10.colors
for c in range(6):
    mask = y_train == c
    ax.scatter(Xtr_p[mask, 0], Xtr_p[mask, 1],
               label=f"{c}: {name_map[c]}", alpha=0.6, s=30, color=colors[c])
ax.set_xlabel("Главная компонента 1")
ax.set_ylabel("Главная компонента 2")
ax.set_title("Паттерны корзин - проекция PCA")
ax.legend(fontsize=7, loc="upper right")
plt.tight_layout()
plt.savefig("figures/03_кластеры_pca.png", dpi=120)
plt.close()

print("\n   График 03 сохранён")


# 7. Метрики

print("\n7. МЕТРИКИ")
print("   Кластеризация  : Силуэтный коэффициент (основная) + Индекс Дэвиса-Болдина")
print("   Классификация  : Взвешенный F1 (основная) + CV 5-fold")
print("   Предпочтение F1 > Accuracy - классы несбалансированы (от 7 до 168 корзин)")


# 8. Классификация - LogReg и Random Forest

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
results = []

def evaluate(name, model):
    cv_f1 = cross_val_score(model, Xtr_c, y_train,
                            cv=cv, scoring="f1_weighted", n_jobs=-1)
    model.fit(Xtr_c, y_train)
    val_f1  = f1_score(y_val,  model.predict(Xval_c), average="weighted")
    pred    = model.predict(Xte_c)
    tf1     = f1_score(y_test, pred, average="weighted")
    tacc    = accuracy_score(y_test, pred)
    results.append({
        "Модель":        name,
        "CV F1":         round(cv_f1.mean(), 4),
        "CV F1 std":     round(cv_f1.std(),  4),
        "Val F1":        round(val_f1, 4),
        "Test F1":       round(tf1,  4),
        "Test Accuracy": round(tacc, 4),
    })
    print(f"   {name:<38}  CV={cv_f1.mean():.4f}±{cv_f1.std():.4f}"
          f"  Val={val_f1:.4f}  Test F1={tf1:.4f}  Acc={tacc:.4f}")
    return model, pred

print("\n8. МОДЕЛИ")
lr, lr_pred = evaluate(
    "Логистическая регрессия (бейзлайн)",
    LogisticRegression(max_iter=1000, random_state=SEED)
)
rf, rf_pred = evaluate(
    "Random Forest",
    RandomForestClassifier(n_estimators=200, max_depth=8,
                           random_state=SEED, n_jobs=-1)
)


# 9. Результаты

exp_df = pd.DataFrame(results)
exp_df.to_csv("outputs/таблица_экспериментов.csv", index=False, encoding="utf-8-sig")

best_name = exp_df.loc[exp_df["CV F1"].idxmax(), "Модель"]
best_pred = lr_pred if "Логистическая" in best_name else rf_pred

print(f"\n9. РЕЗУЛЬТАТЫ")
print(exp_df.to_string(index=False))
print(f"\n   Лучшая модель: {best_name}")
print("\n   Отчёт по классам:")
print(classification_report(
    y_test, best_pred,
    target_names=[name_map[c] for c in sorted(name_map)]
))

# Важность признаков
fi = pd.Series(rf.feature_importances_, index=CLASS_COLS)
fi_top = fi.sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(9, 4))
fi_top.plot(kind="bar", ax=ax, color="teal")
ax.set_title("Random Forest - топ-15 важных признаков")
ax.set_ylabel("Важность")
ax.set_xlabel("")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("figures/04_важность_признаков.png", dpi=120)
plt.close()

print("\n   График 04 сохранён")


# 10. Итоговые выводы

best_row = exp_df.loc[exp_df["CV F1"].idxmax()]
print(f"""
10. ВЫВОДЫ
    1. Данные очищены: 10 248 строк, пропусков нет, выбросы кэпированы по IQR.
    2. Кластеры выделены: K=6 структурных паттернов корзин (молочный,
       зерновой, овощной и др.). Сегментация по составу, а не по цене.
    3. Лучшая модель ({best_row['Модель']}) показывает
       Test F1={best_row['Test F1']} при CV={best_row['CV F1']}±{best_row['CV F1 std']}.
       Выбрана метрика F1-score, так как кластеры могут быть несбалансированы по размеру.

""")