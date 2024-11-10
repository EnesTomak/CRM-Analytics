import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter, GammaGammaFitter
from sklearn.model_selection import ParameterGrid
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Outlier tresholds to handle extreme values
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Veri setini okuma
df_ = pd.read_excel("datasets/online_retail.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

def create_cltv_p(dataframe):
    # 1. Veri Ön İşleme
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]  # İptal edilen işlemler çıkarılır
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]

    # Analiz tarihi
    today_date = dt.datetime(2011, 12, 11)

    # Müşteri bazlı verilerin hazırlanması
    cltv_df = dataframe.groupby('Customer ID').agg(
        {'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                         lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
         'Invoice': lambda Invoice: Invoice.nunique(),
         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

    cltv_df.columns = cltv_df.columns.droplevel(0)
    cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
    cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
    cltv_df = cltv_df[cltv_df['frequency'] > 1]  # En az 2 işlem yapan müşteriler
    cltv_df["recency"] = cltv_df["recency"] / 7  # Haftalık olarak
    cltv_df["T"] = cltv_df["T"] / 7

    return cltv_df

# CLTV tahmin modeli
def model_cltv(cltv_df, month=6, best_bgf_params=None, best_ggf_params=None):
    # 2. BG-NBD Modelinin Kurulması
    bgf = BetaGeoFitter(**best_bgf_params)
    bgf.fit(cltv_df['frequency'], cltv_df['recency'], cltv_df['T'])

    # 1 Aylık ve 12 Aylık Tahmin
    cltv_df["expected_purc_1_month"] = bgf.predict(4, cltv_df['frequency'], cltv_df['recency'], cltv_df['T'])
    cltv_df["expected_purc_12_month"] = bgf.predict(4 * 12, cltv_df['frequency'], cltv_df['recency'], cltv_df['T'])

    # 3. GAMMA-GAMMA Modelinin Kurulması
    ggf = GammaGammaFitter(**best_ggf_params)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary'])
    cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                                 cltv_df['monetary'])

    # 4. CLTV'nin hesaplanması (6 aylık tahmin)
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency'],
                                       cltv_df['T'],
                                       cltv_df['monetary'],
                                       time=month,  # 6 aylık tahmin
                                       freq="W",  # Haftalık
                                       discount_rate=0.01)

    cltv = cltv.reset_index()
    cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")

    # 5. CLTV Segmentasyonu (4 Segment)
    cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

    return cltv_final


# Hiperparametre optimizasyonu (BG/NBD Modeli için)
def tune_bgf(cltv_df):
    param_grid = {'penalizer_coef': [0.001, 0.01, 0.1, 0.5]}
    best_score = float('inf')
    best_params = None
    for params in ParameterGrid(param_grid):
        bgf = BetaGeoFitter(**params)
        bgf.fit(cltv_df['frequency'], cltv_df['recency'], cltv_df['T'])
        predicted_clv = bgf.predict(4 * 6, cltv_df['frequency'], cltv_df['recency'], cltv_df['T'])
        score = ((cltv_df["monetary"] - predicted_clv).abs()).mean()  # MAE as score
        if score < best_score:
            best_score = score
            best_params = params

    print(f"Best BG/NBD params: {best_params}, Score: {best_score}")
    return best_params

# Hiperparametre optimizasyonu (Gamma-Gamma Modeli için)
def tune_ggf(cltv_df):
    param_grid = {'penalizer_coef': [0.001, 0.01, 0.1, 0.5]}
    best_score = float('inf')
    best_params = None
    for params in ParameterGrid(param_grid):
        ggf = GammaGammaFitter(**params)
        ggf.fit(cltv_df['frequency'], cltv_df['monetary'])
        predicted_avg_value = ggf.conditional_expected_average_profit(cltv_df['frequency'], cltv_df['monetary'])
        score = ((cltv_df["monetary"] - predicted_avg_value).abs()).mean()  # MAE as score
        if score < best_score:
            best_score = score
            best_params = params

    print(f"Best Gamma-Gamma params: {best_params}, Score: {best_score}")
    return best_params

# CLTV verisinin hazırlanması
cltv_df = create_cltv_p(df)

# BG/NBD ve Gamma-Gamma hiperparametre optimizasyonu
best_bgf_params = tune_bgf(cltv_df)
best_ggf_params = tune_ggf(cltv_df)

# En iyi parametrelerle modeli çalıştırma
cltv_final = model_cltv(cltv_df, month=6, best_bgf_params=best_bgf_params, best_ggf_params=best_ggf_params)

# Sonuçları yazdır
print(cltv_final.head())


def plot_cltv_segments(cltv_final):
    plt.figure(figsize=(10, 6))
    sns.barplot(x="segment", y="clv", data=cltv_final, palette="viridis", ci=None)
    plt.title("CLTV Segmentlerine Göre Dağılım", fontsize=16)
    plt.xlabel("Segment", fontsize=12)
    plt.ylabel("CLTV", fontsize=12)
    plt.show()


plot_cltv_segments(cltv_final)


def plot_cltv_by_country(dataframe, cltv_final):
    # Merge the original dataframe with CLTV final to add country info
    country_cltv = dataframe[['Customer ID', 'Country']].drop_duplicates().merge(cltv_final, on='Customer ID')

    plt.figure(figsize=(12, 8))
    sns.barplot(x='clv', y='Country', data=country_cltv, palette='coolwarm', ci=None)
    plt.title('CLTV by Country', fontsize=16)
    plt.xlabel('CLTV', fontsize=12)
    plt.ylabel('Country', fontsize=12)
    plt.show()


plot_cltv_by_country(df, cltv_final)

