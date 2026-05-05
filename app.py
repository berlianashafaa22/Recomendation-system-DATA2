
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from itertools import combinations
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎵",
    layout="wide"
)

# ─────────────────────────────────────────
# LOAD & PROCESS DATA (CACHE RESOURCE)
# ─────────────────────────────────────────
@st.cache_resource
def load_all_data():
    df_raw = pd.read_csv("music.csv")
    df_clean = df_raw.copy()
    df_clean = df_clean[~df_clean["genres"].str.contains("no genres listed")]
    
    genres_series = df_clean["genres"].str.split("|").apply(lambda x: tuple(g.strip() for g in x))
    
    df_clean["num_genres"] = genres_series.apply(len)
    df_clean["decade"] = (df_clean["year"] // 10 * 10).astype(str) + "s"

    all_genres = genres_series.explode()
    all_genres = all_genres[all_genres != "no genres listed"]
    unique_genres = sorted(all_genres.unique().tolist())

    for genre in unique_genres:
        df_clean[f"genre_{genre}"] = genres_series.apply(lambda x: 1 if genre in x else 0)

    scaler = MinMaxScaler()
    df_clean["year_normalized"] = scaler.fit_transform(df_clean[["year"]]).flatten()

    genre_cols = [f"genre_{g}" for g in unique_genres]
    feature_cols = genre_cols + ["year_normalized"]
    feature_matrix = df_clean[feature_cols].values

    te = TransactionEncoder()
    te_array = te.fit_transform(genres_series.apply(list).tolist())
    genre_encoded = pd.DataFrame(te_array, columns=te.columns_)
    frequent_genres = apriori(genre_encoded, min_support=0.05, use_colnames=True, max_len=2)
    genre_rules = association_rules(frequent_genres, metric="lift", min_threshold=1.0)
    genre_rules = genre_rules.sort_values("lift", ascending=False).reset_index(drop=True)

    cooccurrence = pd.DataFrame(0, index=unique_genres, columns=unique_genres)
    for genres_tuple in genres_series:
        clean = [g for g in genres_tuple if g in unique_genres]
        for g1, g2 in combinations(clean, 2):
            cooccurrence.loc[g1, g2] += 1
            cooccurrence.loc[g2, g1] += 1

    return df_raw, df_clean, tuple(unique_genres), feature_matrix, genre_rules, cooccurrence

# ─────────────────────────────────────────
# FUNGSI REKOMENDASI
# ─────────────────────────────────────────
def get_similarity_scores_on_demand(song_title, df_clean, feature_matrix):
    row_indices = np.where(df_clean["title"] == song_title)[0]
    if len(row_indices) == 0:
        return {}
    idx = row_indices[0] 
    
    song_vector = feature_matrix[idx].reshape(1, -1)
    sim_scores = cosine_similarity(song_vector, feature_matrix).flatten()
    
    titles = df_clean["title"].values
    return {titles[i]: float(sim_scores[i]) for i in range(len(titles)) if titles[i] != song_title}

def cb_recommend(song_title, df_clean, feature_matrix, top_n=5):
    scores = get_similarity_scores_on_demand(song_title, df_clean, feature_matrix)
    if not scores:
        return pd.DataFrame(columns=["Judul", "Genre", "Tahun", "CB Score"])
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    result = []
    for title, score in top:
        rows = df_clean[df_clean["title"] == title]
        if len(rows) == 0: continue
        result.append({
            "Judul": title, "Genre": rows["genres"].values[0],
            "Tahun": int(rows["year"].values[0]), "CB Score": round(score, 4)
        })
    return pd.DataFrame(result)

def pseudo_cf_recommend(song_title, df_clean, genre_rules, top_n=5):
    rows = df_clean[df_clean["title"] == song_title]
    if len(rows) == 0:
        return pd.DataFrame(columns=["Judul", "Genre", "Tahun", "CF Score"])

    song_genres = [g.strip() for g in rows["genres"].values[0].split("|")]
    related_genres = set()
    for genre in song_genres:
        rules_filtered = genre_rules[genre_rules["antecedents"].apply(lambda x: genre in x)]
        for _, row in rules_filtered.iterrows():
            for g in row["consequents"]: related_genres.add(g)

    if not related_genres: return pd.DataFrame(columns=["Judul", "Genre", "Tahun", "CF Score"])

    related_cols = [f"genre_{g}" for g in related_genres if f"genre_{g}" in df_clean.columns]
    if not related_cols: return pd.DataFrame(columns=["Judul", "Genre", "Tahun", "CF Score"])

    mask = df_clean["title"] != song_title
    overlap_counts = df_clean.loc[mask, related_cols].sum(axis=1)
    valid_indices = overlap_counts[overlap_counts > 0].index
    
    if len(valid_indices) == 0: return pd.DataFrame(columns=["Judul", "Genre", "Tahun", "CF Score"])
        
    cf_scores = overlap_counts.loc[valid_indices] / len(related_genres)
    top_indices = cf_scores.sort_values(ascending=False).head(top_n).index
    
    result = []
    for idx in top_indices:
        row = df_clean.loc[idx]
        result.append({
            "Judul": row["title"], "Genre": row["genres"],
            "Tahun": int(row["year"]), "CF Score": round(cf_scores.loc[idx], 4)
        })
    return pd.DataFrame(result)

def hybrid_recommend(song_title, df_clean, feature_matrix, genre_rules, top_n=5):
    if song_title not in df_clean["title"].values:
        return pd.DataFrame(columns=["Judul", "Genre", "Tahun", "Hybrid Score"])

    cb_scores = get_similarity_scores_on_demand(song_title, df_clean, feature_matrix)
    cf_result = pseudo_cf_recommend(song_title, df_clean, genre_rules, top_n=50)
    
    cf_scores = {row["Judul"]: float(row["CF Score"]) for _, row in cf_result.iterrows()} if len(cf_result) > 0 else {}

    if cb_scores:
        mx = max(cb_scores.values())
        if mx > 0: cb_scores = {k: v/mx for k, v in cb_scores.items()}
    if cf_scores:
        mx = max(cf_scores.values())
        if mx > 0: cf_scores = {k: v/mx for k, v in cf_scores.items()}

    all_songs = set(list(cb_scores.keys()) + list(cf_scores.keys()))
    hybrid_scores = {s: 0.5 * cb_scores.get(s, 0) + 0.5 * cf_scores.get(s, 0) for s in all_songs}
    top_songs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    result = []
    for title, score in top_songs:
        rows = df_clean[df_clean["title"] == title]
        if len(rows) == 0: continue
        result.append({
            "Judul": title, "Genre": rows["genres"].values[0],
            "Tahun": int(rows["year"].values[0]), "Hybrid Score": round(score, 4)
        })
    return pd.DataFrame(result)

# ─────────────────────────────────────────
# EKSEKUSI DATA 
# ─────────────────────────────────────────
with st.spinner("Memuat dan memproses data..."):
    df_raw, df_clean, unique_genres, feature_matrix, genre_rules, cooccurrence = load_all_data()

all_genres_series = df_clean["genres"].str.split("|").explode().str.strip()
all_genres_series = all_genres_series[all_genres_series != "no genres listed"]

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/music.png")
    st.title("🎵 MusicRec")
    st.caption("Music Recommendation System")
    st.divider()
    page = st.selectbox("Navigasi Halaman", ["🏠 Home", "📊 Eksplorasi Data", "🤖 Sistem Rekomendasi", "🔬 Analisis Genre Rules", "📈 Evaluasi Metode", "💼 Implikasi Manajerial"])
    st.divider()
    st.caption(f"🎵 Total lagu: {len(df_clean):,}")
    st.caption(f"🎸 Genre unik: {len(unique_genres)}")
    st.caption(f"📅 {df_raw['year'].min()} - {df_raw['year'].max()}")
    st.caption(f"📋 Genre rules: {len(genre_rules)}")

# ─────────────────────────────────────────
# HALAMAN 1: HOME
# ─────────────────────────────────────────
if page == "🏠 Home":
    st.title("🎵 Music Recommendation System")
    st.subheader("Genre-Based & Pseudo Collaborative Filtering")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Lagu", f"{len(df_clean):,}")
    col2.metric("Genre Unik", f"{len(unique_genres)}")
    col3.metric("Rentang Tahun", f"{df_raw['year'].min()}–{df_raw['year'].max()}")
    col4.metric("Genre Rules", f"{len(genre_rules)}")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        dominant_decade = df_clean["decade"].value_counts().idxmax()
        dominant_pct = df_clean["decade"].value_counts().max() / len(df_clean) * 100
        top_genre = all_genres_series.value_counts().index[0]
        st.info(f"**📌 Insight Utama**\n- Genre paling dominan: **{top_genre}**\n- Dekade terbanyak: **{dominant_decade}** ({dominant_pct:.1f}% data)\n- Rata-rata genre per lagu: **{df_clean['num_genres'].mean():.2f}**\n- Ditemukan **{len(genre_rules)} genre association rules**")
    with col2:
        st.success("**🎯 Tentang Sistem Ini**\n- Menggunakan **Switching Hybrid Recommendation**\n- **Content-Based**: kemiripan genre + tahun (Cosine Similarity)\n- **Pseudo-CF**: pola co-occurrence genre (Association Rules)\n- Cold-start fallback ke popularitas genre")

# ─────────────────────────────────────────
# HALAMAN 2: EDA
# ─────────────────────────────────────────
elif page == "📊 Eksplorasi Data":
    st.title("📊 Eksplorasi Data")
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["🎸 Distribusi Genre", "📅 Analisis Dekade", "🎵 Karakteristik Lagu", "📈 Tren Genre per Dekade"])

    with tab1:
        genre_counts = all_genres_series.value_counts()
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(genre_counts.sort_values(), orientation="h", color=genre_counts.sort_values().values, color_continuous_scale="Blues", title="Frekuensi Tiap Genre")
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig)
        with col2:
            top8 = genre_counts.head(8)
            others = genre_counts[8:].sum()
            pie_data = pd.concat([top8, pd.Series({"Others": others})])
            fig = px.pie(values=pie_data.values, names=pie_data.index, title="Proporsi Genre (Top 8 + Others)", color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig)

    with tab2:
        decade_counts = df_clean["decade"].value_counts().sort_index()
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(decade_counts, color=decade_counts.values, color_continuous_scale="Oranges", title="Jumlah Lagu per Dekade")
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig)
        with col2:
            fig = px.line(x=decade_counts.index, y=decade_counts.values, markers=True, title="Tren Jumlah Lagu per Dekade", color_discrete_sequence=["coral"])
            st.plotly_chart(fig)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            genre_per_song = df_clean["num_genres"].value_counts().sort_index()
            fig = px.bar(genre_per_song, color=genre_per_song.values, color_continuous_scale="Purples", title="Distribusi Jumlah Genre per Lagu")
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig)
        with col2:
            pairs = [{"Genre 1": g1, "Genre 2": g2, "Co-occurrence": int(cooccurrence.loc[g1, g2])} for g1 in unique_genres for g2 in unique_genres if g1 < g2 and cooccurrence.loc[g1, g2] > 0]
            st.dataframe(pd.DataFrame(pairs).sort_values("Co-occurrence", ascending=False).head(10))
        fig = px.imshow(cooccurrence, color_continuous_scale="Blues", title="Co-occurrence Antar Genre", text_auto=True)
        st.plotly_chart(fig)

    with tab4:
        df_exploded = df_clean.copy()
        df_exploded["genres_list"] = df_exploded["genres"].str.split("|").apply(lambda x: tuple(g.strip() for g in x))
        df_exploded = df_exploded.explode("genres_list")
        df_exploded = df_exploded[df_exploded["genres_list"].isin(unique_genres)]

        genre_decade = df_exploded.groupby(["decade", "genres_list"]).size().unstack(fill_value=0)
        top_genres_list = all_genres_series.value_counts().head(8).index.tolist()
        genre_decade_top = genre_decade[[g for g in top_genres_list if g in genre_decade.columns]]

        fig = px.imshow(genre_decade_top.T, color_continuous_scale="YlOrRd", title="Heatmap Genre per Dekade", text_auto=True)
        st.plotly_chart(fig)

# ─────────────────────────────────────────
# HALAMAN 3: SISTEM REKOMENDASI
# ─────────────────────────────────────────
elif page == "🤖 Sistem Rekomendasi":
    st.title("🤖 Sistem Rekomendasi Musik")
    st.markdown("---")
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1: selected_song = st.selectbox("🎵 Pilih Lagu", sorted(df_clean["title"].unique().tolist()))
    with col2: top_n = st.slider("Jumlah Rekomendasi", 3, 10, 5)
    with col3: method = st.radio("Metode", ["Hybrid", "Content-Based", "Pseudo-CF"], index=0)

    st.markdown("---")
    song_info = df_clean[df_clean["title"] == selected_song].iloc[0]
    col1, col2, col3 = st.columns(3)
    col1.info(f"🎵 **{selected_song}**")
    col2.info(f"🎸 {song_info['genres']}")
    col3.info(f"📅 {song_info['year']} ({song_info['decade']})")

    if st.button("🚀 Cari Rekomendasi"):
        if method == "Hybrid": result, score_col, color = hybrid_recommend(selected_song, df_clean, feature_matrix, genre_rules, top_n), "Hybrid Score", "🟡"
        elif method == "Content-Based": result, score_col, color = cb_recommend(selected_song, df_clean, feature_matrix, top_n), "CB Score", "🔵"
        else: result, score_col, color = pseudo_cf_recommend(selected_song, df_clean, genre_rules, top_n), "CF Score", "🟢"

        st.subheader(f"{color} Rekomendasi untuk: **{selected_song}**")
        if len(result) == 0: st.warning("⚠️ Tidak ada rekomendasi. Coba metode lain.")
        else:
            for i, row in result.iterrows():
                c1, c2, c3 = st.columns([3, 2, 1])
                c1.markdown(f"**{i+1}. {row['Judul']}**")
                c2.caption(f"🎸 {row['Genre']} | 📅 {row['Tahun']}")
                c3.progress(float(min(row[score_col], 1.0)))

# ─────────────────────────────────────────
# HALAMAN 4: ANALISIS GENRE RULES
# ─────────────────────────────────────────
elif page == "🔬 Analisis Genre Rules":
    st.title("🔬 Analisis Genre Association Rules")
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    min_sup, min_conf, min_lift = col1.slider("Min Support", 0.01, 0.5, 0.05), col2.slider("Min Conf", 0.1, 1.0, 0.3), col3.slider("Min Lift", 1.0, 10.0, 1.0)

    filtered = genre_rules[(genre_rules["support"] >= min_sup) & (genre_rules["confidence"] >= min_conf) & (genre_rules["lift"] >= min_lift)].copy()
    filtered["antecedents"] = filtered["antecedents"].apply(lambda x: ", ".join(list(x)))
    filtered["consequents"] = filtered["consequents"].apply(lambda x: ", ".join(list(x)))

    st.metric("Rules yang memenuhi filter", len(filtered))
    st.dataframe(filtered[["antecedents","consequents","support","confidence","lift"]].round(4).reset_index(drop=True))

# ─────────────────────────────────────────
# HALAMAN 5: EVALUASI
# ─────────────────────────────────────────
elif page == "📈 Evaluasi Metode":
    st.title("📈 Evaluasi & Perbandingan Metode")
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📊 Coverage", "🎯 Diversity", "📅 Decade Bias"])

    with tab1:
        antecedents_list = genre_rules["antecedents"].apply(lambda a: list(a)[0]).tolist()
        cf_covered = len(df_clean[df_clean["genres"].apply(lambda x: any(g.strip() in antecedents_list for g in x.split("|")))])
        st.dataframe(pd.DataFrame({"Metode": ["Content-Based", "Pseudo-CF", "Hybrid"], "Lagu Tercovered": [len(df_clean), cf_covered, len(df_clean)], "Coverage (%)": [100.0, cf_covered/len(df_clean)*100, 100.0]}))

    with tab2:
        def diversity_score(recs): return round(recs["Genre"].str.split("|").explode().nunique() / len(recs["Genre"].str.split("|").explode()), 4) if len(recs) > 0 else 0
        div_results = [{"Lagu": song[:35], "CB": diversity_score(cb_recommend(song, df_clean, feature_matrix)), "Pseudo-CF": diversity_score(pseudo_cf_recommend(song, df_clean, genre_rules)), "Hybrid": diversity_score(hybrid_recommend(song, df_clean, feature_matrix, genre_rules))} for song in df_clean["title"].sample(10, random_state=42).tolist()]
        st.dataframe(pd.DataFrame(div_results))

    with tab3:
        fig = px.bar(df_clean["decade"].value_counts().sort_index(), title="Distribusi Dekade — Data Asli", color_discrete_sequence=["steelblue"])
        st.plotly_chart(fig)

# ─────────────────────────────────────────
# HALAMAN 6: IMPLIKASI MANAJERIAL
# ─────────────────────────────────────────
elif page == "💼 Implikasi Manajerial":
    st.title("💼 Implikasi Manajerial")
    st.markdown("---")
    st.subheader("🚀 Roadmap Pengembangan")
    col1, col2, col3 = st.columns(3)
    col1.info("**Fase 1 (Pendek)**\n- Tambah data era lain\n- Spotify API integrasi")
    col2.warning("**Fase 2 (Menengah)**\n- Kumpul interaksi user\n- Implementasi CF User-based")
    col3.error("**Fase 3 (Panjang)**\n- Deep learning audio\n- Real-time recommendation")
