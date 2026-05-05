
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
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
# LOAD & PROCESS DATA
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("music.csv")
    return df

@st.cache_data
def preprocess(df):
    df_clean = df.copy()
    df_clean = df_clean[~df_clean["genres"].str.contains("no genres listed")]
    df_clean["genres_list"] = df_clean["genres"].str.split("|").apply(
        lambda x: [g.strip() for g in x]
    )
    df_clean["num_genres"] = df_clean["genres_list"].apply(len)
    df_clean["decade"] = (df_clean["year"] // 10 * 10).astype(str) + "s"

    all_genres = df_clean["genres"].str.split("|").explode().str.strip()
    all_genres = all_genres[all_genres != "no genres listed"]
    unique_genres = sorted(all_genres.unique().tolist())

    for genre in unique_genres:
        df_clean[f"genre_{genre}"] = df_clean["genres_list"].apply(
            lambda x: 1 if genre in x else 0
        )

    scaler = MinMaxScaler()
    df_clean["year_normalized"] = scaler.fit_transform(df_clean[["year"]])

    genre_cols = [f"genre_{g}" for g in unique_genres]
    feature_cols = genre_cols + ["year_normalized"]
    feature_matrix = df_clean[feature_cols].values

    return df_clean, unique_genres, feature_matrix

@st.cache_data
def build_similarity(_feature_matrix, titles):
    cb_sim = cosine_similarity(_feature_matrix)
    cb_sim_df = pd.DataFrame(cb_sim, index=titles, columns=titles)
    return cb_sim_df

@st.cache_data
def build_genre_rules(df_clean):
    genre_transactions = df_clean["genres_list"].tolist()
    te = TransactionEncoder()
    te_array = te.fit_transform(genre_transactions)
    genre_encoded = pd.DataFrame(te_array, columns=te.columns_)
    frequent_genres = apriori(genre_encoded, min_support=0.05,
                               use_colnames=True, max_len=2)
    genre_rules = association_rules(frequent_genres, metric="lift",
                                     min_threshold=1.0)
    genre_rules = genre_rules.sort_values("lift", ascending=False).reset_index(drop=True)
    return genre_rules

@st.cache_data
def build_cooccurrence(df_clean, unique_genres):
    cooccurrence = pd.DataFrame(0, index=unique_genres, columns=unique_genres)
    for genres in df_clean["genres_list"]:
        genres_clean = [g for g in genres if g in unique_genres]
        for g1, g2 in combinations(genres_clean, 2):
            cooccurrence.loc[g1, g2] += 1
            cooccurrence.loc[g2, g1] += 1
    return cooccurrence

# Fungsi rekomendasi
def get_similarity_scores(song_title, cb_sim_df):
    if song_title not in cb_sim_df.index:
        return {}
    sim = cb_sim_df[song_title]
    if isinstance(sim, pd.DataFrame):
        sim = sim.iloc[0]
    sim = sim[sim.index != song_title]
    result = {}
    for k, v in sim.items():
        result[k] = float(v.iloc[0]) if isinstance(v, pd.Series) else float(v)
    return result

def cb_recommend(song_title, df_clean, cb_sim_df, top_n=5):
    scores = get_similarity_scores(song_title, cb_sim_df)
    if not scores:
        return pd.DataFrame(columns=["Judul", "Genre", "Tahun", "CB Score"])
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    result = []
    for title, score in top:
        rows = df_clean[df_clean["title"] == title]
        if len(rows) == 0:
            continue
        result.append({
            "Judul": title,
            "Genre": rows["genres"].values[0],
            "Tahun": int(rows["year"].values[0]),
            "CB Score": round(score, 4)
        })
    return pd.DataFrame(result)

def pseudo_cf_recommend(song_title, df_clean, genre_rules, top_n=5):
    rows = df_clean[df_clean["title"] == song_title]
    if len(rows) == 0:
        return pd.DataFrame(columns=["Judul", "Genre", "Tahun", "CF Score"])

    song_genres = rows["genres_list"].values[0]
    related_genres = set()
    for genre in song_genres:
        rules_filtered = genre_rules[
            genre_rules["antecedents"].apply(lambda x: genre in x)
        ]
        for _, row in rules_filtered.iterrows():
            for g in row["consequents"]:
                related_genres.add(g)

    if not related_genres:
        return pd.DataFrame(columns=["Judul", "Genre", "Tahun", "CF Score"])

    cf_scores = {}
    for _, song_row in df_clean[df_clean["title"] != song_title].iterrows():
        overlap = set(song_row["genres_list"]) & related_genres
        if overlap:
            cf_scores[song_row["title"]] = len(overlap) / len(related_genres)

    result_items = sorted(cf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    result = []
    for title, score in result_items:
        rows = df_clean[df_clean["title"] == title]
        if len(rows) == 0:
            continue
        result.append({
            "Judul": title,
            "Genre": rows["genres"].values[0],
            "Tahun": int(rows["year"].values[0]),
            "CF Score": round(score, 4)
        })
    return pd.DataFrame(result)

def hybrid_recommend(song_title, df_clean, cb_sim_df, genre_rules, top_n=5):
    if song_title not in df_clean["title"].values:
        return pd.DataFrame(columns=["Judul", "Genre", "Tahun", "Hybrid Score"])

    cb_scores = get_similarity_scores(song_title, cb_sim_df)
    cf_result = pseudo_cf_recommend(song_title, df_clean, genre_rules, top_n=50)
    cf_scores = {}
    if len(cf_result) > 0:
        for _, row in cf_result.iterrows():
            cf_scores[row["Judul"]] = float(row["CF Score"])

    if cb_scores:
        mx = max(cb_scores.values())
        if mx > 0:
            cb_scores = {k: v/mx for k, v in cb_scores.items()}
    if cf_scores:
        mx = max(cf_scores.values())
        if mx > 0:
            cf_scores = {k: v/mx for k, v in cf_scores.items()}

    all_songs = set(list(cb_scores.keys()) + list(cf_scores.keys()))
    hybrid_scores = {
        s: 0.5 * cb_scores.get(s, 0) + 0.5 * cf_scores.get(s, 0)
        for s in all_songs
    }

    top_songs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    result = []
    for title, score in top_songs:
        rows = df_clean[df_clean["title"] == title]
        if len(rows) == 0:
            continue
        result.append({
            "Judul": title,
            "Genre": rows["genres"].values[0],
            "Tahun": int(rows["year"].values[0]),
            "Hybrid Score": round(score, 4)
        })
    return pd.DataFrame(result)

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
with st.spinner("Memuat dan memproses data..."):
    df_raw = load_data()
    df_clean, unique_genres, feature_matrix = preprocess(df_raw)
    cb_sim_df = build_similarity(feature_matrix, df_clean["title"].tolist())
    genre_rules = build_genre_rules(df_clean)
    cooccurrence = build_cooccurrence(df_clean, unique_genres)

all_genres_series = df_clean["genres"].str.split("|").explode().str.strip()
all_genres_series = all_genres_series[all_genres_series != "no genres listed"]

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/music.png", width=80)
    st.title("🎵 MusicRec")
    st.caption("Music Recommendation System")
    st.divider()

    page = st.selectbox(
        "Navigasi Halaman",
        ["🏠 Home",
         "📊 Eksplorasi Data",
         "🤖 Sistem Rekomendasi",
         "🔬 Analisis Genre Rules",
         "📈 Evaluasi Metode",
         "💼 Implikasi Manajerial"]
    )
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
        st.info(f"""
        **📌 Insight Utama**
        - Genre paling dominan: **{top_genre}**
        - Dekade terbanyak: **{dominant_decade}** ({dominant_pct:.1f}% data)
        - Rata-rata genre per lagu: **{df_clean["num_genres"].mean():.2f}**
        - Ditemukan **{len(genre_rules)} genre association rules**
        """)
    with col2:
        st.success(f"""
        **🎯 Tentang Sistem Ini**
        - Menggunakan **Switching Hybrid Recommendation**
        - **Content-Based**: kemiripan genre + tahun (Cosine Similarity)
        - **Pseudo-CF**: pola co-occurrence genre (Association Rules)
        - Cold-start fallback ke popularitas genre
        """)

    st.markdown("---")
    st.markdown("### 🚀 Mulai Eksplorasi")
    col1, col2, col3 = st.columns(3)
    col1.info("📊 **Eksplorasi Data** Distribusi genre, dekade, dan co-occurrence")
    col2.success("🤖 **Sistem Rekomendasi** Cari rekomendasi lagu secara interaktif")
    col3.warning("💼 **Implikasi Manajerial** Temuan dan rekomendasi pengembangan platform")

# ─────────────────────────────────────────
# HALAMAN 2: EDA
# ─────────────────────────────────────────
elif page == "📊 Eksplorasi Data":
    st.title("📊 Eksplorasi Data")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎸 Distribusi Genre",
        "📅 Analisis Dekade",
        "🎵 Karakteristik Lagu",
        "📈 Tren Genre per Dekade"
    ])

    with tab1:
        st.subheader("Frekuensi Genre")
        genre_counts = all_genres_series.value_counts()

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                genre_counts.sort_values(),
                orientation="h",
                labels={"value": "Jumlah Lagu", "index": "Genre"},
                color=genre_counts.sort_values().values,
                color_continuous_scale="Blues",
                title="Frekuensi Tiap Genre"
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            top8 = genre_counts.head(8)
            others = genre_counts[8:].sum()
            pie_data = pd.concat([top8, pd.Series({"Others": others})])
            fig = px.pie(
                values=pie_data.values,
                names=pie_data.index,
                title="Proporsi Genre (Top 8 + Others)",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)

        st.info(f"""
        📌 **Insight:** Genre terbanyak adalah **{genre_counts.index[0]}** 
        ({genre_counts.iloc[0]:,} lagu) dan **{genre_counts.index[1]}** 
        ({genre_counts.iloc[1]:,} lagu). Distribusi sangat tidak seimbang — 
        dua genre teratas mendominasi lebih dari separuh dataset.
        """)

    with tab2:
        st.subheader("Distribusi Lagu per Dekade")
        decade_counts = df_clean["decade"].value_counts().sort_index()

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                decade_counts,
                labels={"value": "Jumlah Lagu", "index": "Dekade"},
                color=decade_counts.values,
                color_continuous_scale="Oranges",
                title="Jumlah Lagu per Dekade",
                text=decade_counts.values
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.line(
                x=decade_counts.index,
                y=decade_counts.values,
                markers=True,
                labels={"x": "Dekade", "y": "Jumlah Lagu"},
                title="Tren Jumlah Lagu per Dekade",
                color_discrete_sequence=["coral"]
            )
            fig.update_traces(fill="tozeroy", fillcolor="rgba(255,127,80,0.2)")
            st.plotly_chart(fig, use_container_width=True)

        dominant = decade_counts.idxmax()
        pct = decade_counts.max() / decade_counts.sum() * 100
        st.warning(f"""
        ⚠️ **Bias Dekade Terdeteksi!**
        Dekade **{dominant}** mendominasi dengan **{decade_counts.max():,} lagu ({pct:.1f}%)** 
        dari total data. Ini akan menyebabkan sistem cenderung merekomendasikan 
        lagu era {dominant} meskipun bukan yang paling relevan secara konten.
        """)

    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Jumlah Genre per Lagu")
            genre_per_song = df_clean["num_genres"].value_counts().sort_index()
            fig = px.bar(
                genre_per_song,
                labels={"value": "Jumlah Lagu", "index": "Jumlah Genre"},
                color=genre_per_song.values,
                color_continuous_scale="Purples",
                title="Distribusi Jumlah Genre per Lagu",
                text=genre_per_song.values
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Top 10 Pasangan Genre")
            pairs = []
            for g1 in unique_genres:
                for g2 in unique_genres:
                    if g1 < g2 and cooccurrence.loc[g1, g2] > 0:
                        pairs.append({
                            "Genre 1": g1,
                            "Genre 2": g2,
                            "Co-occurrence": int(cooccurrence.loc[g1, g2])
                        })
            pairs_df = pd.DataFrame(pairs).sort_values(
                "Co-occurrence", ascending=False
            ).head(10)
            st.dataframe(pairs_df, use_container_width=True)

        st.subheader("Heatmap Co-occurrence Genre")
        fig = px.imshow(
            cooccurrence,
            color_continuous_scale="Blues",
            title="Co-occurrence Antar Genre",
            text_auto=True
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.info("📌 Semakin gelap = semakin sering dua genre muncul bersama dalam satu lagu")

    with tab4:
        st.subheader("Tren Genre per Dekade")

        df_exploded = df_clean.copy()
        df_exploded = df_exploded.explode("genres_list")
        df_exploded = df_exploded[df_exploded["genres_list"].isin(unique_genres)]

        genre_decade = df_exploded.groupby(
            ["decade", "genres_list"]
        ).size().unstack(fill_value=0)

        top_genres_list = all_genres_series.value_counts().head(8).index.tolist()
        genre_decade_top = genre_decade[
            [g for g in top_genres_list if g in genre_decade.columns]
        ]

        fig = px.imshow(
            genre_decade_top.T,
            color_continuous_scale="YlOrRd",
            title="Heatmap Genre per Dekade",
            text_auto=True,
            labels={"x": "Dekade", "y": "Genre", "color": "Jumlah Lagu"}
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        genre_decade_pct = genre_decade_top.div(
            genre_decade_top.sum(axis=1), axis=0
        ) * 100

        fig2 = px.bar(
            genre_decade_pct.reset_index(),
            x="decade",
            y=genre_decade_pct.columns.tolist(),
            title="Proporsi Genre per Dekade (%)",
            labels={"decade": "Dekade", "value": "Proporsi (%)"},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig2.update_layout(barmode="stack", legend_title="Genre")
        st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────
# HALAMAN 3: SISTEM REKOMENDASI
# ─────────────────────────────────────────
elif page == "🤖 Sistem Rekomendasi":
    st.title("🤖 Sistem Rekomendasi Musik")
    st.markdown("---")

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        song_list = sorted(df_clean["title"].unique().tolist())
        selected_song = st.selectbox("🎵 Pilih Lagu", song_list)
    with col2:
        top_n = st.slider("Jumlah Rekomendasi", 3, 10, 5)
    with col3:
        method = st.radio("Metode", ["Hybrid", "Content-Based", "Pseudo-CF"], index=0)

    st.markdown("---")

    # Info lagu yang dipilih
    song_info = df_clean[df_clean["title"] == selected_song].iloc[0]
    col1, col2, col3 = st.columns(3)
    col1.info(f"🎵 **{selected_song}**")
    col2.info(f"🎸 {song_info['genres']}")
    col3.info(f"📅 {song_info['year']} ({song_info['decade']})")

    if st.button("🚀 Cari Rekomendasi", use_container_width=True):
        if method == "Hybrid":
            result = hybrid_recommend(selected_song, df_clean, cb_sim_df, genre_rules, top_n)
            score_col = "Hybrid Score"
            color = "🟡"
            desc = "Hybrid (50% Content-Based + 50% Pseudo-CF)"
        elif method == "Content-Based":
            result = cb_recommend(selected_song, df_clean, cb_sim_df, top_n)
            score_col = "CB Score"
            color = "🔵"
            desc = "Content-Based (Kemiripan Genre + Tahun)"
        else:
            result = pseudo_cf_recommend(selected_song, df_clean, genre_rules, top_n)
            score_col = "CF Score"
            color = "🟢"
            desc = "Pseudo-CF (Co-occurrence Genre)"

        st.subheader(f"{color} Rekomendasi untuk: **{selected_song}**")
        st.caption(f"Metode: {desc}")

        if len(result) == 0:
            st.warning("⚠️ Tidak ada rekomendasi. Coba metode lain.")
        else:
            for i, row in result.iterrows():
                col1, col2, col3 = st.columns([3, 2, 1])
                col1.markdown(f"**{i+1}. {row['Judul']}**")
                col2.caption(f"🎸 {row['Genre']} | 📅 {row['Tahun']}")
                col3.progress(float(min(row[score_col], 1.0)))

        st.markdown("---")
        st.subheader("📊 Perbandingan Semua Metode")

        res_cb = cb_recommend(selected_song, df_clean, cb_sim_df, top_n)
        res_cf = pseudo_cf_recommend(selected_song, df_clean, genre_rules, top_n)
        res_hybrid = hybrid_recommend(selected_song, df_clean, cb_sim_df, genre_rules, top_n)

        def pad(lst, n):
            return lst + ["-"] * (n - len(lst))

        comparison = pd.DataFrame({
            "Rank": range(1, top_n + 1),
            "🔵 Content-Based": pad(res_cb["Judul"].tolist(), top_n),
            "🟢 Pseudo-CF": pad(res_cf["Judul"].tolist(), top_n),
            "🟡 Hybrid": pad(res_hybrid["Judul"].tolist(), top_n)
        }).set_index("Rank")

        st.dataframe(comparison, use_container_width=True)

# ─────────────────────────────────────────
# HALAMAN 4: ANALISIS GENRE RULES
# ─────────────────────────────────────────
elif page == "🔬 Analisis Genre Rules":
    st.title("🔬 Analisis Genre Association Rules")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    min_sup = col1.slider("Min Support", 0.01, 0.5, 0.05, 0.01)
    min_conf = col2.slider("Min Confidence", 0.1, 1.0, 0.3, 0.05)
    min_lift = col3.slider("Min Lift", 1.0, 10.0, 1.0, 0.5)

    filtered = genre_rules[
        (genre_rules["support"] >= min_sup) &
        (genre_rules["confidence"] >= min_conf) &
        (genre_rules["lift"] >= min_lift)
    ].copy()

    filtered["antecedents"] = filtered["antecedents"].apply(lambda x: ", ".join(list(x)))
    filtered["consequents"] = filtered["consequents"].apply(lambda x: ", ".join(list(x)))

    st.metric("Rules yang memenuhi filter", len(filtered))
    st.dataframe(
        filtered[["antecedents","consequents","support","confidence","lift"]]
        .round(4).reset_index(drop=True),
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("📈 Scatter Plot Metrik")
    fig = px.scatter(
        filtered,
        x="support", y="confidence",
        color="lift", size="lift",
        hover_data=["antecedents","consequents"],
        color_continuous_scale="YlOrRd",
        title="Support vs Confidence (warna & ukuran = Lift)"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🏆 Top 10 Genre Rules Terkuat")
    top10 = genre_rules.head(10).copy()
    top10["antecedents"] = top10["antecedents"].apply(lambda x: ", ".join(list(x)))
    top10["consequents"] = top10["consequents"].apply(lambda x: ", ".join(list(x)))

    for i, row in top10.iterrows():
        with st.expander(
            f"#{i+1} — {row['antecedents']} → {row['consequents']} | Lift: {row['lift']:.2f}"
        ):
            col1, col2, col3 = st.columns(3)
            col1.metric("Support", f"{row['support']:.4f}")
            col2.metric("Confidence", f"{row['confidence']:.4f}")
            col3.metric("Lift", f"{row['lift']:.4f}")
            st.caption(
                f"Artinya: {row['confidence']*100:.1f}% lagu bergenre "
                f"**{row['antecedents']}** juga bergenre **{row['consequents']}**, "
                f"dan hubungan ini {row['lift']:.1f}x lebih kuat dari kebetulan."
            )

# ─────────────────────────────────────────
# HALAMAN 5: EVALUASI
# ─────────────────────────────────────────
elif page == "📈 Evaluasi Metode":
    st.title("📈 Evaluasi & Perbandingan Metode")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Coverage",
        "🎯 Diversity",
        "📅 Decade Bias",
        "🔄 Side-by-Side"
    ])

    with tab1:
        st.subheader("Coverage per Metode")

        cb_covered = len(df_clean)
        cf_covered = len(df_clean[df_clean["genres_list"].apply(
            lambda x: any(
                g in genre_rules["antecedents"].apply(
                    lambda a: list(a)[0]
                ).tolist() for g in x
            )
        )])
        hybrid_covered = len(df_clean)
        total = len(df_clean)

        coverage_df = pd.DataFrame({
            "Metode": ["Content-Based", "Pseudo-CF", "Hybrid"],
            "Lagu Tercovered": [cb_covered, cf_covered, hybrid_covered],
            "Coverage (%)": [
                cb_covered/total*100,
                cf_covered/total*100,
                hybrid_covered/total*100
            ]
        })

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                coverage_df, x="Metode", y="Coverage (%)",
                color="Metode", text="Coverage (%)",
                color_discrete_sequence=["steelblue","mediumseagreen","coral"],
                title="Coverage per Metode"
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.dataframe(coverage_df, use_container_width=True)
            st.info(f"""
            📌 **Insight:**
            - CB dan Hybrid cover **100%** lagu
            - Pseudo-CF cover **{cf_covered/total*100:.1f}%** lagu
            - {total - cf_covered} lagu tidak tercover CF karena genre-nya tidak masuk rules
            """)

    with tab2:
        st.subheader("Diversity Score per Metode")
        st.caption("Mengukur keberagaman genre dari hasil rekomendasi")

        def diversity_score(recs, genre_col="Genre"):
            if len(recs) == 0:
                return 0
            all_g = recs[genre_col].str.split("|").explode()
            return round(all_g.nunique() / len(all_g), 4)

        sample_songs = df_clean["title"].sample(10, random_state=42).tolist()
        div_results = []
        for song in sample_songs:
            rec_cb = cb_recommend(song, df_clean, cb_sim_df)
            rec_cf = pseudo_cf_recommend(song, df_clean, genre_rules)
            rec_h = hybrid_recommend(song, df_clean, cb_sim_df, genre_rules)
            div_results.append({
                "Lagu": song[:35],
                "CB": diversity_score(rec_cb),
                "Pseudo-CF": diversity_score(rec_cf),
                "Hybrid": diversity_score(rec_h)
            })

        div_df = pd.DataFrame(div_results)
        st.dataframe(div_df, use_container_width=True)

        avg_div = {
            "Content-Based": div_df["CB"].mean(),
            "Pseudo-CF": div_df["Pseudo-CF"].mean(),
            "Hybrid": div_df["Hybrid"].mean()
        }

        fig = px.bar(
            x=list(avg_div.keys()),
            y=list(avg_div.values()),
            color=list(avg_div.keys()),
            text=[f"{v:.4f}" for v in avg_div.values()],
            color_discrete_sequence=["steelblue","mediumseagreen","coral"],
            title="Rata-rata Diversity Score per Metode",
            labels={"x": "Metode", "y": "Diversity Score"}
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.info("""
        📌 **Interpretasi Diversity Score:**
        - Mendekati **1.0** = setiap rekomendasi genre-nya berbeda (sangat beragam)
        - Mendekati **0.0** = semua rekomendasi genre-nya sama (monoton)
        - Hybrid idealnya lebih beragam dari CB murni
        """)

    with tab3:
        st.subheader("Analisis Bias Dekade")

        decade_data = df_clean["decade"].value_counts().sort_index()

        rec_decades = []
        sample = df_clean["title"].sample(20, random_state=42).tolist()
        for song in sample:
            recs = hybrid_recommend(song, df_clean, cb_sim_df, genre_rules, 5)
            if len(recs) > 0 and "Tahun" in recs.columns:
                for year in recs["Tahun"]:
                    rec_decades.append(str((int(year) // 10) * 10) + "s")

        rec_decade_counts = pd.Series(rec_decades).value_counts().sort_index()

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                decade_data,
                title="Distribusi Dekade — Data Asli",
                labels={"value": "Jumlah Lagu", "index": "Dekade"},
                color_discrete_sequence=["steelblue"]
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                rec_decade_counts,
                title="Distribusi Dekade — Hasil Rekomendasi",
                labels={"value": "Frekuensi", "index": "Dekade"},
                color_discrete_sequence=["coral"]
            )
            st.plotly_chart(fig, use_container_width=True)

        if len(rec_decade_counts) > 0:
            dom = rec_decade_counts.idxmax()
            pct = rec_decade_counts.max() / rec_decade_counts.sum() * 100
            if dom == decade_data.idxmax():
                st.error(f"""
                ⚠️ **Bias Terkonfirmasi!**
                Rekomendasi didominasi lagu era **{dom}** ({pct:.1f}%) —
                sama dengan dominasi di data asli. Sistem merekomendasikan
                lagu {dom} bukan karena paling relevan, tapi karena paling banyak di data.
                """)
            else:
                st.success("✅ Bias dekade relatif terkontrol!")

    with tab4:
        st.subheader("Perbandingan Side-by-Side")
        eval_song = st.selectbox(
            "Pilih lagu untuk evaluasi",
            sorted(df_clean["title"].unique().tolist()),
            key="eval_song"
        )
        top_eval = st.slider("Top N", 3, 10, 5, key="eval_n")

        res_cb = cb_recommend(eval_song, df_clean, cb_sim_df, top_eval)
        res_cf = pseudo_cf_recommend(eval_song, df_clean, genre_rules, top_eval)
        res_h = hybrid_recommend(eval_song, df_clean, cb_sim_df, genre_rules, top_eval)

        def pad(lst, n):
            return lst + ["-"] * (n - len(lst))

        comp = pd.DataFrame({
            "Rank": range(1, top_eval + 1),
            "🔵 Content-Based": pad(res_cb["Judul"].tolist(), top_eval),
            "🟢 Pseudo-CF": pad(res_cf["Judul"].tolist(), top_eval),
            "🟡 Hybrid": pad(res_h["Judul"].tolist(), top_eval)
        }).set_index("Rank")

        st.dataframe(comp, use_container_width=True)

# ─────────────────────────────────────────
# HALAMAN 6: IMPLIKASI MANAJERIAL
# ─────────────────────────────────────────
elif page == "💼 Implikasi Manajerial":
    st.title("💼 Implikasi Manajerial")
    st.markdown("---")

    st.subheader("🔍 Temuan Utama")
    col1, col2, col3 = st.columns(3)

    with col1:
        top_pair_g1, top_pair_g2 = "", ""
        max_cooc = 0
        for g1 in unique_genres:
            for g2 in unique_genres:
                if g1 < g2 and cooccurrence.loc[g1, g2] > max_cooc:
                    max_cooc = cooccurrence.loc[g1, g2]
                    top_pair_g1, top_pair_g2 = g1, g2

        st.success(f"""
        **🎸 Pola Genre Kuat**
        - {len(genre_rules)} genre rules ditemukan
        - Pasangan genre terkuat:
          **{top_pair_g1} + {top_pair_g2}**
          ({max_cooc:,} lagu)
        - Lift tertinggi: **{genre_rules["lift"].max():.2f}x**
        """)

    with col2:
        dom_decade = df_clean["decade"].value_counts().idxmax()
        dom_pct = df_clean["decade"].value_counts().max() / len(df_clean) * 100
        st.warning(f"""
        **📅 Bias Dekade**
        - **{dom_pct:.1f}%** data dari era **{dom_decade}**
        - Rekomendasi cenderung condong ke lagu {dom_decade}
        - Perlu penambahan data era lain untuk balance
        """)

    with col3:
        st.error(f"""
        **⚙️ Keterbatasan Data**
        - Hanya 3 kolom (title, genres, year)
        - Tidak ada audio features
        - Tidak ada data interaksi user
        - Sistem jauh lebih baik dengan Spotify API
        """)

    st.markdown("---")
    st.subheader("🎯 Rekomendasi Aksi")

    aksi = pd.DataFrame({
        "Temuan": [
            "Genre co-occurrence kuat",
            "Bias data ke 1990s",
            "Tidak ada audio features",
            "Tidak ada data user",
            "Cold-start problem"
        ],
        "Aksi yang Disarankan": [
            "Buat fitur playlist otomatis lintas genre yang berkaitan",
            "Tambah data lagu era 1920s–1980s dan 2000s–2020s",
            "Integrasikan Spotify API untuk tambah fitur audio",
            "Tambahkan sistem rating atau play history user",
            "Buat onboarding pilih genre favorit untuk user baru"
        ],
        "Prioritas": [
            "🔴 Tinggi",
            "🔴 Tinggi",
            "🟡 Sedang",
            "🟡 Sedang",
            "🟢 Rendah"
        ]
    })

    st.dataframe(aksi, use_container_width=True)

    st.markdown("---")
    st.subheader("🚀 Roadmap Pengembangan")

    col1, col2, col3 = st.columns(3)
    col1.info("""
    **Fase 1 — Jangka Pendek**
    - Tambah data dari era lain
    - Integrasikan Spotify API
    - Tambah fitur search by genre
    """)
    col2.warning("""
    **Fase 2 — Jangka Menengah**
    - Kumpulkan data interaksi user
    - Implementasi CF berbasis user
    - A/B testing sistem rekomendasi
    """)
    col3.error("""
    **Fase 3 — Jangka Panjang**
    - Deep learning audio analysis
    - Real-time recommendation
    - Personalisasi mood-based
    """)
