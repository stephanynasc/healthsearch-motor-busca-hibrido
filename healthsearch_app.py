# -*- coding: utf-8 -*-
"""
HealthSearch — Motor de Busca Híbrido (BM25 + Busca Semântica Vetorial + RRF)
Desafio Integrador — UNIPÊ — Tendências em Ciência da Computação (RI/PLN)

Como rodar localmente:
    pip install -r requirements.txt
    streamlit run healthsearch_app.py
"""

import re
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Modelos pesados (sentence-transformers / cross-encoder) são carregados sob
# demanda e cacheados, para que a interface abra rápido mesmo sem GPU.
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from sentence_transformers.util import cos_sim
    SEMANTIC_BACKEND = "sentence-transformers"
except ImportError:
    SEMANTIC_BACKEND = "simulado"


# =============================================================================
# FASE 1 — INGESTÃO DO CORPUS MÉDICO E PRÉ-PROCESSAMENTO
# =============================================================================

CORPUS = [
    {
        "id": "Doc 1",
        "titulo": "Protocolo Emergência ECG",
        "texto": (
            "Pacientes com dor precordial aguda e suspeita de síndrome "
            "coronariana devem realizar eletrocardiograma CÓD-ECG-12D em "
            "até 10 minutos."
        ),
    },
    {
        "id": "Doc 2",
        "titulo": "Guia de Farmacologia Cardíaca",
        "texto": (
            "O uso imediato de ácido acetilsalicílico e antiagregantes "
            "plaquetários reduz a mortalidade no infarto agudo do miocárdio."
        ),
    },
    {
        "id": "Doc 3",
        "titulo": "Diretriz de Hipertensão Arterial",
        "texto": (
            "A crise hipertensiva severa requer administração de "
            "anti-hipertensivos venosos e monitoramento contínuo da pressão "
            "arterial na UTI."
        ),
    },
    {
        "id": "Doc 4",
        "titulo": "Manual de AVC Isquêmico",
        "texto": (
            "O acidente vascular cerebral isquêmico agudo deve ser tratado "
            "com trombolíticos venosos em até quatro horas e meia do início "
            "dos sintomas."
        ),
    },
    {
        "id": "Doc 5",
        "titulo": "Protocolo de Reanimação RCR",
        "texto": (
            "Parada cardiorrespiratória em adultos exige compressões "
            "torácicas contínuas de alta qualidade e desfibrilação precoce "
            "no código azul."
        ),
    },
    {
        "id": "Doc 6",
        "titulo": "Procedimentos de UTI Geral",
        "texto": (
            "Para diagnóstico do protocolo CÓD-ECG-12D em arritmias "
            "complexas, recomenda-se a monitorização cardíaca contínua por "
            "telemetria."
        ),
    },
]

# Lista de stopwords em português (compacta, sem dependência externa como nltk)
STOPWORDS_PT = set("""
a ao aos aquela aquelas aquele aqueles aquilo as até com como da das de dela
delas dele deles depois do dos e ela elas ele eles em entre era eram essa
essas esse esses esta estas este estes eu foi fomos for foram fosse fossem
fui há isso isto já lhe lhes mais mas me mesmo meu meus minha minhas muito
na nas nem no nos nossa nossas nosso nossos num numa não o os ou para pela
pelas pelo pelos por qual quando que quem se seu seus só sua suas também
te tem tém temos tenho teu teus tinha tive tu tua tuas um uma você vocês
""".split())


def preprocess(text: str) -> list[str]:
    """Normaliza para minúsculas, remove acentos/caracteres especiais,
    tokeniza e remove stopwords em português."""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS_PT and len(t) > 1]


@st.cache_data(show_spinner=False)
def build_tokenized_corpus():
    return [preprocess(doc["texto"]) for doc in CORPUS]


# =============================================================================
# FASE 2 — MOTOR LÉXICO (BM25) COM PARÂMETROS INTERATIVOS
# =============================================================================

def run_bm25(query: str, k1: float, b: float):
    tokenized_corpus = build_tokenized_corpus()
    bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
    tokenized_query = preprocess(query)
    scores = bm25.get_scores(tokenized_query)
    return scores


# =============================================================================
# FASE 3 — MOTOR SEMÂNTICO VETORIAL
# =============================================================================

@st.cache_resource(show_spinner="Carregando modelo de embeddings...")
def load_embedding_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


@st.cache_data(show_spinner=False)
def corpus_embeddings(_model_flag: str):
    """_model_flag apenas invalida o cache se trocarmos de backend."""
    if SEMANTIC_BACKEND == "sentence-transformers":
        model = load_embedding_model()
        textos = [doc["texto"] for doc in CORPUS]
        return model.encode(textos, convert_to_numpy=True, normalize_embeddings=True)
    return None


def simulated_semantic_scores(query: str):
    """Fallback documentado: caso sentence-transformers não esteja disponível,
    simula uma busca semântica via similaridade de Jaccard sobre um pequeno
    dicionário de sinônimos médicos, apenas para manter a aplicação
    funcional offline. Isso NÃO substitui embeddings reais e serve só como
    salvaguarda de execução."""
    synonyms = {
        "infarto": {"sindrome", "coronariana", "miocardio", "cardiaca", "cardiaco"},
        "ataque": {"sindrome", "coronariana", "miocardio"},
        "cardiaco": {"cardiaca", "miocardio", "coronariana"},
        "avc": {"vascular", "cerebral", "isquemico"},
        "derrame": {"vascular", "cerebral", "isquemico"},
        "pressao": {"hipertensiva", "hipertensao", "hipertensivos"},
    }
    q_tokens = set(preprocess(query))
    expanded = set(q_tokens)
    for t in q_tokens:
        expanded |= synonyms.get(t, set())

    scores = []
    for tokens in build_tokenized_corpus():
        d_tokens = set(tokens)
        inter = len(expanded & d_tokens)
        union = len(expanded | d_tokens) or 1
        scores.append(inter / union)
    return np.array(scores)


def run_semantic(query: str):
    if SEMANTIC_BACKEND == "sentence-transformers":
        model = load_embedding_model()
        doc_emb = corpus_embeddings("st-multilingual")
        query_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        sims = cos_sim(query_emb, doc_emb).numpy().flatten()
        return sims
    return simulated_semantic_scores(query)


# =============================================================================
# FASE 4 — ALGORITMO DE FUSÃO RRF (RECIPROCAL RANK FUSION)
# =============================================================================

def scores_to_ranks(scores: np.ndarray) -> np.ndarray:
    """Converte scores em ranks (1 = melhor)."""
    order = np.argsort(-scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def rrf_fusion(bm25_scores: np.ndarray, semantic_scores: np.ndarray,
               alpha: float, k_rrf: int = 60) -> np.ndarray:
    """Score_RRF(D) = alpha * [1/(k+Rank_BM25)] + (1-alpha) * [1/(k+Rank_Semantico)]"""
    rank_bm25 = scores_to_ranks(bm25_scores)
    rank_sem = scores_to_ranks(semantic_scores)
    rrf = alpha * (1.0 / (k_rrf + rank_bm25)) + (1 - alpha) * (1.0 / (k_rrf + rank_sem))
    return rrf, rank_bm25, rank_sem


# =============================================================================
# DESAFIO BÔNUS — CROSS-ENCODER RE-RANKING
# =============================================================================

@st.cache_resource(show_spinner="Carregando Cross-Encoder...")
def load_cross_encoder():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def cross_encoder_rerank(query: str, top_docs: list[dict]):
    ce = load_cross_encoder()
    pairs = [[query, d["texto"]] for d in top_docs]
    scores = ce.predict(pairs)
    return scores


# =============================================================================
# INTERFACE STREAMLIT
# =============================================================================

st.set_page_config(page_title="HealthSearch", page_icon="🩺", layout="wide")

st.title("🩺 HealthSearch — Motor de Busca Híbrido")
st.caption(
    "BM25 (léxico) + Busca Semântica Vetorial (embeddings) unificados via "
    "Reciprocal Rank Fusion (RRF)."
)

if SEMANTIC_BACKEND != "sentence-transformers":
    st.warning(
        "⚠️ `sentence-transformers` não está instalado neste ambiente. "
        "A busca semântica está usando um modo simulado documentado "
        "(similaridade de Jaccard com expansão de sinônimos médicos), "
        "apenas para manter a demonstração funcional. Instale "
        "`sentence-transformers` para embeddings reais."
    )

with st.sidebar:
    st.header("⚙️ Parâmetros de Calibração")

    st.subheader("BM25 (Léxico)")
    k1 = st.slider("k₁ — Saturação de Frequência", 0.0, 3.0, 1.2, 0.1)
    b = st.slider("b — Normalização por Comprimento", 0.0, 1.0, 0.75, 0.05)

    st.subheader("Fusão RRF")
    alpha = st.slider("α — Peso BM25 ↔ Semântico", 0.0, 1.0, 0.5, 0.05,
                       help="α=1 → 100% léxico | α=0 → 100% semântico")
    st.caption("k_RRF fixo em 60 (constante de suavização de posição).")

    st.divider()
    use_cross_encoder = st.checkbox(
        "🏆 Desafio Bônus: Re-ranking com Cross-Encoder (+0.3 pts)",
        value=False,
        help="Aplica cross-encoder/ms-marco-MiniLM-L-6-v2 sobre o Top-3 "
             "resultado da busca híbrida RRF.",
    )

st.divider()

query = st.text_input(
    "🔎 Digite sua consulta médica:",
    placeholder='Ex.: "infarto", "AAS 100mg", "CÓD-ECG-12D", "pressão alta"',
)

if query:
    bm25_scores = run_bm25(query, k1, b)
    semantic_scores = run_semantic(query)
    rrf_scores, rank_bm25, rank_sem = rrf_fusion(bm25_scores, semantic_scores, alpha)

    df = pd.DataFrame({
        "ID": [d["id"] for d in CORPUS],
        "Título": [d["titulo"] for d in CORPUS],
        "Trecho": [d["texto"] for d in CORPUS],
        "Score BM25": bm25_scores,
        "Rank BM25": rank_bm25,
        "Score Semântico": semantic_scores,
        "Rank Semântico": rank_sem,
        "Score RRF": rrf_scores,
    })

    tab_lex, tab_sem, tab_hib, tab_matriz = st.tabs(
        ["📖 Léxico (BM25)", "🧠 Semântico", "⚡ Híbrido RRF", "📊 Matriz Comparativa"]
    )

    with tab_lex:
        st.subheader("Ranking Léxico — BM25")
        df_lex = df.sort_values("Score BM25", ascending=False)
        st.dataframe(
            df_lex[["ID", "Título", "Score BM25", "Trecho"]],
            use_container_width=True, hide_index=True,
        )
        st.bar_chart(df_lex.set_index("ID")["Score BM25"])

    with tab_sem:
        st.subheader("Ranking Semântico — Similaridade de Cosseno")
        df_sem = df.sort_values("Score Semântico", ascending=False)
        st.dataframe(
            df_sem[["ID", "Título", "Score Semântico", "Trecho"]],
            use_container_width=True, hide_index=True,
        )
        st.bar_chart(df_sem.set_index("ID")["Score Semântico"])

    with tab_hib:
        st.subheader(f"Ranking Híbrido — RRF (α={alpha})")
        df_hib = df.sort_values("Score RRF", ascending=False).reset_index(drop=True)
        st.dataframe(
            df_hib[["ID", "Título", "Score RRF", "Rank BM25", "Rank Semântico", "Trecho"]],
            use_container_width=True, hide_index=True,
        )
        st.bar_chart(df_hib.set_index("ID")["Score RRF"])

        if use_cross_encoder:
            st.markdown("---")
            st.subheader("🏆 Re-ranking com Cross-Encoder (Top-3)")
            top3 = df_hib.head(3).to_dict("records")
            top3_docs = [{"texto": r["Trecho"]} for r in top3]
            with st.spinner("Aplicando Cross-Encoder sobre o Top-3..."):
                ce_scores = cross_encoder_rerank(query, top3_docs)

            df_ce = pd.DataFrame({
                "ID": [r["ID"] for r in top3],
                "Título": [r["Título"] for r in top3],
                "Score RRF (antes)": [r["Score RRF"] for r in top3],
                "Score Cross-Encoder (depois)": ce_scores,
            }).sort_values("Score Cross-Encoder (depois)", ascending=False)

            st.dataframe(df_ce, use_container_width=True, hide_index=True)
            st.caption(
                "Compare a ordem original do RRF com a nova ordem sugerida "
                "pelo Cross-Encoder — ele reavalia par (consulta, documento) "
                "de forma conjunta, geralmente mais precisa que a fusão por "
                "rank isolada."
            )

    with tab_matriz:
        st.subheader("Matriz Comparativa de Ranks")
        df_matriz = df[["ID", "Título", "Rank BM25", "Rank Semântico"]].copy()
        df_matriz["Rank RRF"] = df["Score RRF"].rank(ascending=False).astype(int)
        df_matriz = df_matriz.sort_values("Rank RRF")
        st.dataframe(df_matriz, use_container_width=True, hide_index=True)

        st.markdown("##### Comparação visual dos ranks por documento")
        chart_df = df_matriz.set_index("ID")[["Rank BM25", "Rank Semântico", "Rank RRF"]]
        st.line_chart(chart_df)

        st.info(
            "💡 Quanto **menor** o rank, melhor a posição do documento. "
            "Repare como termos técnicos exatos (ex.: CÓD-ECG-12D) tendem a "
            "vencer no BM25, enquanto sinônimos (ex.: 'infarto' → 'síndrome "
            "coronariana') dependem do motor semântico para aparecer bem "
            "posicionados — a fusão RRF equilibra os dois pontos cegos."
        )
else:
    st.info("👆 Digite uma consulta acima para ver os rankings Léxico, Semântico e Híbrido.")

st.divider()
with st.expander("📚 Sobre o corpus médico utilizado"):
    st.dataframe(pd.DataFrame(CORPUS), use_container_width=True, hide_index=True)
