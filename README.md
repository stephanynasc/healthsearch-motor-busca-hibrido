# 🩺 HealthSearch — Motor de Busca Híbrido (BM25 + Semântico + RRF)

Protótipo do Desafio Integrador (UNIPÊ — Tendências em Ciência da Computação).

## Rodando localmente

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run healthsearch_app.py
```

## Publicando no GitHub

```bash
git init
git add .
git commit -m "HealthSearch: BM25 + busca semântica + RRF + bônus cross-encoder"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/healthsearch.git
git push -u origin main
```

## Deploy no Streamlit Community Cloud

1. Acesse https://share.streamlit.io e faça login com sua conta GitHub.
2. Clique em **"New app"**.
3. Selecione o repositório, a branch `main` e o arquivo `healthsearch_app.py`.
4. Clique em **Deploy** — o Streamlit Cloud instala automaticamente o
   `requirements.txt` e publica a URL pública do app.

## Desafio Bônus

Marque a checkbox **"Desafio Bônus: Re-ranking com Cross-Encoder"** na
barra lateral para aplicar `cross-encoder/ms-marco-MiniLM-L-6-v2` sobre o
Top-3 do ranking híbrido RRF e comparar a nota antes/depois na aba
"Híbrido RRF".
