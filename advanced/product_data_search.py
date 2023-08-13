import openai, os, backoff
import pandas as pd
from openai.embeddings_utils import get_embedding, cosine_similarity

openai.api_key = os.getenv("OPENAI_API_KEY")
COMPLETION_MODEL = "text-davinci-003"
embedding_model = "text-embedding-ada-002"

df = pd.read_parquet("data/taobao_product_title.parquet")


# 基于embedding的语义搜索
def search_product(df, query, n=3, pprint=True):
    product_embedding = get_embedding(
        query,
        engine=embedding_model
    )
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        .product_name
    )
    if pprint:
        for r in results:
            print(r)
    return results


results = search_product(df, "自然淡雅背包", n=3)


# 冷启动阶段的推荐
def recommend_product(df, product_name, n=3, pprint=True):
    product_embedding = df[df['product_name'] == product_name].iloc[0].embedding
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        .product_name
    )
    if pprint:
        for r in results:
            print(r)
    return results


results = recommend_product(df, "【热销】华为 MatePad Pro 10", n=3)
