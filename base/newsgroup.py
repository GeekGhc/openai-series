from sklearn.datasets import fetch_20newsgroups
from openai.embeddings_utils import get_embeddings
import openai, os, tiktoken, backoff
import pandas as pd


def twenty_newsgroup_to_csv():
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

    # T为对dataframe的转置
    df = pd.DataFrame([newsgroups_train.data, newsgroups_train.target.tolist()]).T
    df.columns = ['text', 'target']
    print(df)

    targets = pd.DataFrame(newsgroups_train.target_names, columns=['title'])

    out = pd.merge(df, targets, left_on='target', right_index=True)
    out.to_csv('20_newsgroup.csv', index=False)


twenty_newsgroup_to_csv()
