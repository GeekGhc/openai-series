import pandas as pd
import openai
import os
from IPython.display import display

openai.api_key = os.environ.get("OPENAI_API_KEY")
# list all open AI models

engines = openai.Engine.list()
pd = pd.DataFrame(openai.Engine.list()['data'])
display(pd[['id', 'owner']])
