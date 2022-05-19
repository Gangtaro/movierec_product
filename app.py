import streamlit as st
import yaml
import io
from PIL import Image
from predict import load_model, get_prediction
import torch

from confirm_button_hack import cache_on_button_press

from model import TorchEASE
from dataloader import BaseDataset

from utils import set_seed
import joblib
import pandas as pd


# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout = "wide")

# st.write("Hello World")

def main():
    st.title("Movie Rec Model | test module")

    # with open("config.yaml") as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)

    # 영화 정보 업로드
    titles_df = pd.read_csv("./data/train/titles.tsv", sep="\t")
    
    # model = load_model()
    # @st.cache
    def load_model() -> TorchEASE:
        # model = joblib.load('./models/ease.pkl')
        model = torch.load('./models/ease.pt')
        return model

    # @st.cache
    def load_dataset() -> BaseDataset:
        model_dataset = torch.load('./models/dataset.pt')
        return model_dataset

    model = load_model()
    dataset = load_dataset()

    # 영화 정보와 아이템 인덱스 결합
    item2idx = dataset.item2idx.reset_index()
    item2idx.columns = ['item', 'item_idx']
    titles_df = titles_df.merge(item2idx, on ='item')

    # prediction
    output = model.predict_all(dataset.train_data, k = 10, genre_filter = True)
    solution = output.drop('user_id', axis=1).set_index('user')['predicted_items']

    # 유저 선택
    user_id = st.slider('Select user', 0, dataset.n_users, 0) 
    st.write("user_id(system) :", user_id)

    # 추천된 영화
    user_movie_list_rec = solution[user_id]

    # 유저가 본 영화
    query = f"user == {user_id}"
    df_user = dataset.train_data.query(query)
    user_movie_list_seen = df_user['item'].to_list()

    df_user_movie_list_rec = titles_df[titles_df['item_idx'].isin(user_movie_list_rec)]['title']
    df_user_movie_list_seen = titles_df[titles_df['item_idx'].isin(user_movie_list_seen)]['title']

    # st.write(f"user_movie_list_seen : {user_movie_list_seen}")
    # st.write(f"user_movie_list_rec :  {user_movie_list_rec}")

    # st.write(f"user_movie_list_seen : {df_user_movie_list_seen.head()}")
    # st.write(f"user_movie_list_rec :  {df_user_movie_list_rec.head()}")

    st.write('해당 유저가 본 영화 무작위 10개')
    st.dataframe(titles_df[titles_df['item_idx'].isin(user_movie_list_seen)]['title'].sample(10))

    st.write('해당 유저에게 추천된 영화 상위 10개')
    st.dataframe(titles_df[titles_df['item_idx'].isin(user_movie_list_rec)]['title'].head(10))


main()
