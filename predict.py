import torch
import streamlit as st
# from model import MyEfficientNet
# from utils import transform_image
from model import TorchEASE
from dataloader import BaseDataset
import yaml
from typing import Tuple

import joblib

@st.cache
def load_model() -> TorchEASE:
    # model = joblib.load('./models/ease.pkl')
    model = torch.load('./models/ease.pt')
    return model

def get_prediction(model, user_id, dataset):
    user_id = user_id
    # 쿼리문 작성
    query = f"user == {user_id}"

    # 해당 유저가 본 영화의 데이터 프레임 (예측의 대상이 된다.)
    df_user = dataset.train_data.query(query)

    # 해당 유저가 어떤 영화를 봤을까?
    user_movie_list_seen = df_user['item'].to_list()

    # 해당 유저에게 k = 10 개의 영화를 추천해줄게
    output = model.predict_all(dataset.train_data, k = 10, genre_filter = True)

    solution = output.drop('user_id', axis=1).set_index('user')['predicted_items']
    user_movie_list_rec = solution[user_id]

    return user_movie_list_seen, user_movie_list_rec