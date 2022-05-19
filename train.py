from dataloader import BaseDataset
from model import TorchEASE
from utils import set_seed
import joblib

# set_seed
set_seed(seed=42)

# parameter setting
reg = 600
is_genre_filter = False
score_col=None
k = 10

# dataset setting
dataset = BaseDataset(path = '../data/', mode = 'train')

# model setting
model = TorchEASE(dataset.train_data, user_col='user', item_col='item', score_col=score_col, reg=reg, dataset = dataset)

# fit
model.fit()

# save model
joblib.dump(model,'./models/ease.pkl')


