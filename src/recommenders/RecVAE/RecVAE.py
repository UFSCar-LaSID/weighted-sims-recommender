
# Code adapted from: https://github.com/ilya-shenbin/RecVAE/blob/master/run.py

import numpy as np
import torch
from torch import optim
import bottleneck as bn
from src.recommenders.RecVAE.model import VAE
import src as kw
import pandas as pd

import os
import pickle

from src.recommenders.mf import SparseRepr



def generate(batch_size, device, data_in, data_out=None, shuffle=False, samples_perc_per_epoch=1):
    assert 0 < samples_perc_per_epoch <= 1
    
    total_samples = data_in.shape[0]
    samples_per_epoch = int(total_samples * samples_perc_per_epoch)
    
    if shuffle:
        idxlist = np.arange(total_samples)
        np.random.shuffle(idxlist)
        idxlist = idxlist[:samples_per_epoch]
    else:
        idxlist = np.arange(samples_per_epoch)
    
    for st_idx in range(0, samples_per_epoch, batch_size):
        end_idx = min(st_idx + batch_size, samples_per_epoch)
        idx = idxlist[st_idx:end_idx]

        yield Batch(device, idx, data_in, data_out)


class Batch:
    def __init__(self, device, idx, data_in, data_out=None):
        self._device = device
        self._idx = idx
        self._data_in = data_in
        self._data_out = data_out
    
    def get_idx(self):
        return self._idx
    
    def get_idx_to_dev(self):
        return torch.LongTensor(self.get_idx()).to(self._device)
        
    def get_ratings(self, is_out=False):
        data = self._data_out if is_out else self._data_in
        return data[self._idx]
    
    def get_ratings_to_dev(self, is_out=False):
        return torch.Tensor(
            self.get_ratings(is_out).toarray()
        ).to(self._device)





class RecVAE(object):

    def __init__(self, gamma, embeddings_filepath):
        self.device = 'cuda:0' if kw.TRAIN_MODE == 'gpu' else kw.TRAIN_MODE
        self.hidden_dim = 600
        self.latent_dim = 200
        self.batch_size = 5000
        self.beta = None
        self.gamma = gamma
        self.lr = 5e-4
        self.n_enc_epochs = 3
        self.n_dec_epochs = 1
        self.n_epochs = 50
        self.embeddings_filepath = embeddings_filepath

    def __run(self, model, n_epochs, opts, train_data, dropout_rate):
        model.train()
        for _ in range(n_epochs):
            generator = generate(batch_size=self.batch_size, device=self.device, data_in=train_data, shuffle=True)
            for batch in generator:
                ratings = batch.get_ratings_to_dev()

                for optimizer in opts:
                    optimizer.zero_grad()
                    
                _, loss = model(ratings, beta=self.beta, gamma=self.gamma, dropout_rate=dropout_rate)
                loss.backward()
                
                for optimizer in opts:
                    optimizer.step()
    
    def _save_embeddings(self, train_data):
        self.model.eval()

        item_embeddings = self.model.get_items_embeddings()

        user_embeddings = []
        generator = generate(batch_size=self.batch_size, device=self.device, data_in=train_data, shuffle=False)
        for batch in generator:
            ratings = batch.get_ratings_to_dev()
            user_embeddings += self.model.get_user_embeddings(ratings, 0).tolist()

        np.save(os.path.join(self.embeddings_filepath, kw.FILE_ITEMS_EMBEDDINGS), item_embeddings)
        np.save(os.path.join(self.embeddings_filepath, kw.FILE_USERS_EMBEDDINGS), user_embeddings)
        pickle.dump(self.sparse_repr, open(os.path.join(self.embeddings_filepath, kw.FILE_SPARSE_REPR), 'wb'))

    def fit(self, raw_df_train):
        self.sparse_repr = SparseRepr(raw_df_train)
        self.train_data = self.sparse_repr.get_user_items_matrix(raw_df_train)

        model_kwargs = {
            'hidden_dim': self.hidden_dim,
            'latent_dim': self.latent_dim,
            'input_dim': self.train_data.shape[1]
        }

        self.model = VAE(**model_kwargs).to(self.device)

        decoder_params = set(self.model.decoder.parameters())
        encoder_params = set(self.model.encoder.parameters())

        optimizer_encoder = optim.Adam(encoder_params, lr=self.lr)
        optimizer_decoder = optim.Adam(decoder_params, lr=self.lr)

        for i in range(self.n_epochs):

            self.__run(opts=[optimizer_encoder], n_epochs=self.n_enc_epochs, dropout_rate=0.5, model=self.model, train_data=self.train_data)
            self.model.update_prior()
            self.__run(opts=[optimizer_decoder], n_epochs=self.n_dec_epochs, dropout_rate=0, model=self.model, train_data=self.train_data)
        
        if not os.listdir(self.embeddings_filepath):
            self._save_embeddings(self.train_data)



    def recommend(self, raw_df_test):
        test_indexes = self.sparse_repr.get_user_index(raw_df_test[kw.COLUMN_USER_ID].unique())

        self.model.eval()

        recommended_items = []

        for batch in generate(batch_size=self.batch_size, device=self.device, data_in=self.train_data[test_indexes], samples_perc_per_epoch=1):
            ratings_in = batch.get_ratings_to_dev()

            ratings_pred = self.model(ratings_in, calculate_loss=False, dropout_rate=0).cpu().detach().numpy()
            
            ratings_pred[batch.get_ratings().nonzero()] = -np.inf
            
            batch_users = ratings_pred.shape[0]
            idx_topk_part = bn.argpartition(-ratings_pred, kw.TOP_N, axis=1)
            topk_part = ratings_pred[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :kw.TOP_N]]
            idx_part = np.argsort(-topk_part, axis=1)
            idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

            recommended_items += self.sparse_repr.get_item_id(idx_topk.flatten()).tolist()

        recommendations = pd.DataFrame(
            np.vstack([
                np.repeat(self.sparse_repr.get_user_id(test_indexes), kw.TOP_N), 
                recommended_items
            ]).T,
            columns=[kw.COLUMN_USER_ID, kw.COLUMN_ITEM_ID]
        )
        return recommendations

