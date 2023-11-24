from typing import Any, Iterator
import torch
import pandas as pd
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

class TCN_data(Dataset):
    def __init__(self, dict_data, timepoints, mode="train", T=63, H=3):
        super(TCN_data, self).__init__()
        self.fields = ['open', 'high', 'low', 'close', 'vwap', 'volume', 'total_turnover']
        # test_data = pd.read_csv(filename)
        # test_data = test_data.fillna(0.)
        self.data1 = dict_data
        # self.data1 = {f:test_data.pivot_table(index='datetime', columns='instrument', values=f) for f in self.fields} # 每一天 每一支股票的属性


        self.time_points = timepoints
        self.time_points.sort()

        train_set_point = len(self.time_points[self.time_points<'2020-01-01'])
        valid_set_point = len(self.time_points[(self.time_points>='2020-01-01') & 
                                        (self.time_points<'2021-01-01')])

        test_set_point = len(self.time_points[self.time_points>='2021-01-01'])
        assert train_set_point + valid_set_point + test_set_point == len(self.time_points)
        if mode == 'train':
            self.data_range = list(range(T, train_set_point))
        elif mode == 'valid':
            self.data_range = list(range(train_set_point, train_set_point + valid_set_point))
        else:
            self.data_range = list(range(train_set_point + valid_set_point, len(self.time_points) - H))
        # split_point = len(self.time_points[self.time_points<'2020-01-01'])
    def __len__(self):
        return len(self.data_range)
    def get_batch(self, t, T=63, H=3):
        assert t>=T and t<=len(self.time_points)-H
        tmp = []
        fds = []
        columns = self.data1['close'].columns
        valid_columes = []
        
        valid_columes_idx = [self.data1['close'][i].loc[self.time_points[t-T]: self.time_points[t+H+1]].isnull().any() for i in self.data1['close'].columns]
        for i in range(len(columns)):
            if not valid_columes_idx[i]:
                valid_columes.append(columns[i])
        closing_price_at_start = self.data1['close'][valid_columes].loc[self.time_points[t-T]]
        for i, f in enumerate(self.fields):
            current_data = self.data1[f][valid_columes]
            # closing_price_at_start = self.data1['close'].loc[self.time_points[t-T]]
            if f in ['open', 'high', 'low', 'close', 'vwap']:
                _field = current_data.loc[self.time_points[t-T]: self.time_points[t-1]] / closing_price_at_start
                
            else:
                min_num = current_data.loc[self.time_points[t-T]: self.time_points[t-1]].min()
                max_num = current_data.loc[self.time_points[t-T]: self.time_points[t-1]].max()
                _field = (current_data.loc[self.time_points[t-T]: self.time_points[t-1]] - min_num) / (max_num - min_num)

            _field = _field.fillna(0.)
            tmp.append(_field.values)
            
        prediction = (self.data1['close'][valid_columes].loc[self.time_points[t+H]] - self.data1['open'][valid_columes].loc[self.time_points[t+1]]) / self.data1['open'][valid_columes].loc[self.time_points[t+1]]
        percentile_ranks = prediction.rank(pct=True)
        # prediction = prediction.fillna(0.)
        Y = percentile_ranks.values
        # Y = prediction.values
        
        X = np.transpose(np.stack(tmp), [2, 0, 1])

        return X, Y
    
    def __getitem__(self, index) -> Any:
        return self.get_batch(self.data_range[index])
    
class TCN_data2(Dataset):
    def __init__(self, filename, mode="valid") -> None:
        super().__init__()
        self.df = pd.read_csv(filename)
        print("Finish reading data")
        self.mode = mode
        if self.df['datetime'].dtype == 'object':
            self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        if mode == 'train':
            self.times = self.df[self.df['datetime'] < '2020-01-01']['datetime'].unique()
        elif mode == 'valid':
            self.times = self.df[(self.df['datetime'] < '2021-01-01') & (self.df['datetime'] >= '2020-01-01')]['datetime'].unique()
        else:
            self.times = self.df[self.df['datetime'] >= '2021-01-01']['datetime'].unique()
        
    def __len__(self):
        return len(self.times)
    
    def __getitem__(self, t):
        current_time = self.times[t]
        current_time_data = self.df[self.df['datetime'] == current_time]
        
        # current_instrument = current_time_data['instrument'].unique()
        batch_x = []
        batch_y = []
        
        for idx, row in tqdm(current_time_data.iterrows()):
            instrument = row['instrument']
            instrument_df = self.df[self.df['instrument'] == instrument]
            # row = current_time_data[current_time_data['in']]
            # time_T = row['datetime']
            
            # Define the start of the prediction period (T+1)
            
            
            # Filter rows for the historical window
            historical_data = instrument_df[instrument_df['datetime'] < current_time].tail(63)
            predict_data = instrument_df[instrument_df['datetime'] > current_time].head(3)
            
            # Safety check for sufficient historical data
            if historical_data.shape[0] != 63:
                continue
            if predict_data.shape[0] != 3:
                continue
            closing_price_at_start = historical_data.iloc[0]['close']
            for col in ['open', 'high', 'low', 'close', 'vwap']:
                historical_data[col] = historical_data[col] / closing_price_at_start

            # 过去 63 日的成交量和成交金额分别进行最大最小值归一
            for col in ['volume', 'total_turnover']:
                min_col = historical_data[col].min()
                max_col = historical_data[col].max()
                if max_col != min_col:
                    historical_data[col] = (historical_data[col] - min_col) / (max_col - min_col)
                else:
                    historical_data[col] = 0.0
            predict_return_rate = (predict_data['close'].iloc[-1] - predict_data['close'].iloc[0]) / predict_data['close'].iloc[0]
            batch_x.append(historical_data[['open', 'high', 'low', 'close', 'vwap', 'volume', 'total_turnover']].values)
            batch_y.append(predict_return_rate)
        return torch.Tensor(np.stack(batch_x)), torch.Tensor(batch_y)


def pearson_correlation(x, y):
    """
    Compute the Pearson correlation coefficient between
    two tensors while preserving gradient information.
    """
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = torch.sum(xm * ym)
    r_den = torch.sqrt(torch.sum(xm ** 2) * torch.sum(ym ** 2))
    r = r_num / r_den

    # To avoid division by zero, in case of zero variance
    r = torch.where(torch.isnan(r), torch.zeros_like(r), r)
    return r
  
class CustomLoss(nn.Module):
    def __init__(self, lambda1, lambda2):
        super(CustomLoss, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        # self.n2 = n2

    def forward(self, outputs, target):
        
        """
        outputs 维度: T*64 T是股票个数，64 是因子个数
        targets: 维度 T
        """
        
        # Convert target to a float tensor in case it's not
        # target = target.float()

        # Ensure that target is a 2D row vector
        target = target.view(1, -1)

        # Calculate correlation for each factor with the target
        # Pearson correlation coefficient is used here
        corrs = [pearson_correlation(outputs[:, i], target).unsqueeze(0) for i in range(outputs.shape[1])]
        corrs = torch.cat(corrs)
        
        # Calculate the first term of the loss function
        term1 = -torch.mean(corrs)

        # Calculate the sum of correlations for the second term
        # sum_outputs =
        sum_output = torch.sum(outputs, 1)
        sum_corr = pearson_correlation(sum_output, target)
        # sum_corr = torch.sum(corrs)

        # Calculate the second term of the loss function
        term2 = -self.lambda1 * sum_corr

        # Calculate the squared correlations for the third term
        # corr_squared = corrs**2
        # pair_wise_corr = torch.cat([pearson_correlation(outputs[:, i], outputs[:, j]).unsqueeze(0) for i in range(outputs.shape[1]) for j in range(outputs.shape[1])]) ** 2
        

        # n_squared = outputs.shape[1] ** 2

        # Calculate the third term of the loss function
        term3 = self.lambda2 * torch.mean(torch.corrcoef(outputs.T) ** 2)

        # Sum all terms to get final loss
        loss_final = term1 + term2 + term3

        return loss_final
    


if __name__ == '__main__':
    print("reading data ... ...")
    df_data = pd.read_pickle("tcn_data.pkl")
    df_data = df_data.fillna(0.)
    # df_data.to_pickle("tcn_data.pkl")
    # quit()
    time_points = df_data["datetime"].unique()
    # test_data = test_data.fillna(0.)  
    fields = ['open', 'high', 'low', 'close', 'vwap', 'volume', 'total_turnover']
    dict_data = {f:df_data.pivot_table(index='datetime', columns='instrument', values=f) for f in fields} # 每一天 每一支股票的属性

    print("loading train/valid/test set ,,, ,, ")
    train_data = TCN_data(dict_data, time_points, "train")
    valid_data = TCN_data(dict_data, time_points, "valid")
    test_data = TCN_data(dict_data, time_points, "test")
    print(test_data[0])
    print(len(train_data), len(valid_data), len(test_data))
    
    # T = 10  # Let's say we have 10 stocks
    # custom_loss = CustomLoss(lambda1=0.5, lambda2=0.5)
    # outputs = torch.randn(T, 64, requires_grad=True)
    # target = torch.randn(T)

    # # Calculate loss
    # loss = custom_loss(outputs, target)
    # print(f"Calculated loss: {loss.item()}")