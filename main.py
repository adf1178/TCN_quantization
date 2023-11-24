import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import *
import torch.nn.functional as F
from tcn import TCN
from tqdm import tqdm
import logging

# 1. 加载数据集
# logging.basicConfig(
#     level=logging.INFO,  # 设置日志级别，默认为 INFO
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 设置日志格式
#     datefmt='%Y-%m-%d %H:%M:%S',  # 设置日期格式
#     filename='train.log',  # 日志输出到文件，指定文件名
#     filemode='a'  # 写入模式，'w' 为覆盖写入，'a' 为追加
# )

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# 配置日志输出到文件
file_handler = logging.FileHandler('train2.log', mode='a')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# 获取根日志器，并添加上面定义的两个处理器
logging.getLogger().addHandler(console_handler)
logging.getLogger().addHandler(file_handler)
logging.getLogger().setLevel(logging.INFO)


logging.info("reading data ... ...")
df_data = pd.read_pickle("tcn_data.pkl")
df_data = df_data.fillna(0.)
time_points = df_data["datetime"].unique()
# test_data = test_data.fillna(0.)  
fields = ['open', 'high', 'low', 'close', 'vwap', 'volume', 'total_turnover']
dict_data = {f:df_data.pivot_table(index='datetime', columns='instrument', values=f) for f in fields} # 每一天 每一支股票的属性

logging.info("loading train/valid/test set ,,, ,, ")
train_data = TCN_data(dict_data, time_points, "train")
valid_data = TCN_data(dict_data, time_points, "valid")
test_data = TCN_data(dict_data, time_points, "test")
logging.info(f"{len(train_data)}, {len(valid_data)}, {len(test_data)}")
custom_loss = CustomLoss(lambda1=0.5, lambda2=1)


train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=16)
valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=8)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=8)


# 2. 创建模型实例
# 假设您的输入大小、输出大小、通道数和卷积核大小是已知的
# input_size = 7  # 例如，7个特征
# output_size = 1  # 例如，1个预测目标
# num_channels = [25, 50, 100]  # TCN 的通道数
# kernel_size = 3  # 卷积核大小
# dropout = 0.2

model = TCN = TCN(7, 64,[7,7,7,7,7], 2, 0.2)
if torch.cuda.is_available():
    model = model.cuda()

# 3. 定义优化器
optimizer = optim.Adam(model.parameters())



# 4. 训练循环
def train(epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
    # for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        if torch.cuda.is_available():
            data, target = data.float().squeeze(0).cuda(), target.float().squeeze(0).cuda()
        outputs = model(data)
        optimizer.zero_grad()
        loss = custom_loss(outputs, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return total_loss / len(train_loader)

def evaluate_on_validation_dataset(model, validation_dataloader):
    """
    Evaluate the model on the validation dataset to calculate mean IC and mean Rank IC.

    Parameters:
    model (torch.nn.Module): The model to evaluate.
    validation_dataloader (torch.utils.data.DataLoader): The DataLoader providing the validation dataset.

    Returns:
    Tuple[float, float]: The mean IC and mean Rank IC for the validation dataset.
    """
    model.eval()  # Set the model to evaluation mode.
    ic_scores = []
    rank_ic_scores = []
    total_loss = 0
    with torch.no_grad():  # No gradient computation during evaluation.
        for features, targets in validation_dataloader:
        # for features, targets in tqdm(validation_dataloader, total = len(validation_dataloader)):
            if torch.cuda.is_available():
                features, targets = features.float().squeeze(0).cuda(), targets.float().squeeze(0).cuda()
            outputs = model(features)
            loss = custom_loss(outputs, targets)
            ic, rank_ic = evaluate_ic_rankic(outputs, targets)
            ic_scores.append(ic)
            rank_ic_scores.append(rank_ic)
            total_loss += loss.item()
    # Compute the mean scores across all validation data batches
    mean_ic = sum(ic_scores) / len(ic_scores)
    mean_rank_ic = sum(rank_ic_scores) / len(rank_ic_scores)
    
    return mean_ic, mean_rank_ic, total_loss / len(validation_dataloader)

def evaluate_ic_rankic(outputs, target):
    """
    Evaluate the Information Coefficient (IC) and Rank Information Coefficient (Rank IC)
    between model outputs and actual labels.

    Parameters:
    outputs (Tensor): The model outputs, expected to be of shape (T, num_factors)
    target (Tensor): The actual labels, expected to be of shape (T,)

    Returns:
    Tuple[float, float]: A tuple containing the IC and Rank IC values.
    """
    # Ensure target is a float tensor
    target = target.float()

    # Ensure target is a 2D column vector
    target = target.view(-1, 1)

    # Calculate IC
    # Pearson correlation coefficient between the mean of the outputs and the target
    mean_outputs = torch.mean(outputs, dim=1)
    ic = torch.corrcoef(torch.stack((mean_outputs, target.squeeze())))[0, 1]

    # Calculate Rank IC
    # Spearman's rank correlation coefficient between the mean of the outputs and the target
    # This is done by ranking the data and then calculating Pearson's correlation
    rank_outputs = torch.argsort(torch.argsort(mean_outputs))
    rank_target = torch.argsort(torch.argsort(target.squeeze()))
    rank_ic = torch.corrcoef(torch.stack((rank_outputs.float(), rank_target.float())))[0, 1]

    return ic.item(), rank_ic.item()

# 5. 训练模型

if __name__ == '__main__':
    reserve = False
    if reserve:
        logging.info("Loading checkpoint ... ...")
        saved_dict = torch.load("ckpt/epoch12.bin")
        num_epoch = saved_dict['epoch']
        state_dict = saved_dict['state_dict']
        model.load_state_dict(state_dict)
    else:
        num_epoch = 0
    for epoch in range(num_epoch, 60):  # 假设训练60个epoch
        train_loss = train(epoch)
        logging.info(f"Training loss in epoch {epoch} is {train_loss}")
        # 可以添加验证步骤和保存模型的代码
        mean_ic, mean_rank_ic, eval_loss = evaluate_on_validation_dataset(model=model, validation_dataloader=valid_loader)
        logging.info(f"Epoch {epoch} Mean IC: {mean_ic}, Mean Rank IC: {mean_rank_ic}, Eval loss {eval_loss}")
        if epoch % 5 == 0:
            torch.save({"state_dict":model.state_dict(), "epoch":epoch}, f'ckpt/epoch{epoch}.bin')