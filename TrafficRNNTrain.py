import torch
import torch.nn as nn
import numpy as np
import os
import signal
import sys
import pandas as pd

# ===================== 教学配置：切换模型 RNN / GRU / LSTM =====================
MODEL_TYPE = 'LSTM'  # 可修改为 RNN / GRU / LSTM 对比实验
CHECKPOINT_PATH = f'traffic_{MODEL_TYPE.lower()}_checkpoint.pth'
FINAL_WEIGHTS_PATH = f'traffic_{MODEL_TYPE.lower()}_weights.pth'

# ===================== 共享GPU断点续训 - 信号捕获逻辑（完全保留原代码） =====================
def receive_signal(signum, frame):
    print(f"\n[警告] 收到资源回收信号 (Signal: {signum})!正在紧急保存当前进度...")
    global model, optimizer, epoch, loss
    save_checkpoint(epoch, model, optimizer, loss, path=CHECKPOINT_PATH)
    print(f"[退出] {MODEL_TYPE} 模型的进度已安全保存,程序优雅退出。")
    sys.exit(0)

signal.signal(signal.SIGTERM, receive_signal)
signal.signal(signal.SIGINT, receive_signal)

def save_checkpoint(epoch, model, optimizer, loss, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)
    print(f"-> 检查点已保存至: {path}")

# ===================== 循环网络核心类（完全保留原结构，无修改） =====================
class TrafficRNNDemo(nn.Module):
    def __init__(self, cell_type='LSTM', input_size=1, hidden_size=32, num_layers=1, output_size=1):
        super(TrafficRNNDemo, self).__init__()
        self.cell_type = cell_type.upper()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if self.cell_type == 'RNN':
            self.rnn_core = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        elif self.cell_type == 'GRU':
            self.rnn_core = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        elif self.cell_type == 'LSTM':
            self.rnn_core = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError("未知的网络类型!请选择 'RNN', 'GRU' 或 'LSTM'")

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        device = x.device
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        if self.cell_type == 'LSTM':
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            out, (hn, cn) = self.rnn_core(x, (h0, c0))
        else:
            out, hn = self.rnn_core(x, h0)
        return self.fc(out)

# ===================== 【核心改造】交通流时序数据加载（替换原脑电模拟数据） =====================
def load_traffic_data(device, data_path="traffic_flow.csv"):
    """
    加载加州高速车流一维时序数据，构造 输入x / 标签y (单步预测)
    逻辑：用 t 时刻车流 预测 t+1 时刻车流
    """
    try:
        # 读取车流数据，取单车道车流量列
        df = pd.read_csv(data_path)
        # 提取车流时序列，根据你的CSV列名修改此处！示例列名：flow
        flow_data = df["flow"].values.astype(np.float32)
    except Exception as e:
        print(f"[警告] 未找到交通流数据集，使用模拟车流数据演示: {e}")
        # 兜底：模拟周期性车流（早晚高峰+随机噪声，贴合交通特征）
        t = np.linspace(0, 80, 600)
        flow_data = 20 + 15 * np.sin(t/4) + 8 * np.cos(t/8) + np.random.normal(0, 1.5, t.shape)

    # 构造时序样本：x = 前一时刻, y = 后一时刻
    x = torch.tensor(flow_data[:-1], dtype=torch.float32).view(1, -1, 1).to(device)
    y = torch.tensor(flow_data[1:], dtype=torch.float32).view(1, -1, 1).to(device)
    return x, y

# ===================== 主训练逻辑（仅改名称，逻辑完全保留） =====================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"======== 交通流时序预测实验: 当前训练 【{MODEL_TYPE}】 模型 =========")
    print(f"运行设备: {device}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 实例化模型
    model = TrafficRNNDemo(cell_type=MODEL_TYPE).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    start_epoch = 0
    total_epochs = 200
    loss = torch.tensor(0.0)

    # 断点续训
    if os.path.exists(CHECKPOINT_PATH):
        print(f"发现【{MODEL_TYPE}】历史训练记录，正在恢复...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        print(f"成功恢复! 从第 {start_epoch} 个 Epoch 继续。")

    # 加载交通流数据
    x, y = load_traffic_data(device)

    try:
        for epoch in range(start_epoch, total_epochs):
            model.train()
            outputs = model(x)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'[{MODEL_TYPE}] Epoch [{epoch+1}/{total_epochs}], Loss: {loss.item():.4f}')
            save_checkpoint(epoch, model, optimizer, loss, path=CHECKPOINT_PATH)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"\n[成功] {MODEL_TYPE} 交通流模型训练完成!")
        torch.save(model.state_dict(), FINAL_WEIGHTS_PATH)
        print(f"部署权重已保存至: {FINAL_WEIGHTS_PATH}")

        if os.path.exists(CHECKPOINT_PATH):
            os.remove(CHECKPOINT_PATH)

    except Exception as e:
        print(f"训练发生意外: {e}")
        save_checkpoint(epoch, model, optimizer, loss, path=CHECKPOINT_PATH)