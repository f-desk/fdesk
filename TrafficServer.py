import torch
import torch.nn as nn
import numpy as np
import asyncio
import json
import time
import random

# ===================== 工业部署配置 =====================
MODEL_TYPE = 'LSTM'  # 与训练端保持一致
WEIGHTS_PATH = f'traffic_{MODEL_TYPE.lower()}_weights.pth'
HOST = "127.0.0.1"
PORT = 8765

# ===================== 模型结构（与训练端完全一致） =====================
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

# ===================== WebSocket 实时车流推送（原逻辑，仅替换数据流） =====================
async def stream_data(websocket):
    print(f"\n[终端接入] 客户端已连接，开始下发 {MODEL_TYPE} 车流预测数据...")
    model = TrafficRNNDemo(cell_type=MODEL_TYPE)

    # 加载训练好的权重
    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device('cpu')))
        model.eval()
        print(f"[系统] 成功加载权重: {WEIGHTS_PATH}")
    except FileNotFoundError:
        print(f"[警告] 未找到权重，使用初始权重仿真演示")

    # 生成交通流测试时序（早晚高峰周期性车流，贴合真实路况）
    t = np.linspace(0, 120, 4000)
    test_flow = 20 + 15 * np.sin(t/4) + 8 * np.cos(t/8) + np.random.normal(0, 1.5, t.shape)

    try:
        with torch.no_grad():
            for i in range(len(test_flow) - 1):
                start_time = time.time()
                input_point = torch.tensor([[[test_flow[i]]]], dtype=torch.float32)
                pred_point = model(input_point).item()
                actual_point = test_flow[i+1]

                # 模拟推理延迟 + 网络抖动
                calc_time = (time.time() - start_time) * 1000
                simulated_latency = calc_time + random.uniform(5.0, 15.0)

                # 封装JSON数据流（字段和前端完全兼容，无需改前端解析）
                payload = {
                    "timestamp": time.time() * 1000,
                    "model_type": MODEL_TYPE,
                    "ch1_actual": float(actual_point),
                    "ch2_predict": float(pred_point),
                    "error_abs": abs(float(actual_point) - float(pred_point)),
                    "latency_ms": round(simulated_latency, 2)
                }
                await websocket.send(json.dumps(payload))
                await asyncio.sleep(0.03)  # 采样间隔30ms

    except Exception as e:
        print(f"[断开] 客户端连接异常: {e}")

async def main():
    import websockets
    async with websockets.serve(stream_data, HOST, PORT):
        print("=============================================")
        print(" [SYS] 交通流边缘计算终端已启动")
        print(f" [SYS] 计算核心: {MODEL_TYPE} 时序网络")
        print(f" [SYS] 监听端口: ws://{HOST}:{PORT}")
        print("=============================================")
        print("等待前端监控面板接入...")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())