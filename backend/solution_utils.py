import numpy as np
import os, sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
os.environ['ON_PY'] = "1"
from params import configs
from data_utils import CaseGenerator, SD2_instance_generator, text_to_matrix, matrix_to_text
import random
from model.PPO import PPO_initialize, Memory
os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id
import torch
from common_utils import *
import time
from fjsp_env_various_op_nums import FJSPEnvForVariousOpNums
from fastapi.middleware.cors import CORSMiddleware  # 添加这行

app = FastAPI()

# 添加CORS中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议指定具体域名
    allow_credentials=True,
    allow_methods=["*"],    # 允许所有方法
    allow_headers=["*"],    # 允许所有头
)

# 加载模型（在启动时加载一次）
model_path = "trained_network/vari_vae_SD2_2025-03-06T19-39-59.pth"
ppo = PPO_initialize(configs)
ppo.policy.load_state_dict(torch.load(model_path, map_location='cuda', weights_only=True))
ppo.policy.eval()  # 设置为评估模式

class InstanceRequest(BaseModel):
    instance_text: str

def process_instance(instance_text: str):
    try:
        # 解析输入文本
        JobLength, OpPT, *args = text_to_matrix(instance_text.split('\n'))
        n_j = JobLength.shape[0]
        n_m = OpPT.shape[1]
        
        # 初始化环境
        env = FJSPEnvForVariousOpNums(n_j, n_m)
        JobLength_list = JobLength[np.newaxis, :]
        OpPT_list = OpPT[np.newaxis, :]
        state = env.set_initial_data(JobLength_list, OpPT_list)

        memory = Memory(gamma=configs.gamma, gae_lambda=configs.gae_lambda)

        # 运行调度
        while True:
            with torch.no_grad():
                pi, *_ = ppo.policy(
                    fea_j=state.fea_j_tensor,
                    op_mask=state.op_mask_tensor,
                    candidate=state.candidate_tensor,
                    fea_m=state.fea_m_tensor,
                    mch_mask=state.mch_mask_tensor,
                    comp_idx=state.comp_idx_tensor,
                    dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                    fea_pairs=state.fea_pairs_tensor
                )
            action = greedy_select_action(pi)
            state, reward, done = env.step(actions=action.cpu().numpy())
            if done:
                break
        # 修改后的process_instance函数末尾
        valid_tasks = [task for task in env.tasks_data if task["Duration"] > 0]
        return {"schedule": valid_tasks}  # 只返回有效任务

    except Exception as e:
        raise RuntimeError(f"Error processing instance: {str(e)}")

@app.post("/schedule")
async def create_schedule(request: InstanceRequest):
    try:
        # 添加输入日志
        print("Received instance text:", request.instance_text)
        
        result = process_instance(request.instance_text)
        return result
    except Exception as e:
        # 打印完整错误堆栈
        import traceback
        traceback.print_exc()
        
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)