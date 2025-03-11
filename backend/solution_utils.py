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
import re
import numpy as np
from typing import Dict, Any

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


class Chat:
    def __init__(self):
        self.instance = None       # 原始问题实例
        self.schedule = []         # 当前调度计划
        self.JobLength = None           # 当前作业数据 (numpy数组)
        self.OpPt = None       # 当前机器数据 (numpy数组)
        self.env = None            # FJSP环境实例
        self.state = None          # 当前环境状态
        self.disruptions = {       # 记录突发事件
            "machine_breakdowns": [],  # 格式: (machine_id, start_time, end_time)
            "new_jobs": [],            # 新增作业数据
            "canceled_jobs": set(),    # 被取消的作业ID
            "maintenance": []          # 维护计划 (machine_id, start, duration)
        }

    def reset(self):
        """重置对话状态"""
        self.__init__()

    def update_schedule(self, new_schedule):
        """更新调度计划并保持历史"""
        valid_tasks = [task for task in new_schedule if task["Duration"] > 0]
        self.schedule = valid_tasks

    def apply_maintenance(self, env):
        """应用所有已记录的维护计划到环境"""
        for machine_id, start, duration in self.maintenance_plan:
            # 在指定时间点触发维护
            env.schedule_maintenance(
                machine_id=machine_id,
                start_time=start,
                duration=duration
            )


chat = Chat()

# ================== 事件识别模块 ==================
def parse_input(input_data: str) -> Dict[str, Any]:
    """解析用户输入并返回结构化事件数据"""
    input_data = input_data.strip().lower()
    
    # 模式匹配字典（顺序敏感，优先匹配复杂模式）
    patterns = [
        ("machine_breakdown", 
         r"机器(\d+)(?:在)?(\d+)[时至]?到?(\d+)?[时]?发生故障"),
        ("new_job", 
         r"新增作业[::](.+)(?:，|,)处理时间((?:\d+[ms]? ?)+)"),
        ("cancel_job",
         r"取消作业(\d+)(?:，|,)?(?:原因:.+)?"),
        ("delay_operation",
         r"作业(\d+)的操作(\d+)延迟(\d+)(单位时间|分钟|小时)"),
        ("maintenance", 
        r"机器(\d+)在(\d+)时间进行(\d+)单位时间维护")
    ]

    for event_type, pattern in patterns:
        match = re.findall(pattern, input_data)
        if match:
            return {
                "event_type": event_type,
                "params": match,
                "raw_input": input_data
            }
    
    # 检查是否是FJSP实例
    if re.match(r"^\d+\s+\d+\s+[\d\.]+", input_data):
        return {"event_type": "fjsp_instance", "data": input_data}
    
    return {"event_type": "unknown", "raw_input": input_data}

# ================== 事件处理模块 ==================
def handle_machine_breakdown(chat: Chat, params: tuple):
    """处理机器故障事件"""
    # 示例输入："机器3在10到15时发生故障"
    machine_id = int(params[0]) - 1  # 转换为0-based索引
    start = int(params[1])
    end = int(params[2]) if params[2] else start + 1
    
    # 记录故障时间段
    chat.disruptions["machine_breakdowns"].append(
        (machine_id, start, end)
    )
    
    # 更新环境中的机器可用性
    if chat.env:
        chat.env.update_machine_availability(machine_id, start, end, available=False)

def handle_new_job(chat: Chat, params: tuple):
    """处理新增作业事件"""
    # 示例输入："新增作业：工序序列3 2，处理时间5 7"
    ops = list(map(int, params[0].split()))
    proc_times = list(map(int, params[1].split()))
    
    # 创建新作业数据
    new_job = {
        "job_id": len(chat.jobs) + 1,
        "operations": list(zip(ops, proc_times))
    }
    
    # 更新作业数据
    chat.disruptions["new_jobs"].append(new_job)
    chat.jobs = np.append(chat.jobs, new_job_data)  # 需要根据实际数据结构调整

# 添加维护处理函数
def handle_maintenance(chat: Chat, params: tuple):
    """处理周期性维护事件"""
    # 示例输入："机器0在50时间进行40单位时间维护"
    print(params)
    for machine_id, start, duration in params:
        chat.env.schedule_maintenance(int(machine_id), int(start), int(duration))

    chat.env.reset()

    return reschedule(chat)


    
# ================== 调度更新模块 ==================
def reschedule(chat: Chat):
    """基于当前状态重新生成调度"""

    current_time = 0
    state = chat.env.reset()
    while True:
        with torch.no_grad():
            pi, *_ = ppo.policy(fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor, candidate=state.candidate_tensor,
                fea_m=state.fea_m_tensor, mch_mask=state.mch_mask_tensor, comp_idx=state.comp_idx_tensor,
                dynamic_pair_mask=state.dynamic_pair_mask_tensor, fea_pairs=state.fea_pairs_tensor
            )
        action = greedy_select_action(pi)
        state, _, done = chat.env.step(action)
        chat.env.check_maintenance(current_time)
        current_time = max([task['Start'] + task['Duration'] for task in chat.env.tasks_data])
        if done: break
    
    # 5. 更新对话状态
    chat.update_schedule(chat.env.tasks_data)
    return chat.schedule

# ================== 主处理流程 ==================
def analysis_data(input_data: str):
    # 步骤1：解析输入
    event = parse_input(input_data)
    
    # 步骤2：处理不同事件类型
    if event["event_type"] == "fjsp_instance":
        chat.reset()
        initialize_system(event["data"])
    elif event["event_type"] == "machine_breakdown":
        handle_machine_breakdown(chat, event["params"])
    elif event["event_type"] == "new_job":
        handle_new_job(chat, event["params"])
    elif event["event_type"] == "maintenance":
        handle_maintenance(chat, event["params"])
    
    # 步骤3：触发重新调度
    if event["event_type"] != "unknown":
        new_schedule = reschedule(chat)
        return {"schedule": new_schedule}
    else:
        return {"error": "无法识别输入类型"}

# ================== 辅助函数 ==================
def initialize_system(instance_text: str):
    """初始化FJSP系统"""
    lines = [line.strip().replace('\t', ' ') 
            for line in instance_text.split('\n') 
            if line.strip()]
    
    # 解析原始数据
    JobLength, OpPT, *args = text_to_matrix(lines)
    
    # 初始化环境
    chat.JobLength = JobLength
    chat.OpPt = OpPT
    chat.env = FJSPEnvForVariousOpNums(
        n_j=JobLength.shape[0], 
        n_m=OpPT.shape[1]
    )
    chat.state = chat.env.set_initial_data(
        JobLength[np.newaxis, :],
        OpPT[np.newaxis, :]
    )
    
    # 初始调度
    reschedule(chat)

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
                pi, *_ = ppo.policy(fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor, candidate=state.candidate_tensor,
                    fea_m=state.fea_m_tensor, mch_mask=state.mch_mask_tensor, comp_idx=state.comp_idx_tensor,
                    dynamic_pair_mask=state.dynamic_pair_mask_tensor, fea_pairs=state.fea_pairs_tensor
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
        
        result = analysis_data(request.instance_text)
        print(result)
        return result
    
    except Exception as e:
        # 打印完整错误堆栈
        import traceback
        traceback.print_exc()
        
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)