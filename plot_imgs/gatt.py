import pygame

def initialize_pygame():
    pygame.init()
    screen_width = 1200
    screen_height = 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    return screen

def get_color_for_task(task_name, task_colors, COLORS):
    if task_name not in task_colors:
        task_colors[task_name] = COLORS[len(task_colors) % len(COLORS)]
    return task_colors[task_name]

def render_gantt_chart(tasks_data, screen):
    WHITE = (255, 255, 255)
    COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 165, 0), (75, 0, 130), (238, 130, 238)]
    task_colors = {}
    
    screen.fill(WHITE)
    for task in tasks_data:
        color = get_color_for_task(task['Task'], task_colors, COLORS)
        x = 100 + task['Start'] * 5  # 缩放因子调整
        y = 100 + int(task['Station'].replace('Machine', '')) * 100  # 间隔调整
        width = task['Duration'] * 5 - 1  # 缩放因子调整
        height = int(60 * task['Width'])  # 高度调整
        pygame.draw.rect(screen, color, (x, y, width, height))
    pygame.display.flip()

# 假设这是强化学习环境中的一步
def step():
    # 更新环境状态逻辑（此处略）
    pass

# 假设这是环境重置逻辑
def reset():
    # 重置环境状态逻辑（此处略）
    pass

# 示例：初始化pygame，并在强化学习的某个时间点渲染甘特图
screen = initialize_pygame()
tasks_data = [
    {'Task': 'Job1', 'Station': 'Machine0', 'Start': 0.0, 'Duration': 45, 'Width': 0.4},
    {'Task': 'Job1', 'Station': 'Machine0', 'Start': 45.0, 'Duration': 47, 'Width': 0.4},
    {'Task': 'Job2', 'Station': 'Machine0', 'Start': 92.0, 'Duration': 48, 'Width': 0.4},
    {'Task': 'Job2', 'Station': 'Machine1', 'Start': 140.0, 'Duration': 81, 'Width': 0.4},
    {'Task': 'Job0', 'Station': 'Machine0', 'Start': 140.0, 'Duration': 40, 'Width': 0.4},
    {'Task': 'Job0', 'Station': 'Machine0', 'Start': 180.0, 'Duration': 31, 'Width': 0.4},
]  # 强化学习环境中的任务数据

for i in range(100):

    render_gantt_chart(tasks_data, screen)

# 此处可加入主循环或其它逻辑
