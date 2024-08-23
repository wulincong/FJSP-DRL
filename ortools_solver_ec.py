import os
os.environ['ON_PY']="1"
from ortools.sat.python import cp_model
import collections
import time
import numpy as np
from tqdm import tqdm
import sys
from params import configs
from data_utils import pack_data_from_config
import collections


class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""
    def __init__(self):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__solution_count = 0

    def on_solution_callback(self):
        self.__solution_count += 1
        # print(f'Solution {self.__solution_count}, objective = {self.ObjectiveValue()}')

def fjsp_solver(jobs, num_machines, time_limits, processing_energy, standby_energy, energy_weight=0.0, makespan_weight=1.0):
    # 定义倍数来消除小数部分
    scale_factor = 10

    # 将处理能耗和待机能耗标准化为整数
    processing_energy = [int(e * scale_factor) for e in processing_energy]
    standby_energy = [int(e * scale_factor) for e in standby_energy]

    num_jobs = len(jobs)
    all_jobs = range(num_jobs)
    all_machines = range(num_machines)

    # Model the flexible jobshop problem.
    model = cp_model.CpModel()

    horizon = 0
    for job in jobs:
        for task in job:
            max_task_duration = 0
            for alternative in task:
                max_task_duration = max(max_task_duration, alternative[0])
            horizon += max_task_duration

    # Global storage of variables.
    intervals_per_resources = collections.defaultdict(list)
    starts = {}  # indexed by (job_id, task_id).
    presences = {}  # indexed by (job_id, task_id, alt_id).
    job_ends = []
    machine_end_times = [model.NewIntVar(0, horizon, f'machine_end_time_{m}') for m in all_machines]
    total_energy = model.NewIntVar(0, int(1e6), 'total_energy')  # Arbitrary large upper bound

    energy_terms = []

    # Scan the jobs and create the relevant variables and intervals.
    for job_id in all_jobs:
        job = jobs[job_id]
        num_tasks = len(job)
        previous_end = None
        for task_id in range(num_tasks):
            task = job[task_id]

            min_duration = task[0][0]
            max_duration = task[0][0]

            num_alternatives = len(task)
            all_alternatives = range(num_alternatives)

            for alt_id in range(1, num_alternatives):
                alt_duration = task[alt_id][0]
                min_duration = min(min_duration, alt_duration)
                max_duration = max(max_duration, alt_duration)

            # Create main interval for the task.
            suffix_name = '_j%i_t%i' % (job_id, task_id)
            start = model.NewIntVar(0, horizon, 'start' + suffix_name)
            duration = model.NewIntVar(min_duration, max_duration, 'duration' + suffix_name)
            end = model.NewIntVar(0, horizon, 'end' + suffix_name)
            interval = model.NewIntervalVar(start, duration, end, 'interval' + suffix_name)

            # Store the start for the solution.
            starts[(job_id, task_id)] = start

            # Add precedence with previous task in the same job.
            if previous_end is not None:
                model.Add(start >= previous_end)
            previous_end = end

            # Create alternative intervals.
            if num_alternatives > 1:
                l_presences = []
                for alt_id in all_alternatives:
                    alt_suffix = '_j%i_t%i_a%i' % (job_id, task_id, alt_id)
                    l_presence = model.NewBoolVar('presence' + alt_suffix)
                    l_start = model.NewIntVar(0, horizon, 'start' + alt_suffix)
                    l_duration = task[alt_id][0]
                    l_end = model.NewIntVar(0, horizon, 'end' + alt_suffix)
                    l_interval = model.NewOptionalIntervalVar(l_start, l_duration, l_end, l_presence, 'interval' + alt_suffix)
                    l_presences.append(l_presence)

                    # Link the master variables with the local ones.
                    model.Add(start == l_start).OnlyEnforceIf(l_presence)
                    model.Add(duration == l_duration).OnlyEnforceIf(l_presence)
                    model.Add(end == l_end).OnlyEnforceIf(l_presence)

                    # Add the local interval to the right machine.
                    intervals_per_resources[task[alt_id][1]].append(l_interval)

                    # Store the presences for the solution.
                    presences[(job_id, task_id, alt_id)] = l_presence

                    # Energy consumption for this task
                    energy_terms.append(l_presence * l_duration * processing_energy[task[alt_id][1]])

                # Select exactly one presence variable.
                model.AddExactlyOne(l_presences)
            else:
                intervals_per_resources[task[0][1]].append(interval)
                presences[(job_id, task_id, 0)] = model.NewConstant(1)

                # Energy consumption for this task
                energy_terms.append(duration * processing_energy[task[0][1]])

        job_ends.append(previous_end)

    # Create machines constraints.
    for machine_id in all_machines:
        intervals = intervals_per_resources[machine_id]
        if len(intervals) > 1:
            model.AddNoOverlap(intervals)
        # Update machine end time
        model.AddMaxEquality(machine_end_times[machine_id], [end for interval in intervals for end in [interval.EndExpr()]])

    # Makespan objective
    makespan = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(makespan, job_ends)
    
    # Add energy consumption objective
    model.Add(total_energy == sum(energy_terms) + sum((machine_end_times[m] - makespan) * standby_energy[m] for m in all_machines))

    # Combined objective
    model.Minimize(makespan_weight * makespan + energy_weight * total_energy)

    # Solve model.
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limits
    solution_printer = SolutionPrinter()

    total1 = time.time()
    status = solver.Solve(model, solution_printer)
    total2 = time.time()
    # Print final solution.
    # for job_id in all_jobs:
    #     print('Job %i:' % job_id)
    #     for task_id in range(len(jobs[job_id])):
    #         start_value = solver.Value(starts[(job_id, task_id)])
    #         machine = -1
    #         duration = -1
    #         selected = -1
    #         for alt_id in range(len(jobs[job_id][task_id])):
    #             if solver.Value(presences[(job_id, task_id, alt_id)]):
    #                 duration = jobs[job_id][task_id][alt_id][0]
    #                 machine = jobs[job_id][task_id][alt_id][1]
    #                 selected = alt_id
    #         print(
    #             '  task_%i_%i starts at %i (alt %i, machine %i, duration %i)' %
    #             (job_id, task_id, start_value, selected, machine, duration))
    
    # print('Solve status: %s' % solver.StatusName(status))
    # print('Optimal objective value: %i' % solver.ObjectiveValue())
    # print('Statistics')
    # print('  - conflicts : %i' % solver.NumConflicts())
    # print('  - branches  : %i' % solver.NumBranches())
    # print('  - wall time : %f s' % solver.WallTime())
    return solver.ObjectiveValue(), solver.Value(makespan), solver.Value(total_energy), total2 - total1

# 示例输入数据
input_data = [
    "5    5    3.2",
    "5 2 1 75 3 60 2 4 94 5 90 3 1 29 4 14 5 56 3 1 24 4 96 5 99 2 2 58 4 11",
    "5 4 1 14 3 38 4 89 5 86 2 1 86 5 95 2 1 58 4 41 5 1 89 2 44 3 34 4 86 5 62 3 1 43 2 58 4 69",
    "5 5 1 59 2 77 3 80 4 81 5 67 4 1 93 2 49 4 96 5 33 4 1 62 2 89 4 80 5 47 4 2 9 3 85 4 5 5 42 5 1 87 2 74 3 36 4 39 5 39",
    "5 1 4 41 3 2 29 4 81 5 53 4 1 99 2 22 3 59 4 93 5 1 18 2 76 3 59 4 4 5 33 2 2 22 4 54",
    "5 2 2 53 4 14 2 2 49 3 63 5 1 69 2 35 3 82 4 96 5 19 2 2 22 5 52 4 1 49 2 93 3 45 4 81",
    "3.7183717542336017 2.4572492347302832 2.0053854193759095 4.193905327881749 2.2462623864898834",
    "0.1 0.1 0.1 0.1 0.1"
]


from data_utils import text_to_matrix_with_ec
from ortools_solver import matrix_to_the_format_for_solving

# job_length, op_pt, processing_energy, standby_energy = text_to_matrix_with_ec(input_data)

# jobs, num_machines = matrix_to_the_format_for_solving(job_length, op_pt)


def solve_instances(config):
    """
        Solve 'test_data' from 'data_source' using OR-Tools
        with time limits 'max_solve_time' for each instance,
        and save the result to './or_solution/{data_source}'
    :param config: a package of parameters
    :return:
    """
    # p = psutil.Process()
    # p.cpu_affinity(range(config.low, config.high))

    if not os.path.exists(f'./or_solution/{config.data_source}'):
        os.makedirs(f'./or_solution/{config.data_source}')

    data_list = pack_data_from_config(config.data_source, config.test_data)

    save_direc = f'./or_solution/{config.data_source}'
    if not os.path.exists(save_direc):
        os.makedirs(save_direc)
    energy_weight= config.factor_Ec
    makespan_weight=config.factor_Mk
    energy_sum = 0
    makespan_sum = 0
    for data in data_list:
        dataset = data[0]
        data_name = data[1]
        save_path = save_direc + f'/solution_{data_name}.npy'
        save_subpath = save_direc + f'/{data_name}'

        if not os.path.exists(save_subpath):
            os.makedirs(save_subpath)

        makespan_list = []
        total_energy_list = []
        if (not os.path.exists(save_path)) or config.cover_flag:
            print("-" * 25 + "Solve Setting" + "-" * 25)
            print(f"solve data name : {data_name}")
            print(f"path : ./data/{config.data_source}/{data_name}")

            # search for the start index
            for root, dirs, files in os.walk(save_subpath):
                index = len([int(f.split("_")[-1][:-4]) for f in files])

            print(f"left instances: dataset[{index}, {len(dataset[0])})")
            for k in tqdm(range(index, len(dataset[0])), file=sys.stdout, desc="progress", colour='blue'):
                # for i in range(1, 3):

                    jobs, num_machines = matrix_to_the_format_for_solving(dataset[0][k], dataset[1][k])
                    processing_energy, standby_energy = dataset[2][k], dataset[3][k]
                    objective_value, makespan, total_energy, solve_time  = fjsp_solver(jobs,
                                                    num_machines,
                                                    config.max_solve_time
                                                    , processing_energy
                                                    , standby_energy
                                                    , energy_weight
                                                    , makespan_weight)
                    makespan_list.append(makespan)
                    total_energy_list.append(total_energy)
                    # tqdm.write(
                    #     f"Instance {k + 1}, solution:{solution}, solveTime:{solveTime}, systemtime:{time.strftime('%m-%d %H:%M:%S')}")
                    # np.save(save_subpath + f'/solution_{data_name}_{str.zfill(str(k + 1), 3)}.npy',
                    #         np.array([solution, solveTime]))
                    print(f"Objective Value: {objective_value} Makespan: {makespan} Total Energy: {total_energy} Solve Time: {solve_time} seconds")
                    # print(f"Makespan: {makespan}")
                    # print(f"Total Energy: {total_energy}")
                    # print(f"Solve Time: {solve_time} seconds")
            print("mean makespan", np.mean(makespan_list))
            print("mean energy", np.mean(total_energy_list))
            # print("load results...")
            # results = []
            # for i in range(len(dataset[0])):
            #     solve_msg = np.load(save_subpath + f'/solution_{data_name}_{str.zfill(str(i + 1), 3)}.npy')
            #     results.append(solve_msg)
        energy_sum += np.mean(total_energy_list)
        makespan_sum += np.mean(makespan_list)
            # np.save(save_path, np.array(results))
            # print("successfully save results...")

    return energy_sum / len(data_list), makespan_sum / len(data_list)



# for i in range(10):
#     energy_weight= i / 100
#     makespan_weight=(10-i) / 10
#     objective_value, makespan, total_energy, solve_time = fjsp_solver(jobs, num_machines, 60, processing_energy, standby_energy, energy_weight, makespan_weight)

#     print(f"Objective Value: {objective_value}")
#     print(f"Makespan: {makespan}")
#     print(f"Total Energy: {total_energy}")
#     print(f"Solve Time: {solve_time} seconds")


if __name__ == '__main__':
    from params import parser
    instances = [ "10x5EC+ECMK",]
    test_data_list = [ "5x5+mix"]
    
    args = ["--test_data", *test_data_list,
            "--test_model", *instances]

    for i in range(10):
        energy_weight= i / 10 
        makespan_weight=(10-i) / 10
        print("="*50)
        print( f"energy_weight: {energy_weight}, makespan_weight:{makespan_weight}" )
        ec_args = ["--fea_j_input_dim", "16", 
            "--fea_m_input_dim", "11",
            '--factor_Mk', f"{makespan_weight}",
            '--factor_Ec', f"{energy_weight}", 
            "--model_source", "SD2EC",
            "--data_source", "SD2EC",
            "--lr", "1e-4",
            "--n_j", "5",
            "--n_m", "5" 
            ]

        args = [*ec_args, *args]

        configs = parser.parse_args(args=args)

        mean_energy, mean_mk = solve_instances(config=configs)

        print(f"mean_energy: {mean_energy}, meanspan_energy: {mean_mk}")
