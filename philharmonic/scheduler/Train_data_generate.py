import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from philharmonic import simulator, inputgen, environment, conf
from philharmonic.scheduler.brute_force import BruteForceScheduler
from philharmonic.cloud.model import State, VMRequest

NUM_SAMPLES_TO_GENERATE = 5000
OUTPUT_DIR = 'gnn_training_data'


def create_bipartite_graph(state, vms_to_schedule, pms, vm_scaler, pm_scaler):
    if not vms_to_schedule or not pms:
        return None

    vm_features_raw = np.array([[vm.res.get('#CPUs', 0), vm.res.get('RAM', 0)] for vm in vms_to_schedule])
    vm_features = vm_scaler.transform(vm_features_raw)

    pm_features_raw = []
    for pm in pms:
        used_cpu = sum(vm.res.get('#CPUs', 0) for vm in state.alloc.get(pm, []))
        used_ram = sum(vm.res.get('RAM', 0) for vm in state.alloc.get(pm, []))
        available_cpu = pm.cap.get('#CPUs', 0) - used_cpu
        available_ram = pm.cap.get('RAM', 0) - used_ram
        pm_features_raw.append([available_cpu, available_ram])

    pm_features = pm_scaler.transform(np.array(pm_features_raw))

    x = torch.tensor(np.vstack([vm_features, pm_features]), dtype=torch.float)
    num_vms, num_pms = len(vms_to_schedule), len(pms)
    node_type = torch.tensor([0] * num_vms + [1] * num_pms, dtype=torch.long)
    edge_list_start = [i for i in range(num_vms) for _ in range(num_pms)]
    edge_list_end = [num_vms + j for _ in range(num_vms) for j in range(num_pms)]
    edge_index = torch.tensor([edge_list_start, edge_list_end], dtype=torch.long)

    edge_index = to_undirected(edge_index)

    graph_data = Data(x=x, edge_index=edge_index, node_type=node_type)
    graph_data.num_vms, graph_data.num_pms = num_vms, num_pms
    return graph_data

def get_optimal_schedule_labels(schedule, vms_to_schedule, pms):

    vm_map = {vm.id: i for i, vm in enumerate(vms_to_schedule)}
    pm_map = {pm.id: i for i, pm in enumerate(pms)}
    labels = torch.full((len(vms_to_schedule),), -1, dtype=torch.long)
    if schedule and not schedule.actions.empty:
        for action in schedule.actions.values:
            if action.name == 'migrate' and action.vm.id in vm_map:
                vm_idx = vm_map[action.vm.id]
                pm_idx = pm_map[action.server.id]
                labels[vm_idx] = pm_idx

    return labels


def collect_all_features(num_samples=100):

    print("Collecting feature data")
    all_vm_features, all_pm_features = [], []

    for _ in range(num_samples):
        sim = simulator.Simulator(conf.get_factory())
        requests = sim.environment.get_requests()
        vms_to_schedule = [req.vm for req in requests.values if req.what == 'boot']
        pms = sim.cloud.servers
        state = sim.cloud.get_current()
        if not vms_to_schedule or not pms: continue

        for vm in vms_to_schedule:
            all_vm_features.append([vm.res.get('#CPUs', 0), vm.res.get('RAM', 0)])

        for pm in pms:
            used_cpu = sum(v.res.get('#CPUs', 0) for v in state.alloc.get(pm, []))
            used_ram = sum(v.res.get('RAM', 0) for v in state.alloc.get(pm, []))
            available_cpu = pm.cap.get('#CPUs', 0) - used_cpu
            available_ram = pm.cap.get('RAM', 0) - used_ram
            all_pm_features.append([available_cpu, available_ram])

    return np.array(all_vm_features), np.array(all_pm_features)


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    vm_features_sample, pm_features_sample = collect_all_features()
    vm_scaler = StandardScaler().fit(vm_features_sample)
    pm_scaler = StandardScaler().fit(pm_features_sample)

    joblib.dump(vm_scaler, os.path.join(OUTPUT_DIR, 'vm_scaler.pkl'))
    joblib.dump(pm_scaler, os.path.join(OUTPUT_DIR, 'pm_scaler.pkl'))
    print("Scalers fitted and saved.")

    print(f"Generating {NUM_SAMPLES_TO_GENERATE} training samples...")
    for i in range(NUM_SAMPLES_TO_GENERATE):
        try:
            sim = simulator.Simulator(conf.get_factory())
            requests = sim.environment.get_requests()
            if requests.empty: continue
            sim.apply_actions(requests)
            vms_to_schedule = [req.vm for req in requests.values if req.what == 'boot']
            pms = sim.cloud.servers

            if not vms_to_schedule or not pms: continue
            oracle_scheduler = BruteForceScheduler(sim.cloud, None, sim.environment)
            optimal_schedule = oracle_scheduler.reevaluate()
            graph_data = create_bipartite_graph(sim.cloud.get_current(), vms_to_schedule, pms, vm_scaler, pm_scaler)
            if graph_data is None: continue
            labels = get_optimal_schedule_labels(optimal_schedule, vms_to_schedule, pms)
            graph_data.y = labels

            torch.save(graph_data, os.path.join(OUTPUT_DIR, f'data_{i}.pt'))

            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{NUM_SAMPLES_TO_GENERATE} samples.")
        except Exception as e:
            print(f"Error generating sample {i + 1}: {e}")
            continue

    print("Data generation complete.")


if __name__ == '__main__':
    main()
