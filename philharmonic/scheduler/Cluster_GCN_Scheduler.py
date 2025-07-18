import os
import torch
import numpy as np
import joblib
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.nn import ClusterGCNConv
from philharmonic.schedule import Schedule
from philharmonic.scheduler.ischeduler import IScheduler
from philharmonic.cloud.model import Migration
from philharmonic.logger import info
from torch_geometric.utils import to_undirected
from torch_geometric.nn import ClusterGCNConv

class ClusterGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = ClusterGCNConv(in_channels, hidden_channels)
        self.conv2 = ClusterGCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class ClusterGCNScheduler(IScheduler):

    def __init__(self, cloud=None, driver=None, environment=None):
        super().__init__(cloud, driver, environment)

        model_save_path = 'gnn_training_data/cluster_gcn_models.pth'
        vm_scaler_path = 'gnn_training_data/vm_scaler.pkl'
        pm_scaler_path = 'gnn_training_data/pm_scaler.pkl'

        if not all(os.path.exists(p) for p in [model_save_path, vm_scaler_path, pm_scaler_path]):
            raise FileNotFoundError("Files not found.")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ClusterGCN(in_channels=2, hidden_channels=64, out_channels=32)
        self.scorer = torch.nn.Linear(32 * 2, 1)
        checkpoint = torch.load(model_save_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scorer.load_state_dict(checkpoint['scorer_state_dict'])
        self.model.to(self.device)
        self.scorer.to(self.device)
        self.model.eval()
        self.scorer.eval()
        self.vm_scaler = joblib.load(vm_scaler_path)
        self.pm_scaler = joblib.load(pm_scaler_path)

        info(f"Scheduler initialized: {self.device}.")

    def reevaluate(self):

        schedule = Schedule()
        state = self.cloud.get_current()
        t = self.environment.get_time()
        vms_to_schedule = [req.vm for req in self.environment.get_requests().values if req.what == 'boot']
        pms = self.cloud.servers
        if not vms_to_schedule or not pms:
            return schedule
        graph_data = self._create_inference_graph(state, vms_to_schedule, pms)
        if graph_data is None: return schedule
        graph_data = graph_data.to(self.device)

        with torch.no_grad():
            node_embeddings = self.model(graph_data.x, graph_data.edge_index)

            num_vms, num_pms = len(vms_to_schedule), len(pms)

            src_nodes, dst_nodes = [], []
            for i in range(num_vms):
                for j in range(num_pms):
                    src_nodes.append(i)
                    dst_nodes.append(num_vms + j)

            src = torch.tensor(src_nodes, device=self.device)
            dst = torch.tensor(dst_nodes, device=self.device)

            edge_emb = torch.cat([node_embeddings[src], node_embeddings[dst]], dim=-1)
            edge_scores = self.scorer(edge_emb).squeeze()

        placement_scores = edge_scores.view(num_vms, num_pms)
        sorted_pm_indices = torch.argsort(placement_scores, dim=1, descending=True)

        pm_cpu_usage = {pm.id: 0 for pm in pms}
        pm_ram_usage = {pm.id: 0 for pm in pms}

        for vm_idx, vm in enumerate(vms_to_schedule):
            for pm_idx in sorted_pm_indices[vm_idx]:
                target_pm = pms[pm_idx.item()]
                required_cpu = vm.res.get('#CPUs', 0)
                required_ram = vm.res.get('RAM', 0)
                if self._check_fit(target_pm, required_cpu, required_ram, pm_cpu_usage, pm_ram_usage):
                    action = Migration(vm, target_pm)
                    schedule.add(action, t)
                    pm_cpu_usage[target_pm.id] += required_cpu
                    pm_ram_usage[target_pm.id] += required_ram
                    info(f"ClusterGCN-Scheduler: Placing VM {vm.id} on PM {target_pm.id}")
                    break
            else:
                info(f"ClusterGCN-Scheduler: No suitable PM for VM {vm.id}")

        return schedule

    def _create_inference_graph(self, state, vms_to_schedule, pms):
        vm_features_raw = np.array([[vm.res.get('#CPUs', 0), vm.res.get('RAM', 0)] for vm in vms_to_schedule])
        vm_features_scaled = self.vm_scaler.transform(vm_features_raw)
        pm_features_raw = []
        for pm in pms:
            used_cpu = sum(vm.res.get('#CPUs', 0) for vm in state.alloc.get(pm, []))
            used_ram = sum(vm.res.get('RAM', 0) for vm in state.alloc.get(pm, []))
            available_cpu = pm.cap.get('#CPUs', 0) - used_cpu
            available_ram = pm.cap.get('RAM', 0) - used_ram
            pm_features_raw.append([available_cpu, available_ram])

        pm_features_scaled = self.pm_scaler.transform(np.array(pm_features_raw))
        x = torch.tensor(np.vstack([vm_features_scaled, pm_features_scaled]), dtype=torch.float)
        num_vms, num_pms = len(vms_to_schedule), len(pms)

        edge_src, edge_dst = [], []
        for i in range(num_vms):
            for j in range(num_pms):
                edge_src.append(i)
                edge_dst.append(num_vms + j)

        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_index = to_undirected(edge_index)

        return Data(x=x, edge_index=edge_index)

    def _check_fit(self, pm, required_cpu, required_ram, temp_cpu_usage, temp_ram_usage):
        state = self.cloud.get_current()
        used_cpu = sum(vm.res.get('#CPUs', 0) for vm in state.alloc.get(pm, [])) + temp_cpu_usage[pm.id]
        used_ram = sum(vm.res.get('RAM', 0) for vm in state.alloc.get(pm, [])) + temp_ram_usage[pm.id]
        total_cpu = pm.cap.get('#CPUs', 0)
        total_ram = pm.cap.get('RAM', 0)
        available_cpu = total_cpu - used_cpu
        available_ram = total_ram - used_ram
        return available_cpu >= required_cpu and available_ram >= required_ram

    def initialize(self):
        pass

    def finalize(self):
        pass
