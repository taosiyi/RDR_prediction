import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import multiprocessing
import gc
import time

def load_global_stats(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    stats = {
        'lumped_mean': data['lumped_mean'],
        'lumped_std': data['lumped_std'],
        'point_mean': data['point_mean'],
        'point_std': data['point_std']
    }
    print(f"Successfully loaded global stats from {npz_path}")
    return stats

def process_file_SF(file_path, window_size, step_size, compute_global,
                    lumped_cols, point_k_cols,win_avg_cols):
    samples, all_lumped_data, all_point_data, all_target_data = [], [], [], []
    car = os.path.basename(os.path.dirname(file_path)).replace("discharge_segment_", "")
    seg = os.path.basename(file_path).replace(".pkl", "")
    try:
        df = pd.read_pickle(file_path)
    except Exception as e:
        print(f"Skip {file_path}: {e}")
        return [], [], [], []

    if 'DR_5soc' not in df.columns: return [], [], [], []

    df['reqDate'] = pd.to_datetime(df['reqDate'])
    df['time_s'] = (df['reqDate'] - df['reqDate'].iloc[0]).dt.total_seconds()
    df['P'] = df['totalVoltage'] * df['totalCurrent']

    temp_cols = [c for c in df.columns if c.startswith('tempProbe_')]
    volt_cols = [c for c in df.columns if c.startswith('cellVoltage_')]

    if temp_cols: df[temp_cols] = df[temp_cols].interpolate(limit_direction='both')
    if volt_cols: df[volt_cols] = df[volt_cols].interpolate(limit_direction='both')

    stats_df = pd.DataFrame(index=df.index)
    if temp_cols:
        stats_df['Temp_mean'] = df[temp_cols].mean(axis=1)
        stats_df['Temp_std'] = df[temp_cols].std(axis=1)
        stats_df['Temp_max'] = df[temp_cols].max(axis=1)
        stats_df['Temp_min'] = df[temp_cols].min(axis=1)
    else:
        stats_df[['Temp_mean', 'Temp_std', 'Temp_max', 'Temp_min']] = 0.0

    if volt_cols:
        stats_df['Volt_mean'] = df[volt_cols].mean(axis=1)
        stats_df['Volt_std'] = df[volt_cols].std(axis=1)
        stats_df['Volt_max'] = df[volt_cols].max(axis=1)
        stats_df['Volt_min'] = df[volt_cols].min(axis=1)
    else:
        stats_df[['Volt_mean', 'Volt_std', 'Volt_max', 'Volt_min']] = 0.0

    df = pd.concat([df, stats_df], axis=1)
    df = df.dropna(subset=['DR_5soc'] + lumped_cols + point_k_cols + win_avg_cols).reset_index(drop=True)
    n = len(df)
    if n <= window_size + 1: return [], [], [], []

    for i in range(0, n - window_size - 1, step_size):
        window_df = df.iloc[i:i + window_size + 1]
        target_v = window_df['totalVoltage'].values.astype(np.float32) 
        delta_mile = window_df['accumulatedMileage'].iloc[-1] - window_df['accumulatedMileage'].iloc[0]
        if delta_mile <= 0: continue
        # lumped_t = window_df[lumped_cols].iloc[-1].to_numpy(dtype=np.float32)
        point_t = window_df[point_k_cols].iloc[-1].to_numpy(dtype=np.float32)
        win_avg = window_df[win_avg_cols].mean(axis=0)
        ECR = (window_df['P'].sum() * 10) / (delta_mile + 1e-6)
        v = window_df['vehicleSpeed']
        valid_n = int(v.notna().sum())
        if valid_n == 0:
            v_low = v_mid = v_high = 0.0
        else:
            v_low = ((v >= 0) & (v < 40)).sum() / valid_n
            v_mid = ((v >= 40) & (v < 80)).sum() / valid_n
            v_high = (v >= 80).sum() / valid_n
        v_ratios = np.array([v_low, v_mid, v_high], dtype=np.float32)

        lumped = window_df[lumped_cols].values.astype(np.float32)
        points = np.concatenate([
            np.array([delta_mile], dtype=np.float32), 
            # lumped_t.astype(np.float32), 
            point_t.astype(np.float32), 
            win_avg.astype(np.float32), 
            np.array([ECR], dtype=np.float32), 
            v_ratios.astype(np.float32)
            ])
        
        label_dr5 = np.float32(df['DR_5soc'].iloc[i + window_size] / 60.0)
        if label_dr5 < 0:
            continue

        target_soc = np.float32(df['soc'].iloc[i + window_size]) # ✅ 提取目标soc
        init_v = np.float32(df['totalVoltage'].iloc[i])

        samples.append({
            'lumped': lumped, 
            'point': points, 
            'target_v': target_v, 
            'target_soc': target_soc,
            'label_dr5': label_dr5,
            'init_v': init_v,
            'car': car,
            'seg': seg,
            'window_idx': i
            })
        if compute_global:
            all_lumped_data.append(lumped)
            all_point_data.append(points)
            all_target_data.append((target_v, label_dr5, target_soc, init_v))

    return samples, all_lumped_data, all_point_data, all_target_data

class BatteryRDRFullDataset_SF(Dataset):
    def __init__(self, file_list, window_size=30, step_size=1, 
                 normalize=True, global_stats=None, 
                 compute_global=False, num_workers=1):
        self.samples = []

        self.lumped_cols = [
            'accumulatedMileage', 'totalCurrent', 'vehicleSpeed', 
            'Temp_mean', 'Temp_std', 'Temp_max', 'Temp_min', 
            'soh', 
            ]
        self.point_k_cols = [
            'soc', 'totalVoltage', 'Temp_mean', 'Volt_min', 
            'totalCurrent', 'soe', 'vehicleSpeed', 'soh',
            'totalV1', 'accumulatedMileage', 'Temp_std', 
            ]
        self.win_avg_cols = [
            'Temp_max', 'Temp_min', 'Temp_std', 'vehicleSpeed', 
            'Volt_max',
            ]

        all_lumped_data = []
        all_point_data = []
        all_target_data = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    process_file_SF,
                    f,
                    window_size,
                    step_size,
                    compute_global, 
                    self.lumped_cols,
                    self.point_k_cols,
                    self.win_avg_cols
                ): f for f in file_list
            }
            for future in as_completed(futures):
                samples, l_data, p_data, t_data = future.result()
                self.samples.extend(samples)

                if compute_global:
                    all_lumped_data.extend(l_data)
                    all_point_data.extend(p_data)
                    all_target_data.extend(t_data)

        if normalize:
            if compute_global:
                l_arr = np.concatenate(all_lumped_data, axis=0)
                p_arr = np.stack(all_point_data)

                tv_list, dr5_list, soc_list, init_v_list = zip(*all_target_data)

                tv_arr = np.concatenate(tv_list)
                dr5_arr = np.array(dr5_list)
                soc_arr = np.array(soc_list)
                init_v_arr = np.array(init_v_list)
                
                self.global_stats = {
                    'lumped_min': l_arr.min(0), 'lumped_max': l_arr.max(0),
                    'point_min': p_arr.min(0), 'point_max': p_arr.max(0),
                    'tv_min': tv_arr.min(), 'tv_max': tv_arr.max(),
                    'dr5_min': dr5_arr.min(), 'dr5_max': dr5_arr.max(),
                    'soc_min': soc_arr.min(), 'soc_max': soc_arr.max(),
                    'init_v_min': init_v_arr.min(), 'init_v_max': init_v_arr.max(),
                }

                del l_arr, p_arr
                del tv_arr, dr5_arr, soc_arr, init_v_arr
                del all_lumped_data, all_point_data, all_target_data
                del tv_list, dr5_list, soc_list, init_v_list
                gc.collect()

            else:
                self.global_stats = global_stats

            l_min, l_max = self.global_stats['lumped_min'], self.global_stats['lumped_max']
            p_min, p_max = self.global_stats['point_min'], self.global_stats['point_max']
            tv_min, tv_max = self.global_stats['tv_min'], self.global_stats['tv_max']
            dr5_min, dr5_max = self.global_stats['dr5_min'], self.global_stats['dr5_max']
            soc_min, soc_max = self.global_stats['soc_min'], self.global_stats['soc_max']
            init_v_min, init_v_max = self.global_stats['init_v_min'], self.global_stats['init_v_max']

            l_range = l_max - l_min
            p_range = p_max - p_min
            l_range[l_range == 0] = 1.0
            p_range[p_range == 0] = 1.0

            tv_range = tv_max - tv_min if tv_max != tv_min else 1.0
            dr5_range = dr5_max - dr5_min if dr5_max != dr5_min else 1.0
            soc_range = soc_max - soc_min if soc_max != soc_min else 1.0
            init_v_range = init_v_max - init_v_min if init_v_max != init_v_min else 1.0

            for s in self.samples:
                s['lumped'] = (s['lumped'] - l_min) / l_range
                s['point'] = (s['point'] - p_min) / p_range

                s['target_v'] = (s['target_v'] - tv_min) / tv_range
                s['label_dr5'] = (s['label_dr5'] - dr5_min) / dr5_range
                s['target_soc'] = (s['target_soc'] - soc_min) / soc_range
                s['init_v'] = (s['init_v'] - init_v_min) / init_v_range

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'lumped': torch.FloatTensor(s['lumped']),       # [Window, Feat]
            'point': torch.FloatTensor(s['point']),         # [Point_Feat]
            'target_v': torch.FloatTensor(s['target_v']),   # [Window]
            'label_dr5': torch.FloatTensor([s['label_dr5']]).squeeze(),
            'target_soc': torch.FloatTensor([s['target_soc']]).squeeze(),
            'init_v': torch.FloatTensor([s['init_v']]).squeeze(),
            'car': s['car'],
            'seg': s['seg'],
            'window_idx': s['window_idx']
            }

def make_train_val_datasets(folder_dir, val_ratio=0.2, window_size=30, step_size=1, global_stats=None, num_workers=None):
    if num_workers is None: num_workers = max(1, multiprocessing.cpu_count() - 1)
    all_files = glob.glob(f"{folder_dir}/**/*.pkl", recursive=True)
    train_files, val_files = train_test_split(all_files, test_size=val_ratio, random_state=42)
    
    train_ds = BatteryRDRFullDataset_SF(train_files, window_size, step_size, compute_global=(global_stats is None), global_stats=global_stats, num_workers=num_workers)
    val_ds = BatteryRDRFullDataset_SF(val_files, window_size, step_size, compute_global=False, global_stats=train_ds.global_stats, num_workers=num_workers)
    return train_ds, val_ds, train_ds.global_stats

class LumpedEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, method='transformer', num_layers=2, num_heads=4, max_len=500):
        super().__init__()
        self.method = method.lower()
        
        if self.method == 'mlp':
            self.encoder = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size)
            )
            
        elif self.method == 'lstm':
            self.encoder = nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
                                   batch_first=True, bidirectional=True)
            self.proj = nn.Linear(hidden_size * 2, hidden_size)

        elif self.method == 'transformer':
            self.proj_in = nn.Linear(input_size, hidden_size)
            self.pos_embed = nn.Parameter(torch.zeros(1, max_len, hidden_size))
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=num_heads, 
                batch_first=True, norm_first=True, activation='gelu'
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        B, T, C = x.shape
        
        if self.method == 'mlp':
            return self.encoder(x) # [B, T, H]
            
        elif self.method == 'lstm':
            out, _ = self.encoder(x)
            return self.proj(out) # [B, T, H]
            
        elif self.method == 'transformer':
            x = self.proj_in(x)
            x = x + self.pos_embed[:, :T, :] 
            return self.encoder(x) # [B, T, H]

class BatteryMFT(nn.Module):
    def __init__(self, lumped_size=15, point_size=27, hidden_size=64, encoder_method='transformer'):
        super().__init__()

        self.lumped_encoder = LumpedEncoder(lumped_size, hidden_size, method=encoder_method)
        self.recon_head = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.GELU(), nn.Linear(hidden_size // 2, 1))
        self.phys_fusion = nn.Linear(hidden_size + 1, hidden_size)
        self.soc_rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.soc_head = nn.Linear(hidden_size * 2, 1)
        self.point_proj = nn.Linear(point_size, hidden_size)
        self.fusion_attn = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.fusion_norm = nn.LayerNorm(hidden_size)
        self.fusion_gate = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size * 2), nn.Sigmoid())
        self.fusion_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.dr5_head = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())

    def forward(self, lumped_seq, point_feat, target_v_actual, init_v):
        B, T, _ = lumped_seq.shape

        base_feat = self.lumped_encoder(lumped_seq) # [B, T, H]
        recon_v = init_v.unsqueeze(-1) + self.recon_head(base_feat).squeeze(-1)
        v_expanded = target_v_actual.unsqueeze(-1) # [B, T, 1]
        fused_phys_seq = torch.relu(self.phys_fusion(torch.cat([base_feat, v_expanded], dim=-1))) # [B, T, H]
        soc_out, (h_n, _) = self.soc_rnn(fused_phys_seq)
        soc_pred = self.soc_head(torch.cat([h_n[-2], h_n[-1]], dim=-1)).squeeze(-1)
        p_emb = torch.relu(self.point_proj(point_feat)) # [B, H]
        p_query = p_emb.unsqueeze(1) # [B, 1, H]
        attn_out, _ = self.fusion_attn(p_query, fused_phys_seq, fused_phys_seq)
        context_feat = attn_out.squeeze(1) # [B, H]
        combined = torch.cat([p_emb, context_feat], dim=-1) # [B, H*2]
        gate = self.fusion_gate(combined)
        dr5_pred = self.dr5_head(self.fusion_proj(combined * gate)).squeeze(-1)

        return {"recon_v": recon_v, "soc_pred": soc_pred, "dr5_pred": dr5_pred}

def train_battery_model(
        model, save_dir, train_dataset, val_dataset, mode='dr5_prediction',
        batch_size=32, epochs=100, lr=1e-3, device='cuda', patience=15, 
        grad_clip=1.0, warmup_epochs=5, lambda_recon=0.1, lambda_soc=0.5
    ):    
    os.makedirs(save_dir, exist_ok=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=1e-6)
    
    mse_loss = nn.MSELoss()
    model.to(device)

    best_val_loss = float('inf')
    best_epoch = 0

    print(f"Training [{mode}] mode for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        if epoch <= warmup_epochs:
            curr_lr = lr * epoch / warmup_epochs
            for pg in optimizer.param_groups: pg['lr'] = curr_lr
        else:
            curr_lr = optimizer.param_groups[0]['lr']

        model.train()
        train_metrics = {'total': 0, 'recon': 0, 'soc': 0, 'dr5': 0}
        
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            
            lumped = batch['lumped'].to(device)
            point = batch['point'].to(device)
            target_v = batch['target_v'].to(device)
            target_soc = batch['target_soc'].to(device)
            label_dr5 = batch['label_dr5'].to(device)
            init_v = batch['init_v'].to(device) 

            out = model(lumped, point, target_v, init_v)

            loss_recon = mse_loss(out['recon_v'], target_v)
            loss_soc = mse_loss(out['soc_pred'], target_soc)
            loss_dr5 = mse_loss(out['dr5_pred'], label_dr5)

            if mode == 'pretrain': 
                loss = loss_recon
            elif mode == 'soc_estimation':
                loss = loss_soc + lambda_recon * loss_recon
            else: # DR5 
                loss = loss_dr5 + lambda_soc * loss_soc + lambda_recon * loss_recon

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        model.eval()
        val_metrics = {'total': 0, 'recon': 0, 'soc': 0, 'dr5': 0}
        with torch.no_grad():
            for batch in val_loader:
                lumped = batch['lumped'].to(device); point = batch['point'].to(device)
                target_v = batch['target_v'].to(device); target_soc = batch['target_soc'].to(device)
                label_dr5 = batch['label_dr5'].to(device); init_v = batch['init_v'].to(device)

                out = model(lumped, point, target_v, init_v)
                
                v_recon = mse_loss(out['recon_v'], target_v)
                v_soc = mse_loss(out['soc_pred'], target_soc)
                v_dr5 = mse_loss(out['dr5_pred'], label_dr5)

                if mode == 'pretrain': v_total = v_recon
                elif mode == 'soc_estimation': v_total = v_soc + lambda_recon * v_recon
                else: v_total = v_dr5 + lambda_soc * v_soc + lambda_recon * v_recon
                
        num_train, num_val = len(train_dataset), len(val_dataset)
        avg_train_loss = train_metrics['total'] / num_train
        avg_val_loss = val_metrics['total'] / num_val

        if epoch > warmup_epochs: scheduler.step()

        print(f"Epoch {epoch:03d} | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_{mode}.pth"))
        elif epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    return model

def make_test_dataset_car_folder(
    folder_dir,
    global_stats,
    car_type='P',
    window_size=30,
    step_size=1,
    normalize=True,
    num_workers=None
):

    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    car_dirs = sorted(glob.glob(f"{folder_dir}/discharge_segment_{car_type}*"))

    if len(car_dirs) == 0:
        raise ValueError(
            f"No matching vehicle folders found: {folder_dir}/discharge_segment_{car_type}*"
        )

    all_files = []
    for car_dir in car_dirs:
        files = sorted(glob.glob(f"{car_dir}/*.pkl"))
        all_files.extend(files)
    print(f"Find {len(all_files)} pkl files")

    dataset = BatteryRDRFullDataset_SF(
        all_files,
        window_size=window_size,
        step_size=step_size,
        normalize=normalize,
        global_stats=global_stats,
        compute_global=False,
        num_workers=num_workers
    )

    print(f"✅ Dataset created, {len(dataset)} windows in total.")

    return dataset

def evaluate_multitask(
    model, dataset, device, global_stats, batch_size=32, 
    save_dir="./results/dr5_prediction"
):
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    model.eval()

    print(f"Starting Evaluation (Device: {device})")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_pred_v, all_label_v = [], []
    all_pred_soc, all_label_soc = [], []
    all_pred_dr5, all_label_dr5 = [], []
    all_car, all_seg, all_win = [], [], []

    tv_min, tv_max = global_stats['tv_min'], global_stats['tv_max']
    soc_min, soc_max = global_stats['soc_min'], global_stats['soc_max']

    tv_range = tv_max - tv_min if tv_max != tv_min else 1.0
    soc_range = soc_max - soc_min if soc_max != soc_min else 1.0

    start_infer = time.time()
    
    with torch.no_grad():
        for batch in loader:
            lumped = batch['lumped'].to(device)
            point = batch['point'].to(device)            
            init_v = batch['init_v'].to(device) 
            target_v = batch['target_v'].to(device)
            target_soc = batch['target_soc'].cpu().numpy()
            label_dr5 = batch['label_dr5'].cpu().numpy() * 60

            car = batch['car']
            seg = batch['seg']
            win_id = batch['window_idx']

            out = model(lumped, point, target_v, init_v)

            pred_v = out['recon_v'].cpu().numpy()   # [B, T]
            gt_v = target_v.cpu().numpy()
            pred_v = pred_v * tv_range + tv_min
            gt_v = gt_v * tv_range + tv_min
            
            pred_soc_0 = out['soc_pred'].cpu().numpy()  # [B]
            pred_soc_0 = pred_soc_0 * soc_range + soc_min
            tgt_soc = target_soc * soc_range + soc_min
            pred_soc = np.clip(pred_soc_0, 0, 100)

            pred_dr5_0 = out['dr5_pred'].cpu().numpy() * 60 # [B]
            pred_dr5 = np.clip(pred_dr5_0, 0, 60)

            all_pred_v.extend(pred_v)
            all_label_v.extend(gt_v)
            all_pred_soc.extend(pred_soc)
            all_label_soc.extend(tgt_soc)
            all_pred_dr5.extend(pred_dr5)
            all_label_dr5.extend(label_dr5)

            all_car.extend(car)
            all_seg.extend(seg)
            all_win.extend(np.array(win_id).tolist())

    end_infer = time.time()
    infer_time = end_infer - start_infer
    print('Infer Time: ',infer_time )

    df = pd.DataFrame({
        'v_label': [v.tolist() for v in all_label_v],
        'v_recon': [v.tolist() for v in all_pred_v],
        'soc_label': all_label_soc,
        'soc_pred': all_pred_soc,
        'dr5_label': all_label_dr5,
        'dr5_pred': all_pred_dr5,
        'car': all_car,
        'seg': all_seg,
        "win_id": all_win
    })

    df = df[df['dr5_label'] > 0]
    df_save_path = os.path.join(save_dir, f"result_multitask_all.pkl")
    pd.to_pickle(df, df_save_path)

    return df



