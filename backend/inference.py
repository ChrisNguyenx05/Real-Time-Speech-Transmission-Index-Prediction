import torch
import numpy as np
import joblib
from backend.model_df import TwoBranchRayNet
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

scaler_X = joblib.load(os.path.join(BASE_DIR, "model/scaler_X.pkl"))
scaler_y = joblib.load(os.path.join(BASE_DIR, "model/scaler_y.pkl"))
surrogate_model = joblib.load(os.path.join(BASE_DIR, "model/surrogate_model.pkl"))

surrogate_model.n_jobs = 1 
for estimator in surrogate_model.estimators_:
    estimator.set_params(n_jobs=1)

# Load model
model_ann = TwoBranchRayNet(direct_dim=5, ray_dim=6, ray_emb_dim=32, output_dim=3)
model_ann.load_state_dict(
    torch.load(os.path.join(BASE_DIR, "model/model.pth"), map_location="cpu")
)
model_ann.eval()

CSV_COLUMN_ORDER = [
    'out:L01S', 'out:L02S', 'out:L03S', 'out:L04S', 'out:L05S', 'out:L06S', 'out:L07S', 'out:L08S', 'out:L09S', 'out:L10S', 'out:L11S', 'out:L12S', 'out:L13S', 'out:L14S',
    'out:L01R', 'out:L02R', 'out:L03R', 'out:L04R', 'out:L05R', 'out:L06R', 'out:L07R', 'out:L08R', 'out:L09R', 'out:L10R', 'out:L11R', 'out:L12R', 'out:L13R', 'out:L14R',
    'out:LSR', 'out:r01', 'out:r02', 'out:r03', 'out:r04', 'out:r05', 'out:r06', 'out:r07', 'out:r08', 'out:r09', 'out:r10', 'out:r11', 'out:r12', 'out:r13', 'out:r14',
    'out:P01x', 'out:P02x', 'out:P03x', 'out:P04x', 'out:P05x', 'out:P06x', 'out:P07x', 'out:P08x', 'out:P09x', 'out:P10x', 'out:P11x', 'out:P12x', 'out:P13x', 'out:P14x',
    'out:P01y', 'out:P02y', 'out:P03y', 'out:P04y', 'out:P05y', 'out:P06y', 'out:P07y', 'out:P08y', 'out:P09y', 'out:P10y', 'out:P11y', 'out:P12y', 'out:P13y', 'out:P14y',
    'out:P01z', 'out:P02z', 'out:P03z', 'out:P04z', 'out:P05z', 'out:P06z', 'out:P07z', 'out:P08z', 'out:P09z', 'out:P10z', 'out:P11z', 'out:P12z', 'out:P13z', 'out:P14z',
    'out:Sx', 'out:Sy', 'out:Rx', 'out:Ry'
]



def split_direct_ray(X_array):
    direct_all = np.concatenate([
        X_array[:, 28:29],   # out:LSR
        X_array[:, 85:89]    # out:Sx, out:Sy, out:Rx, out:Ry
    ], axis=1)

    Ls  = X_array[:, 0:14]    
    Lr  = X_array[:, 14:28]   
    r   = X_array[:, 29:43]   
    Px  = X_array[:, 43:57]   
    Py  = X_array[:, 57:71]   
    Pz  = X_array[:, 71:85]   

    # Gom thành shape (N, 14, 6)
    ray_all = np.stack([Ls, Lr, r, Px, Py, Pz], axis=2)   
    return direct_all, ray_all

def predict_sti_v2(Sx, Sy, Rx, Ry):
    # Bước 1: Tính toán
    LSR = np.sqrt((Sx - Rx)**2 + (Sy - Ry)**2)
    
    # ----------------------------------------------------
    # FIX LỖI Ở ĐÂY: Trực tiếp đưa 5 biến vào thành mảng 2D
    # (Không gọi hàm _feature_engineering_api nữa)
    X_for_surrogate = np.array([[Sx, Sy, Rx, Ry, LSR]])
    # ----------------------------------------------------
    
    # Bước 2: Dùng Surrogate đẻ ra 84 biến
    y_ray_pred = surrogate_model.predict(X_for_surrogate)[0]
    
    # Bước 3: Ghép 89 biến theo thứ tự file CSV
    direct_vals = {'out:Sx': Sx, 'out:Sy': Sy, 'out:Rx': Rx, 'out:Ry': Ry, 'out:LSR': LSR}
    ray_cols = [c for c in CSV_COLUMN_ORDER if c not in direct_vals]
    
    full_dict = {col: val for col, val in zip(ray_cols, y_ray_pred)}
    full_dict.update(direct_vals)
    
    input_89 = np.array([[full_dict[col] for col in CSV_COLUMN_ORDER]])
    
    # Bước 4: Scale 89 biến như lúc train X_train
    X_scaled = scaler_X.transform(input_89)
    
    # Bước 5: Cắt tensor theo đúng hàm trên Colab
    direct_arr, ray_arr = split_direct_ray(X_scaled)
    
    # Bước 6: Chuyển sang PyTorch Tensor và Inference
    direct_tensor = torch.tensor(direct_arr, dtype=torch.float32)
    ray_tensor = torch.tensor(ray_arr, dtype=torch.float32)
    
    with torch.no_grad():
        # Gọi model gốc với 2 biến truyền vào
        output = model_ann(direct_tensor, ray_tensor)
        
    final_sti = scaler_y.inverse_transform(output.numpy())
    return final_sti[0]