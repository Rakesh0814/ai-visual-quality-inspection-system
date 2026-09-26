from functools import lru_cache
import json, torch
from torchvision import transforms
from .config import settings
from training.pytorch_model import QualityClassifier
@lru_cache(maxsize=1)
def model():
    if not settings.model_path.exists(): return None
    m=QualityClassifier(pretrained=False); m.load_state_dict(torch.load(settings.model_path,map_location='cpu')); m.eval(); return m
@lru_cache(maxsize=1)
def meta():
    if settings.meta_path.exists():
        try:return json.loads(settings.meta_path.read_text(encoding='utf-8'))
        except Exception:return {}
    return {}
def predict(image):
    m=model()
    if m is None: raise RuntimeError('No trained model found. Run setup_train_run_windows.bat first.')
    tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    x=tf(image).unsqueeze(0)
    with torch.no_grad(): p=torch.sigmoid(m(x)).item()
    return float(p),meta()
