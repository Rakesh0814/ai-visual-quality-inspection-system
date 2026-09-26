from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
    image_size:int=224
    defect_threshold:float=0.50
    model_path:Path=Path('app/models/pytorch_quality_model.pt')
    meta_path:Path=Path('app/models/pytorch_quality_model.json')
    history_path:Path=Path('data/inspection_history.jsonl')
    model_config=SettingsConfigDict(env_file='.env',extra='ignore')
settings=Settings(); settings.model_path.parent.mkdir(parents=True,exist_ok=True); settings.history_path.parent.mkdir(parents=True,exist_ok=True)
