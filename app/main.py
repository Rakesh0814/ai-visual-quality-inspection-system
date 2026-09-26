from pathlib import Path
import io,json,uuid
from datetime import datetime,timezone
from PIL import Image
from fastapi import FastAPI,File,UploadFile,HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from .config import settings
from .predictor import predict
BASE=Path(__file__).resolve().parent
app=FastAPI(title='AI Visual Quality Inspection',version='2.0')
app.mount('/static',StaticFiles(directory=BASE/'static'),name='static')
@app.get('/',response_class=HTMLResponse)
def home(): return (BASE/'templates'/'index.html').read_text(encoding='utf-8')
@app.get('/health')
def health(): return {'status':'ok','model_available':settings.model_path.exists(),'threshold':settings.defect_threshold}
def add_history(r):
    item={**r,'created_at':datetime.now(timezone.utc).isoformat()}
    with settings.history_path.open('a',encoding='utf-8') as f:f.write(json.dumps(item)+'\n')
    return item
@app.get('/api/history')
def history(limit:int=20):
    if not settings.history_path.exists(): return []
    rows=[]
    for line in settings.history_path.read_text(encoding='utf-8').splitlines():
        try:rows.append(json.loads(line))
        except:pass
    return list(reversed(rows[-max(1,min(limit,100)):]))
@app.post('/api/inspect')
async def inspect(file:UploadFile=File(...)):
    if not file.content_type or not file.content_type.startswith('image/'): raise HTTPException(400,'Upload an image file.')
    try:image=Image.open(io.BytesIO(await file.read())).convert('RGB')
    except Exception as e: raise HTTPException(400,f'Invalid image: {e}')
    try:p,m=predict(image)
    except RuntimeError as e: raise HTTPException(503,str(e))
    label='DEFECT' if p>=settings.defect_threshold else 'PASS'; conf=p if label=='DEFECT' else 1-p
    r={'inspection_id':str(uuid.uuid4()),'filename':file.filename or 'image','label':label,'confidence':round(conf,4),'defect_probability':round(p,4),'backend':'pytorch','model':m.get('architecture','MobileNetV3-Small'),'dataset':m.get('dataset','MVTec AD bottle'),'validation_accuracy':m.get('validation_accuracy')}
    add_history(r); return r
