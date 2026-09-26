from pathlib import Path
import json,random,shutil,torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from huggingface_hub import snapshot_download
from training.pytorch_model import QualityClassifier
ROOT=Path(__file__).resolve().parents[1]; CACHE=ROOT/'data'/'hf_mvtec'; OUT=ROOT/'dataset'; MODEL=ROOT/'app'/'models'/'pytorch_quality_model.pt'; META=ROOT/'app'/'models'/'pytorch_quality_model.json'
random.seed(42); torch.manual_seed(42)
def imgs(p): return [x for x in p.rglob('*') if x.is_file() and x.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.webp'}]
def copy_many(files,dst,prefix):
    dst.mkdir(parents=True,exist_ok=True)
    for i,s in enumerate(files): shutil.copy2(s,dst/f'{prefix}_{i:04d}{s.suffix.lower()}')
def main():
    print('[1/4] Downloading MVTec AD bottle data...')
    base=Path(snapshot_download(repo_id='foersben/mvtec-ad',repo_type='dataset',allow_patterns=['bottle/train/good/*','bottle/test/good/*','bottle/test/broken_large/*','bottle/test/broken_small/*','bottle/test/contamination/*'],local_dir=CACHE))/'bottle'
    print('[2/4] Preparing binary dataset...')
    if OUT.exists(): shutil.rmtree(OUT)
    gt=imgs(base/'train'/'good'); gv=imgs(base/'test'/'good'); df=[]
    for n in ['broken_large','broken_small','contamination']: df+=imgs(base/'test'/n)
    random.shuffle(gt); random.shuffle(gv); random.shuffle(df)
    cut=max(1,int(len(df)*.8)); dt,dv=df[:cut],df[cut:]; gt=gt[:min(len(gt),len(dt)*2)]; gv=gv[:min(len(gv),max(len(dv),5))]
    copy_many(gt,OUT/'train'/'good','good'); copy_many(dt,OUT/'train'/'defect','defect'); copy_many(gv,OUT/'val'/'good','good'); copy_many(dv,OUT/'val'/'defect','defect')
    print(f'train good={len(gt)} defect={len(dt)} | val good={len(gv)} defect={len(dv)}')
    train_tf=transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]); val_tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    tr=datasets.ImageFolder(OUT/'train',transform=train_tf); va=datasets.ImageFolder(OUT/'val',transform=val_tf)
    di_tr=tr.class_to_idx['defect']; di_va=va.class_to_idx['defect']
    def collate(di):
        def fn(batch):
            x,y=zip(*batch); return torch.stack(x),torch.tensor([1.0 if int(v)==di else 0.0 for v in y])
        return fn
    tl=DataLoader(tr,batch_size=16,shuffle=True,num_workers=0,collate_fn=collate(di_tr)); vl=DataLoader(va,batch_size=16,shuffle=False,num_workers=0,collate_fn=collate(di_va))
    print('[3/4] Training MobileNetV3-Small...')
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); m=QualityClassifier(pretrained=True).to(dev)
    for p in m.backbone.features.parameters(): p.requires_grad=False
    opt=torch.optim.AdamW(filter(lambda p:p.requires_grad,m.parameters()),lr=1e-3); lossfn=nn.BCEWithLogitsLoss(); best=0.0
    for e in range(1,7):
        m.train(); total_loss=0
        for x,y in tl:
            x,y=x.to(dev),y.to(dev); opt.zero_grad(); z=m(x); loss=lossfn(z,y); loss.backward(); opt.step(); total_loss+=loss.item()*x.size(0)
        m.eval(); c=t=0
        with torch.no_grad():
            for x,y in vl:
                x,y=x.to(dev),y.to(dev); pred=(torch.sigmoid(m(x))>=.5).float(); c+=(pred==y).sum().item(); t+=y.numel()
        acc=c/max(t,1); print(f'epoch {e}/6 loss={total_loss/max(len(tr),1):.4f} val_acc={acc:.3f}')
        if acc>=best: best=acc; MODEL.parent.mkdir(parents=True,exist_ok=True); torch.save(m.state_dict(),MODEL)
    META.write_text(json.dumps({'architecture':'MobileNetV3-Small','framework':'PyTorch','dataset':'MVTec AD - bottle','task':'binary good vs defect classification','validation_accuracy':round(best,4),'note':'Portfolio supervised split; not official MVTec benchmark protocol.'},indent=2),encoding='utf-8')
    print('[4/4] DONE'); print('Saved:',MODEL); print('Best validation accuracy:',round(best,3))
if __name__=='__main__': main()
