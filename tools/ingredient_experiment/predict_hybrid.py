"""Run frozen experimental ingredient-region selection on a local image."""
import argparse
import json
import sys
from pathlib import Path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('image',type=Path)
    parser.add_argument('--repo',type=Path,default=Path(r'C:\KDT2604\Git_Jarviseo_Kw01\JARVISEO'))
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('hybrid_config.json'))
    parser.add_argument('--output',type=Path,default=Path('hybrid_prediction'))
    args=parser.parse_args()
    import cv2
    import numpy as np
    import torch
    import easyocr
    from ultralytics import YOLO
    from PIL import Image,ImageDraw
    from hybrid_region import choose_region
    sys.path.insert(0,str(args.repo/'src'))
    from jarviseo.nutrition.region import word_boxes,find_region
    torch.set_num_threads(4)
    image=cv2.imdecode(np.fromfile(args.image,dtype=np.uint8),cv2.IMREAD_COLOR)
    if image is None: raise ValueError('Cannot read input image')
    reader=easyocr.Reader(['ko','en'],gpu=False,download_enabled=False,verbose=False)
    words=word_boxes(image,reader)
    base,anchors=find_region(words,image.shape)
    model=YOLO(str(args.weights))
    if model.names!={0:'ingredient_panel'}:raise ValueError('Expected ingredient_panel model')
    prediction=model.predict(image,conf=0.25,imgsz=640,device='cpu',verbose=False)[0]
    proposals=[{'bbox':b,'conf':c} for b,c in zip(prediction.boxes.xyxy.tolist(),prediction.boxes.conf.tolist())]
    config=json.loads(args.config.read_text(encoding='utf-8'))['selected']['params']
    chosen,source=choose_region([base.x1,base.y1,base.x2,base.y2],proposals,
        [[b.x1,b.y1,b.x2,b.y2,t] for b,t in words],image.shape,**config)
    args.output.mkdir(parents=True,exist_ok=True)
    result={'bbox':chosen,'source':source,'candidate_count':len(proposals),'anchors':anchors,'config':config}
    (args.output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    original=Image.open(args.image).convert('RGB')
    original.crop(tuple(map(round,chosen))).save(args.output/'crop.png')
    ImageDraw.Draw(original).rectangle(chosen,outline='red',width=4)
    original.save(args.output/'overlay.png')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
