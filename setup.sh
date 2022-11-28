pip install -r requirements.txt

gdown https://drive.google.com/u/0/uc?id=1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx
mkdir -p waveglow/pretrained_model/
mv waveglow_256channels_ljs_v2.pt waveglow/pretrained_model/waveglow_256channels.pt

git clone https://github.com/xcmyz/FastSpeech.git
mv FastSpeech/audio src/
mv FastSpeech/waveglow/* src/waveglow/
mv FastSpeech/utils.py src/
mv FastSpeech/glow.py src/

wget -O checkpoint_36000.pth.tar https://api.wandb.ai/files/i_vainn/fastspeech2/2vntjxu3/model_new/checkpoint_36000.pth.tar?_gl=1*lmhs4i*_ga*MTcxMTgzMjg1OC4xNjY3NzM0NjAx*_ga_JH1SJHJQXJ*MTY2OTYyMzY2MC4xOC4xLjE2Njk2MjM2NzUuNDUuMC4w