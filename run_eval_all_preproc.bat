@echo off
cd /d "C:\Users\sebag\OneDrive\ASUS DAN\Documents\4 ECOLE\M1 MLSD\15 PPD\SEP-CVDL-main"

call venv\Scripts\activate

python eval_saved_model_preproc.py --model gimefive --dataset fer2013
python eval_saved_model_preproc.py --model gimefiveres --dataset fer2013
python eval_saved_model_preproc.py --model resnet18 --dataset fer2013
python eval_saved_model_preproc.py --model resnet34 --dataset fer2013
python eval_saved_model_preproc.py --model vgg --dataset fer2013

pause