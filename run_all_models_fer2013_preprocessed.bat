@echo off
cd /d "C:\Users\sebag\OneDrive\ASUS DAN\Documents\4 ECOLE\M1 MLSD\15 PPD\SEP-CVDL-main"

call venv\Scripts\activate

echo ========================================
echo Lancement de gimefive sur fer2013 preprocessed
echo ========================================
python train_eval_preprocessed.py --model gimefive --dataset fer2013 --epochs 80 --batch_size 16 --lr 0.001

echo ========================================
echo Lancement de gimefiveres sur fer2013 preprocessed
echo ========================================
python train_eval_preprocessed.py --model gimefiveres --dataset fer2013 --epochs 80 --batch_size 16 --lr 0.001

echo ========================================
echo Lancement de resnet18 sur fer2013 preprocessed
echo ========================================
python train_eval_preprocessed.py --model resnet18 --dataset fer2013 --epochs 80 --batch_size 16 --lr 0.001

echo ========================================
echo Lancement de resnet34 sur fer2013 preprocessed
echo ========================================
python train_eval_preprocessed.py --model resnet34 --dataset fer2013 --epochs 80 --batch_size 16 --lr 0.001

echo ========================================
echo Lancement de vgg sur fer2013 preprocessed
echo ========================================
python train_eval_preprocessed.py --model vgg --dataset fer2013 --epochs 80 --batch_size 16 --lr 0.001

echo ========================================
echo Tous les entrainements preprocessed sont termines
echo ========================================
pause