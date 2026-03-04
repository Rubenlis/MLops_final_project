<<<<<<< HEAD
# MLops_final_project
=======
# MLops_final_project

Installer les bibli notées dans requirement.txt  **Python 1.10**

pytest : pytest -q

Test tres rapide de la training pipeline : 
        python training/train.py `
  --train_json data/train-v2.0.json `
  --val_json data/dev-v2.0.json `
  --subset_train 1000 --subset_val 200 `
  --epochs 1 --batch_size 8 --max_len 128
>>>>>>> 61a7312 (feat: initial commit with training and evaluation pipeline)
