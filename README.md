Ссылка на GitHub:

https://github.com/ilvoron/SpectralGPT_Classification

Ссылка на Weights & Biases:

https://wandb.ai/il-voron-moscow-aviation-institute/eurosat_spectralgpt_finetune

Ссылка на предобученные веса (ложить в папку `pretrained_models`):

https://disk.yandex.ru/d/X8wLPJ_sHZOEkA

Ссылка на точечно обученные веса (ложить в папку `checkpoints`):

https://disk.yandex.ru/d/DPLbqrLWb2sMCw

Ссылка на датасет EuroSAT:

https://github.com/phelber/EuroSAT (или сразу на скачивание: https://madm.dfki.de/files/sentinel/EuroSATallBands.zip)

---

Finetune:

```
python main_finetune.py --batch_size 8 --accum_iter 16 --blr 0.0002 --epochs 150 --num_workers 16 --input_size 128 --patch_size 8 --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dataset_type euro_sat --dropped_bands 10 --train_path txt_file/train_euro_result.txt --test_path txt_file/val_euro_result.txt --output_dir ./checkpoints --log_dir ./checkpoints --finetune ./pretrained_models/SpectralGPT+.pth --nb_classes 10 --save_every 1
```

---

Запуск интерфейса:

```
streamlit run app.py
```
