# MAG-ReID

an official code for MAG-ReID

## To-Do List

* [x] Release the training code.
* [x] Release the evaluation code.
* [x] Release configuration files.
* [x] Add visualization results.
* [x] Release pretrained models.

## Weights and Training logs
[ORBench](https://pan.quark.cn/s/1b97789e8355?pwd=J3U1)

[CUHK-PEDES](https://pan.quark.cn/s/f2978c76ef13)

[PKU-Sketch](https://pan.quark.cn/s/9ff6ba82342b)

## Training & Evaluation

### Training

To train MAG-ReID, run:

```bash
python train.py # all the settings are in utils/options.py
```

Please modify the configuration file according to the target dataset and experimental setting.

### Evaluation

To evaluate a trained model, run:

```bash
python test.py 
```

The evaluation results, including **Rank-1** and **mAP**, will be reported automatically.


## framework

![framework](figures/main.png)


## results

### ORBench

![img](figures/result_orbench.png)

### PKU-Sketch

![img](figures/result_pku.png)

### CUHK-PEDES, ICFG-PEDES, RSTP-reid

![img](figures/result_three_pedes.png)

### RGBNT201

![img](figures/result_rgbnt.png)

## Visulization

![img](figures/vis_tsne.png)

![img](figures/vis_grad_cam.png)

![img](figures/vis_rank.png)

## `Acknowledgement`

Some components of this code implementation are adopted from [IRRA](https://github.com/anosorae/IRRA) and [Reid5o](https://github.com/Zplusdragon/ReID5o_ORBench) .We sincerely appreciate for their contributions.

## Concat

If you have any question, please feel free to contact me: [dingjin1998@std.uestc.edu.cn](dingjin1998@std.uestc.edu.cn)
