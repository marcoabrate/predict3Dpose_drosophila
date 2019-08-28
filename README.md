## Predicting Drosophila melanogaster 3D pose from the 2D pose

This is the code for the semester project

Predicting Drosophila melanogaster 3D pose from the 2D pose
Student: Marco Pietro Abrate
Supervisor: Semih Gunel
EPFL Lab: [Neuroengineering Laboratory (RAMDYA)](https://ramdya-lab.epfl.ch/)

The code in this repository was written by
[Marco Pietro Abrate](https://github.com/marcoabrate)

The aim of this project is to reduce the number of cameras mounted on the platform that are used by [DeepFly3D](https://github.com/NeLy-EPFL/DeepFly3D) to predict the 3D pose of Drosophila melanogaster.

### Dependencies

* [tensorflow](https://www.tensorflow.org/) 1.0 or later

### First of all
Clone this repository and get the data. The data must be downloaded in the right folders. The name of the training and testing files can be found in the [project report](https://github.com/marcoabrate/predict3Dpose_drosophila/blob/master/bioproject_report.pdf).

```bash
git clone https://github.com/marcoabrate/predict3Dpose_drosophila
cd 3d-pose-baseline
mkdir flydata_train
mkdir flydata_test
```

### Show results

For showing the results of the final model, you have to go in the `3d-pose-baseline` folder, and run

`python flysrc/predict_3dpose.py --residual --batch_norm --dropout 0.5 --max_norm --epochs 200 --origin_bc --camera_frame --test --load 266800`

This will produce a visualization similar to [this video](https://www.youtube.com/watch?v=N31742fBUZg).

![Visualization example](/imgs/visualization_example.png)

### Training

To train a model from scratch, run (from the folder `3d-pose-baseline`):

`python flysrc/predict_3dpose.py --residual --batch_norm --dropout 0.5 --max_norm --epochs 200 --origin_bc --camera_frame --train_dir "new_model"`
