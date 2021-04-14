## Predicting Drosophila melanogaster 3D pose from the 2D pose

This is the code for the semester project

_Predicting Drosophila melanogaster 3D pose from the 2D pose_

_Student:_ Marco Pietro Abrate

_Supervisor:_ Semih Gunel

_EPFL Lab:_ [Neuroengineering Laboratory (RAMDYA)](https://ramdya-lab.epfl.ch/)

The aim of this project is to reduce the number of cameras mounted on the platform that are used by [DeepFly3D](https://github.com/NeLy-EPFL/DeepFly3D) to predict the 3D pose of _Drosophila melanogaster_.

Please consider reading the file [bioproject_report.pdf](https://github.com/marcoabrate/predict3Dpose_drosophila/blob/master/bioproject_report.pdf) for a detailed explanation of this project. 

### Dependencies (Python 3.7)

* tensorflow 1.0 or later
* matplotlib

### First of all
Clone this repository and get the data. The data must be downloaded in the right folders. The name of the training and testing files can be found in the [project report](https://github.com/marcoabrate/predict3Dpose_drosophila/blob/master/bioproject_report.pdf), Appendix A.

```bash
git clone https://github.com/marcoabrate/predict3Dpose_drosophila
cd 3d-pose-baseline
mkdir flydata_train
mkdir flydata_test
```

### Show results

For showing the results of the final model, you have to go in the `3d-pose-baseline` folder, and run

`python flysrc/predict_3dpose.py --residual --batch_norm --dropout 0.5 --max_norm --epochs 200 --origin_bc --camera_frame --test --load 266800`

This will produce an animation similar to [this video](https://www.youtube.com/watch?v=N31742fBUZg) and a visualization similar to this:

![Visualization example](/images/visualization_example.png)

### Training

To train a model from scratch, run (always from the `3d-pose-baseline` folder):

`python flysrc/predict_3dpose.py --residual --batch_norm --dropout 0.5 --max_norm --epochs 200 --origin_bc --camera_frame --train_dir "new_model"`
