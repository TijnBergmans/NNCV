# 5LSM0: Neural Networks for Computer Vision - Final Assignment

## Information

Name: Tijn Bergmans
ID: 1245869
E-mail: t.bergmans@student.tue.nl
Github username: TijnBergmans
Codalab username: TijnBergmans

## Important note!

My original baseline model was the U-Net model from the TUE-VCA/NNCV Github. However, due to the maximum time in the slurm jobscript training was cut short. Therefor the score of my baseline model does not match that of the TA baseline, despite the models being identical. 

After discussing this with the TA's, I intended to re-train the baseline model properly and re-submit it under the peak-performance to obtain the necessary metrics, after finishing the other models. 

However, as the credits for Snellius ran out this weekend, I was unable to do so. 

Considering I kept my baseline model and training script identical to the baseline model provided in the TA github, I have used the values from the codalab server for the TA baseline for comparison to my other models. I'm very sorry for not having my own proper baseline measurement, but considering Snellius was unavailable to run a proper experiment, I did not see any other way to obtain the necessary baseline measurements for comparison. 

I sincerely apologize. If possible at a later time, I will re-run the training of the baseline model myself and if necessary update the values.

## Contents

The baseline, hybrid and Mask2Former inspired models as referenced in my paper are provided in the files model, model_medium and model_large respectively. Their respective training scripts are provided in the files train, train_medium and train_large. 

Note that the train_large file includes pre-training pipeline on the Cityscapes coarse dataset. This was an experiment from earlier on in the process which proved ineffective and can be disregarded, as it was not used to obtain the results.

main.sh contains the arguments for the training of the baseline, identical to the file supplied in the TA github. The main_v2.sh contains the settings for the transformer models. The jobscript_slurm.sh should be changed to call this file if the medium or large models are used.

## Libraries

The torch and torchvision libraries are required to run the models. Additionally, the medium and large model use pre-trained weights, which need to be downloaded, but this will happen automatically.

## Using the models

The outputs of the baseline and medium model can immediately be converted to a segmentation map using a softmax function. Note however, that the large model outputs a dictionary instead of a single tensor. The output to construct the segmentation map in the same way as for the other models can be retrieved through output['segmentation'].