# LearningToSeparate

This repository contains python implementation of our paper "Attention-Driven Projections for Soundscape Classification".

## Getting Started

These instructions will help you to run python programs in sequence.

### Steps to decompose the audio files into the foreground and the background using rpca

1. Download or clone this repository into local system.
2. Download DCASE 2017 ASC(task 1) development dataset and extract all the zip files into single folder.
3. Copy all the extracted wav files to folder "<path_to_repo_download>/LearningToSeparate/dataset/stereo/"
4. Navigate to "<path_to_repo_download>/LearningToSeparate/dataset/" and run python program "sterio2mono.py".
```
cd <path_to_repo_download>/LearningToSeparate/dataset/
python sterio2mono.py
```
4. Run the following command to copy the audio files into respective class folders.
```
python copyfiles.py
```
5. Run the following matlab program to decompose all the audio files into foreground and background.

  This Matlab code is borrowed from the authors of paper "[Singing-Voice Separation From Monaural Recordings Using Robust Principal Component Analysis]   (http://posenhuang.github.io/papers/RPCA_Separation_ICASSP2012.pdf)," ICASSP 2012.
  For more information, please check: https://sites.google.com/site/singingvoiceseparationrpca/
```
cd ../rpca
matlab -nodisplay -r readfilenames
```
6. Run the following commands to copy the foreground and the background into seperate folders.
```
cp -r example/output/*_E.wav ../dataset/rpca_out_foreground/
cp -r example/output/*_A.wav ../dataset/rpca_out_background/
```
7. Run the following commands to remove substring "_A" and "_E" from all the file names.
```
cd ../dataset/rpca_out_foreground/
rename 's/_E//g' *.wav

cd ../rpca_out_background/
rename 's/_A//g' *.wav
```
8. Run the following command to copy the audio files into respective class folders.
```
cd ..
python copyfiles_fg.py
python copyfiles_bg.py
```

### Steps to extract L3-Net features and perform other experiments on L3-Net features

1. Run the following program to generate L3-Net features.
```
python l3net/code/l3_feat.py

```
2. Run the following program to generated 4 fold numpy arrays for audio, foreground and background.
```
python l3net/code/test_copy_audio.py
python l3net/code/test_copy_fg.py
python l3net/code/test_copy_bg.py

python l3net/code/train_copy_audio.py
python l3net/code/train_copy_fg.py
python l3net/code/train_copy_bg.py
```
4. Run the following program to generate basis for foreground and background.
```
python l3net/code/pca_full_fg.py
python l3net/code/pca_full_bg.py
```
5. Run the following program to get baseline system accuracy.
```
python l3net/code/svm_audio.py
python l3net/code/svm_bg.py
python l3net/code/svm_fg.py
```
6. Run the following program to prepare data for self-attention model using "C" Subspace projection.
```
python l3net/code/data_prep_att_fg.py
python l3net/code/data_prep_att_bg.py
```

7. Run the following program to train self-attention model to combine 'C' projected embeddings into single embedding. The amount of foregroud or background to be suppressed can be changed by varying p and q values in the program. 
```
python l3net/code/model_att_fold1_fg.py
python l3net/code/model_att_fold2_fg.py
python l3net/code/model_att_fold3_fg.py
python l3net/code/model_att_fold4_fg.py

python l3net/code/model_att_fold1_bg.py
python l3net/code/model_att_fold2_bg.py
python l3net/code/model_att_fold3_bg.py
python l3net/code/model_att_fold4_bg.py
```
8. Run the following program to extract meta-embeddings and classify using SVM classifier. 
```
python l3net/code/feat_fg.py
python l3net/code/feat_bg.py
```
