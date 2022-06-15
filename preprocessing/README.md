# Original Datset
https://magenta.tensorflow.org/datasets/nsynth
Introduced in Jesse Engel, Cinjon Resnick, Adam Roberts, Sander Dieleman, Douglas Eck,
  Karen Simonyan, and Mohammad Norouzi. "Neural Audio Synthesis of Musical Notes
  with WaveNet Autoencoders." 2017.

# Unwrapped dataset
Train: https://gigamove.rwth-aachen.de/en/download/a4f0c70ee1660dc6846a896100dfb00f <br>
Test: https://gigamove.rwth-aachen.de/en/download/2ad2487636a4435b7ebacc2c77b57097 <br>
Validation: https://gigamove.rwth-aachen.de/en/download/99feab2138062c9e0da69a27455cae01 <br>

The above dataset has been created using `convert_to_separate_files.py` since the original dataset is provided in sequential form
which prohibits us from selecting our required subset. If you wish to only train the model on the subset of acoustic data and pitches 24-84,
please download the dataset below

# Subset of acoustic samples with pitches 24-84
Train: https://gigamove.rwth-aachen.de/de/download/ab73c0f4f69034466ee92dd26dc19f2a <br>
Test: <br>
Validation: <br>

The above dataset has been created using `make_data_subset.py` on the unwrapped dataset while specifying the partition of data
that the modules should be working on
