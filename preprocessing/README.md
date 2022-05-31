# Download the unwrapped dataset
Train: https://gigamove.rwth-aachen.de/en/download/a4f0c70ee1660dc6846a896100dfb00f <br>
Test: https://gigamove.rwth-aachen.de/en/download/2ad2487636a4435b7ebacc2c77b57097 <br>
Validation: https://gigamove.rwth-aachen.de/en/download/99feab2138062c9e0da69a27455cae01 <br>

# Example of making a Pytorch dataloader
Example code can be read in `test_data_reader.py`. <br>
This shows how to use the Pytorch Dataset object NSynthDatasetTorch from the library file `data_reader.py`. This object takes three inputs: <br>
1. `select` is a dictionary in which one can specify, which subset of data one wants to work with. 
The `select` variable in the example code is defined as the subset of acoustic instruments with the pitch ranging from 24 to 84.
By specifiying an instrument name for example, one can further slit up the dataset.
2. The variable `path` specifies the folder where the dataset that one wants to load is located.
3. `label` can take the values `pitch` or `instrument_family` which will either load the data with pitches as labels or the intrument_family as labels.
