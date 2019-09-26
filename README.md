#
# LeTS-Drive with SUMMIT simulator integration
## Data Processing
Set up a python virtualenv for an isolated setup of python packages which do not interfere with global python packages.
```
cd
virtualenv --system-site-packes py_bts_rl 
source ~/py_bts_rl/bin/activate
```
Optionally, append the source ~/py_bts_rl... line to ~/.bashrc. Also, to deactivate the current virtual environment, just type 
```
deactivate
```
Use `pip` to install the necessary dependencies:
```
pip install -r requirements.txt
```
Convert bags into h5 files:
```
python Data_processing/bag_to_hdf5.py --bagspath jun9/ --peds_goal_path Maps/
```
or use multiple threads:
```
python Data_processing/parallel_parse.py
```
combine bag_h5 files into training, validation, and test sets:
```
python Data_processing/combine.py --bagspath jun9/
```
or
```
find ./jun9/ -type f -name '*.h5' -exec mv -i {} ./h5/  \;
python Data_processing/combine.py
```
## IL Training
Start training and open tensorboard port
```
python train.py --batch_size 512 --lr 0.01 --train train.h5 --val val.h5
tensorboard --logdir runs --port=6001
```
To check the learning curve, run on your Local machine:
```
ssh -4 -N -f -L localhost:6001:localhost:6001 panpan@unicorn4.d2.comp.nus.edu.sg
```
On your browser, visit http://localhost:6010/

**Main Arguments**:
- `train`: The path to the train dataset
- `test`: The path to the test dataset
- `bagspath`: Path to bag files
- `peds_goal_path`: Path to pedestrian goals
- `lr`: Learning rate 
- `epochs`: Number of epochs to train. Default: 100
- `k`: Number of Value Iterations. Recommended: [10 for 8x8, 20 for 16x16, 36 for 28x28]
- `l_i`: Number of channels in input layer. Default: 2, i.e. obstacles image and goal image.
- `l_h`: Number of channels in first convolutional layer. Default: 150, described in paper.
- `l_q`: Number of channels in q layer (~actions) in VI-module. Default: 10, described in paper.
- `batch_size`: Batch size. 



