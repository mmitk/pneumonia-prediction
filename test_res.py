from common.src import resample as res
from imblearn.over_sampling import RandomOverSampler

#path = r'C:\Users\mmitk\dev\2020\pneumonia\common\data\TEST'
path = r'C:\Users\mmitk\dev\2020\pneumonia\common\data\chest_xray\train'
ros = RandomOverSampler(random_state=0)

res.resample_directory(ros, path, 'ROS', val = False)
