from common.src import resample as res
from common.src import util
from imblearn.over_sampling import RandomOverSampler, ADASYN

#path = r'C:\Users\mmitk\dev\2020\pneumonia\common\data\TEST'
path = r'C:\Users\mmitk\dev\2020\pneumonia\common\data\chest_xray\train'
#ros = RandomOverSampler(random_state=0)

from collections import Counter


train_folder = './common/data/TEST/'
train_datagen = util.create_train_datagen()
training_set = util.create_generator_set(train_datagen, train_folder)

counter = Counter(training_set.classes)
max_val = float(max(counter.values()))
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}

ads = ADASYN()

resampler = res.ImageResampler(target_directory=r'TEST_ADASYN', resampler=ads)
resampler.resample_directory(training_set)
#res.resample_directory(ads, path, 'ADASYN', val = False)
