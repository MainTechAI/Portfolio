import os

train_task_id = '3T256'
initial_epoch = 0
epoch_num = 50
lr = 1e-3
decay = 5e-4
clipvalue = 0  # default 0.5, 0 means no clip
patience = 5
load_weights = True  # False
lambda_inside_score_loss = 4.0
lambda_side_vertex_code_loss = 1.0
lambda_side_vertex_coord_loss = 1.0

total_img = 2077  # training images
validation_split_ratio = 0.1
max_train_img_size = int(train_task_id[-3:])
max_predict_img_size = int(train_task_id[-3:])  # 2400
assert max_train_img_size in [256, 384, 512, 640, 736], \
    'max_train_img_size must in [256, 384, 512, 640, 736]'
if max_train_img_size == 256:
    batch_size = 2
elif max_train_img_size == 384:
    batch_size = 1
elif max_train_img_size == 512:
    batch_size = 1
else:
    batch_size = 1
steps_per_epoch = total_img * (1 - validation_split_ratio) // batch_size
validation_steps = total_img * validation_split_ratio // batch_size

data_dir = 'icpr/'
origin_image_dir_name = 'image_10000/'
origin_txt_dir_name = 'txt_10000/'
train_image_dir_name = 'images_%s/' % train_task_id
train_label_dir_name = 'labels_%s/' % train_task_id
show_gt_image_dir_name = 'show_gt_images_%s/' % train_task_id
show_act_image_dir_name = 'show_act_images_%s/' % train_task_id
gen_origin_img = True
draw_gt_quad = True
draw_act_quad = True
val_fname = 'val_%s.txt' % train_task_id
train_fname = 'train_%s.txt' % train_task_id
shrink_ratio = 0.2
shrink_side_ratio = 0.6
epsilon = 1e-4

num_channels = 3
feature_layers_range = range(5, 1, -1)
feature_layers_num = len(feature_layers_range)
pixel_size = 2 ** feature_layers_range[-1]
locked_layers = False

if not os.path.exists('resources'):
    os.mkdir('resources')

saved_model_file_path = 'resources/east_model_%s.h5' % train_task_id
saved_model_weights_file_path = 'resources/east_model_weights_%s.h5' \
                                % train_task_id

pixel_threshold = 0.9
side_vertex_pixel_threshold = 0.9
trunc_threshold = 0.1
predict_cut_text_line = False
predict_write2txt = True

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

str_file = ''
save_path = ''
clipboard = ''
running = False
