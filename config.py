""" Dataloader """
batch_size = 8
num_workers = 1
dataset_image_size = (2048, 2048)  # (height, width)
input_image_size = (1024, 1024)    # (height, width)  -> goes into the model

train_ratio = 0.8

# filepaths
dir_path_defocused_blur = "./data/defocused_blurred"
dir_path_sharp = "./data/sharp"

""" Training """
num_epochs = 20
learning_rate = 1e-4
scheduler_step_size = 15

""" Saving and Logging """
save_frequency = 2  # save after every x epochs

log_path = "./logs"
log_frequency = 25  # save after every x batches

""" Weights """
load_weights_folder = "weights_defocus_decoder"
models_to_load = ["encoder", "defocus_decoder"]