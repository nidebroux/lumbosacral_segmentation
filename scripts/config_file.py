import os

config = dict()
config["patch_size"] = (64,64, 48)  # Size of the patches to extract
config["patch_overlap"] = 3  # Size of the overlap between the extracted patches along the third dimension
config["batch_size"] = 5  # Size of the batches that the generator will provide
config["n_epochs_1"] = 75 # Total number of epochs to train the model at first training
config["n_epochs_2"] = 75 # Total number of epochs to train the model at second training
config["augment"] = True  # If True, training data will be distorted on the fly so as to avoid over-fitting
config["augment_flip"] = True  # if True and augment is True, then the data will be randomly flipped along the x, y and z axis
config["learning_rate_drop"] = 0.5 # How much at which to the learning rate will decay
config["learning_rate_patience"] = 10  # Number of epochs after which the learning rate will drop

config["gpu_id"] = '0'
config["main_dir"] = '/home/nidebroux/'
config["data_dir"] = '/home/nidebroux/data_mask/preprocess/Data_organized/'
config["data_dict"] = '/home/nidebroux/data_mask/subject_dict.pkl'  # pickle file containing a dictionary with at least the following keys: subject and contrast_foldname


# Model name containing the main parameters, which is useful for the hyperparm optimization
config["model_name"] = "custom_lumbar"

config["path2save_retrained"] = '/home/nidebroux/sct_custom/data/deepseg_sc_models//reTrained_models/' 
config["path2save_finetuned"] = '/home/nidebroux/sct_custom/data/deepseg_sc_models//fineTuned_models/'
config["path2save_transferlearning"] = '/home/nidebroux/sct_custom/data/deepseg_sc_models//transferLearned_models/'
config["path2save"] = '/home/nidebroux/sct_custom/data/deepseg_sc_models//custom_model/'
# Relative path of the folder where the trained models are saved
