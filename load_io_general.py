from brainheart.io.base import BaseLoader
from brainheart.config.config_manager import ConfigManager
from brainheart.io.utils import load_data

local_config = ConfigManager()

set_file_path = local_config.get_path("data_path", "./data") / "eeglab_eeg_ecg" / "sub-001_ses-01_task-GXtESCTT_eeg.set"

test1 = load_data(set_file_path)
test2 = 0
# test = BaseLoader.load(file_path=set_file_path)
