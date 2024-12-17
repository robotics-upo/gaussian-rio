import os
import configparser

def _load_config() -> configparser.ConfigParser:
	cfg_file = os.path.abspath(__file__+'/../../config.ini')
	cp = configparser.ConfigParser()
	cp.read(cfg_file, encoding='utf-8')
	return cp

CONFIG = _load_config()
