import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pl_mmb_training.visualnews_data import prepare
if __name__=='__main__':
    prepare()

