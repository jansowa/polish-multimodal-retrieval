import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pl_mmb_training.wiki_pilot import main
if __name__=='__main__':main()
