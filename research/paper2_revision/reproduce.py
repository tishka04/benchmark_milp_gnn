"""Reproduce exported analysis without historical solver files or absolute paths."""
from pathlib import Path
import pandas as pd
from .analysis import write_analysis
from .core import sha

def main():
    path=Path.cwd()/"observations.csv"
    if not path.exists(): raise SystemExit("Run this command from the extracted reproduction archive")
    df=pd.read_csv(path,float_precision="round_trip")
    # CSV nulls must become JSON nulls before content hashing.
    rows=df.astype(object).where(pd.notna(df),None).to_dict("records")
    print(write_analysis(Path(__file__).parent/"runs/reproduction",rows,"reproduced",dict(file=path.name,sha256=sha(path))))

if __name__=="__main__": main()
