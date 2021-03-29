import pandas as pd
import numpy as np

QM7_CSV_PATH = 'gdb7.sdf.csv'

df = pd.read_csv(QM7_CSV_PATH)
csv: np.ndarray = df.values
properties = csv[:, 0: 1].astype(np.float)
std = np.std(properties)
print(std)
