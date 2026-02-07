# IN SHORT, THE DATASET HAS SOME TINY ISSUES. IN THE CSV THAT CONTAINED THE IMAGE PATHS AND WHICH SET THEY BELONG TO,
# THERE WERE A FEW SLIGHT ERRORS AND SO WE NEEDED TO CORRECT THEM. IN SHORT: SOME IMAGES LISTED AS BEING IN SUBfolderDER trainA WERE
# ACTUALLY IN trainB. AT FIRST I THOUGHT THE DATASET IS CORRUPT(IT HAS GARBAGE IMAGES), BUT ACTUALLY ONLY THE PATHS WERE THE
# ISSUE(NOT A LOT OF THEM), AS I SAID. SO, THE IMAGES THAT I SAID WERE ACTUALLY IN trainB(ZEBRAS) INSTEAD OF trainA(HORSES), WERE
# ACTUALLY ZEBRAS, AND VICEVERSA. SO THE ISSUE WAS NOT THE IMAGES THEMSELVES, NOR THEIR PLACEMENT IN THE folderDERS, BUT JUST THE PATHS IN THE CSV

# IMPORTANT - I LATER(WHEN WORKING WITH THE PREPROCESSING OF DATA) RENAMED THE "metadata_fixed.csv" FILE TO "metadata.csv"
# AND THE "metadata.csv" FILE TO "metadata_buggy.csv" SO YOU MIGHT WANT TO DO THE SAME OR ELSE THE PROGRAMS WON'T WORK


import os
import pandas as pd

root = os.path.dirname(os.path.abspath(__file__))
dataset = os.path.join(root, "Horse2zebra")
dataframe_original = pd.read_csv(os.path.join(dataset, "metadata.csv"))
rows = []
folders = ["trainA","trainB","testA","testB"]
split = {"trainA":"train","trainB":"train","testA":"test","testB":"test"}
domain = {"trainA":"A (Horse)","testA":"A (Horse)","trainB":"B (Zebra)","testB":"B (Zebra)"}
for folder in folders:
    for file in os.listdir(os.path.join(dataset, folder)):
        rows.append({"image_id": os.path.splitext(file)[0], "domain": domain[folder], "split": split[folder],
                     "image_path": os.path.join(folder, file).replace("\\","/")})
df = pd.DataFrame(rows).sort_values(["split","domain","image_id"]).reset_index(drop=True)
df.to_csv(os.path.join(dataset, "metadata_fixed.csv"), index=False)