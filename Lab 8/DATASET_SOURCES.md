# Dataset sourcing

- Synthetic: `DATASETS/team16_Synthetic/data.pkl` (provided Team 16 pickle); archive `Synthetic.zip` retained.
- MNIST: `DATASETS/team16_mnist/data.pkl` (provided MNIST-labelled pickle; validated duplicate of Synthetic file, so verify with course team).
- Image dataset: `DATASETS/team16_imageFeatures/Dataset2/{train,val,test}_{data,label}.csv`.
- TI digits: `DATASETS/tdigits.zip`; notebook extracts to `DATASETS/tdigits_extracted/` and reads `Data/ExtractedData/<digit>.npz`.

Each notebook obtains `DATA` from the shared helper relative to the project folder.
