python inference.py --debug
python datasets/build_test_dataset.py

# Step 2 (Sequential): Inference on the whole dataset
python inference.py --sequential

# Or Step 2 (Parallel): Inference on the whole dataset
python inference.py