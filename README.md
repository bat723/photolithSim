# Optical-Lithography Failure-Analysis Pipeline

*Unsupervised defect detection with a physics simulator + auto-encoder*

## How It Works

This project is a pipeline designed to find microscopic defects in semiconductor manufacturing (optical lithography) without needing a massive dataset of manually labeled errors.

1. **Simulate the Physics:** I use a physics simulator to generate synthetic images of lithography patterns. This gives me a controlled mix of "clean" patterns and "defect" patterns to work with.

2. **Learn the "Normal" (The Machine Learning):** I trained a small auto-encoder model strictly on the clean images. The model learns how to compress and perfectly reconstruct normal patterns. 

3. **Spot the "Abnormal":** During inference, I feed the model new images. When it encounters a defect, it struggles to reconstruct the anomaly. By comparing the model's output to the original input, I generate a localized anomaly score that pinpoints the exact location of the failure.

## Quick-Start

**1. Clone the repo**
```bash
git clone [https://github.com/bat723/photolithSim.git](https://github.com/bat723/photolithSim.git)
cd photolithSim


2. Install Dependencies

```
pip install -r requirements.txt
```

3. Run the physics simulation

```
python scripts/6_contrast_process_window.py
```

4. Run the anomoly detector

```
python src/simulation/resist.py
```

