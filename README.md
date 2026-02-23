# Optical-Lithography Failure-Analysis Pipeline 
_Unsupervised defect detection with a physics simulator + auto-encoder_

---

## How It Works

This project is a pipeline designed to find microscopic defects in semiconductor manufacturing (optical lithography) without needing a massive dataset of manually labeled errors. 

1. **Simulate the Physics:** Using a physics simulator to generate synthetic images of lithography patterns, giving us a controlled mix of "clean" patterns and "defect" patterns.
2. **Learn the "Normal":** I trained a small auto-encoder machine learning model strictly on the clean images. The model learns how to compress and perfectly reconstruct normal patterns. 
3. **Spot the "Abnormal":** During inference, I fed the model new images. When it encounters a defect, it struggles to reconstruct the anomaly. By comparing the model's output to the original input, we generate a localized anomaly score that pinpoints the exact location of the failure.

---

## Quick-Start

**1. Clone the repo**
```bash
git clone [https://github.com/bat723/photolithSim.git](https://github.com/bat723/photolithSim.git) && cd photolithSim
