{\rtf1\ansi\ansicpg1252\cocoartf2865
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fmodern\fcharset0 Courier;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;}
\margl1440\margr1440\vieww14920\viewh17520\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs26 \cf0 \expnd0\expndtw0\kerning0
cd ~/projects/photolith\
\
# 1) wipe test folder completely\
rm -f data/processed_images/test/*.png\
\
# 2) copy 50 clean with prefix\
for i in \{0..49\}; do\
  cp "data/raw_synthetic/clean/$(printf "%04d" $i).png" \\\
     "data/processed_images/test/clean_$(printf "%04d" $i).png"\
done\
\
# 3) copy 200 defect with prefix\
for img in data/raw_synthetic/defect/*.png; do\
  base=$(basename "$img")\
  cp "$img" "data/processed_images/test/defect_$base"\
done\
\
# 4) verify counts (should be 250 total)\
echo "total:  $(ls data/processed_images/test/*.png | wc -l)"\
echo "clean:  $(ls data/processed_images/test/clean_*.png | wc -l)"\
echo "defect: $(ls data/processed_images/test/defect_*.png | wc -l)"\
\
# 5) verify there are NO unprefixed files left\
echo "unprefixed: $(ls data/processed_images/test/[0-9][0-9][0-9][0-9].png 2>/dev/null | wc -l)"\
\
\
\
\
# Optical-Lithography Failure-Analysis cdPipeline  \
_Unsupervised defect detection with a physics simulator + auto-encoder_\
\
---\
\
## 1\uc0\u8194 Quick-Start\
\
```bash\
# \uc0\u10102  clone or cd into repo\
git clone &lt;YOUR-REPO&gt; &amp;&amp; cd photolith\
\
# \uc0\u10103  create / activate virtual-env\
python3.12 -m venv .venv\
source .venv/bin/activate\
pip install -r requirements.txt\
\
# \uc0\u10104  generate data  (200 clean + 200 defect)\
python scripts/1_generate_data.py\
\
# \uc0\u10105  train tiny auto-encoder on CLEAN images only\
python scripts/2b_train_simple_model.py   # 16-dim latent\
\
# \uc0\u10106  run inference with localized anomaly score\
python scripts/3_run_inference_localized.py\
\
# \uc0\u10107  inspect figures + report\
open results/figures/localized_score_hist.png\
cat  results/predictions_localized.txt    # CLEAN vs DEFECT list\
}
