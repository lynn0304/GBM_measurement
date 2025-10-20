# GBM_measurement

model weight [HERE](https://drive.google.com/file/d/1K121woii0Sf2g_vYv3TMc5x6NEMD5jV8/view?usp=sharing)

執行步驟：  
1. **segmentation**  
   調整 source (影像所在資料夾)
   ```sh
   bash run_hrnet.sh
   ```
2. **計算厚度**  
   如果在 1. 有更改檔名，需更改 line 294~296
   ```py
    python measure.py
   ```
3. **輸出統計圖**  
   需改 line 8~10 檔案路徑
   ```py
     python statistic.py
   ```
