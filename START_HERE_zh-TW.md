# 使用說明

這是可供檢查的 GitHub 專案包，尚未上傳 GitHub。

1. 解壓縮後，先閱讀 README.md 的英文介紹與個人分工。
2. 程式實作依檔案證據標為 pix2pixHD，並保留原始來源與授權。
3. 資料集、原始照片、標註、生成圖片與權重沒有放入；正式公開前，確認教授及隊友同意程式公開範圍。
4. 沒有填入缺乏完整紀錄支持的 FID 或 YOLOv9 成果數值。
5. FID 與縮圖程式改成指令列參數。操作方式見 README.md 與 docs/SETUP.md。
6. 這份包只整理此次提供的 GauGAN.zip；Stable Diffusion 與 YOLOv9 尚未整合。

建議 repository 名稱：wind-blade-synthetic-augmentation。

若要手動上傳：在 GitHub 建立 repository，先選 Private；使用 Upload files 上傳解壓縮後的內容，讓 README.md 位於 repository 最外層。記得包含 .gitignore，不要直接把整個 ZIP 當成唯一檔案上傳。確認內容與公開範圍後再設為 Public。

已檢查程式語法、FID 輸入驗證與縮圖時 mask ID 保留；未執行完整 GPU 訓練或實際 FID 計算。原模型為舊版研究程式，可能需要環境相容性調整。
