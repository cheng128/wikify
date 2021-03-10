<h1 align="center">
<p> Wikify
</h1>

### Learning to Link with Wikipedia
本 repo 旨在復現此論文，其中有些地方有進行修正，不完全依照本論文作法，但亦獲得不錯效果，附上論文 pdf 供參考。

### Prototype
在此提供一個簡單的 prototype 以供檢視效果。
http://140.114.89.223:8502/ 

### Installation
可自行考慮是否要在 virtual environment 下執行安裝。 
在 command line 執行以下指令
```
git clone https://github.com/cheng128/wikify.git
pip install -r requirements.txt
```
### Use Directly
 若想直接使用標註功能，可直接在程式中添加以下程式碼，input 為純文字，output 形式見 Main Modules 部分說明：
 `from gen_entity_link import disambiguate_detect`
 
### Scripts
請先至 https://reurl.cc/qmn6d3 下載資料與訓練好的模型，並放入 data_model 資料夾（在此 project 底下創建新資料夾即可）
 - link_prob_dict.json
 - commonness_dict.json
 - relatedness_dict.json
 分別為論文中所提到的三種取得 features 所需資料。

在 ./disambiguate_model 底下有兩個檔案：
 - get_disambiguate_training_data.py: 取得 training 所需的 features，會輸出一個 json 檔。
  （features: commonness, relatedness, context_quality)
 - train_disambiguate_model.py: 訓練 model 並儲存。

在 ./detect_link_model 底下有兩個檔案：
 - get_detect_link_training_data.py: 取得 training 所需 features，會分成 train 與 test 兩個 json 檔。
   (features: link probability, first occurrence, last occurrence, spread)
 - train_detect_link_model.py: 訓練 model 並儲存。
 
可自行使用 wiki dump 資料進行訓練。
訓練好的 model 會儲存在 data_model 資料夾中，以供調用。

### Main Modules
整個系統由以下三個部分構成：
 - `pre_tools`:  負責 load data 與 model，並提供訓練兩種 model 時獲取資料的 function。 
 - `disambi_detect`: 進行文本內 entity detect 之後進行 sense disambiguate，選出適當連結網頁。
 - `gen_entity_link`: 取得 disambi_detect 後的資料，去除重複出現的 entity 並轉成以下形式。
     {entities:[{entity: ,url: , start: , end: },{...}]}）
     其中 url 為英文維基頁面。
     start 與 end 為直接使用 split() 計算出的位置。
     如 "Tracking Covid-19’s global spread.", "Covid-19" start 與 end 皆為 1。
