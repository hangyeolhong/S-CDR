# S-CDR
2021 2nd semester project (skt project)

## Contribution
- Propose *S-CDR*, a cross-domain recommendation framework that utilizes users' social relationships.
- Found that using social information in CDR can improve the performance.

## Requirements
- Python 3.6
- Pytorch > 1.0
- tensorflow
- pandas
- numpy
- Tqdm

You can run this model through ``` python entry.py ```

## Dataset
- Douban dataset, which can be downloaded at https://www.dropbox.com/s/u2ejjezjk08lz1o/Douban.tar.gz?dl=0

|Dataset|#user|#item|#event|sparsity|
|:----------:|:-----:|:------:|:--------:|:------:|
|DoubanMovie|94,890|81,906|11,742,260|0.151%|
|DoubanMusic|39,742|164,223|1,792,501|0.027%|
|DoubanBook|46,548|212,995|1,908,081|0.019%|


|	|#node|#edge|
|:---------:|:------:|:-------:|
|SocialNet|695,800|1,758,302|

|Source|Target|Source (#item)|Target (#item)|Overlap (#user)|Source (#user)|Target (#user)|Source (#rating)|Target (#rating)|
|:-----:|:-----:|:------:|:------:|:------:|:------:|:------:|:----------:|----------:|
|Movie|Music|81,906|164,223|39,710|94,890|39,742|11,742,260|1,792,501|
|Book|Movie|212,995|81,906|46,506|46,548|94,890|1,908,081|11,742,260|
|Book|Music|212,995|164,223|26,360|46,548|39,742|1,908,081|1,792,501|

Especially, cross domain recommendations _from book to music_ were conducted.

## Evaluation result
- Baseline
  - EMCDR
    - Adopt MF to learn embeddings first
    - Utilize a network to bridge the user embeddings from the auxiliary domain to the target domain
  - PTUPCDR
    - Meta network fed with users' characteristic embeddings is learned to generate personalized bridge functions to achieve personalized transfer of user preferences



|Metric|EMCDR|PTUPCDR|S-CDR|Improvement|
|:----------:|:-----:|:------:|:--------:|:-----:|
|MAE|1.6909|1.5275|**1.4395**|5.76%|
|RMSE|2.2821|2.1695|**2.0923**|3.56%|
