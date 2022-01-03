# skt2021
skt project in the 2nd semester of 2021.

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

|Source|Target|Source|Target|Overlap|Source|Target|Source|Target|
|:-----:|:-----:|:------:|:------:|:------:|:------:|:------:|:----------:|----------:|
|Movie|Music|81,906|164,223|39,710|94,890|39,742|11,742,260|1,792,501|
|Book|Movie|212,995|81,906|46,506|46,548|94,890|1,908,081|11,742,260|
|Book|Music|212,995|164,223|26,360|46,548|39,742|1,908,081|1,792,501|

Especially, Cross domain recommendations from book to music were conducted.

## Evaluation result

|Metric|EMCDR|PTUPCDR|S-CDR|Improvement|
|:----------:|:-----:|:------:|:--------:|:-----:|
|MAE|1.6909|1.5275|**1.4395**|5.76%|
|RMSE|2.2821|2.1695|**2.0923**|3.56%|

## Reference
- PTUPCDR(WSDM 2022): https://arxiv.org/pdf/2110.11154.pdf
- GraphRec(WWW 2019): https://arxiv.org/pdf/1902.07243.pdf
