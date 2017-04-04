HkarDemoの実行順番は：
１．1)./STSJP
    2)python3 getDailyNews.py (CNNの判定はKeywordより遅いので、一つのnewsは大体7分ぐらい掛かるので、今の設定ではなく、適当に数を絞ってテストする方が宜しいと思います。)
２．1)./STSJPMS/exps/exp2016
    2)bash expeval (訓練済みのモデルを用いてnewsの類似度を判定します。類似度の数値はkeyword hit numberの代わりに記録されます。今の設定では類似度1.3以上のnewsを抽出しています。)
３．1)./STSJP
    2)python3 rankByPrices.py (price変化で類似記事をソートする)
４．1)./STSJP
    2)python graph.py (web demoを起動します。)
----------------------------------------------------
新たなモデルを訓練したければ：
５．1)./STSJPMS/exps/exp2016
    2)bash exptrain (./STSJPMS/data/sts/2016に格納されている訓練データを用いてモデルを訓練する、現在の訓練結果は約0.45です。)



類似newsの例：simresultに参考