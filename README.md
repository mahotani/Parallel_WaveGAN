# Parallel_WaveGAN

## 1 INTRODUCTION

合成音声生成の分野はtext-to-speechなどの登場により生成される音の質が向上しました。  
しかし、WaveNetなどの非自己回帰モデルは性能は良いが、生成にとても時間がかかります。  
生成時間を早くするために、蒸留を使ったものが提案されていたが、この方法もトレーニング過程の計算が難しかったりして大変。  
なので、今回のParallel WaveGANは蒸留を使わずに生成にかかる時間を縮めたよっていうのが推したいとこ。  
具合的には、多重解像度短時間フーリエ変換(Multi-STFT)と敵対損失関数の組み合わせを最適化することにより、少ないパラメータ数で蒸留なしでより早い音声生成を可能にした。  

### 蒸留
生成モデルは一般的に深くてパラメータ数の多いモデルの方が精度が上がりやすいことが知られている。  
また、単一のモデルで予測するよりも、複数モデルの予測結果を組み合わせるアンサンブル学習の方が精度が上がることも知られている。  
精度のみが問題になる場合、以下のようなアプローチが取られる。  

<img width="625" alt="distillation" src="https://user-images.githubusercontent.com/39772824/87123120-1e3cd000-c2c1-11ea-879a-5404b009b653.png"> 

予測精度の良い深いモデルやアンサンブルさせたモデルを教師モデルとして準備しておき、その知識を軽量で再現できる生徒モデルの学習に利用する。  
これにより、軽量でありながら高精度の教師モデルを引き出すことができる。  

## 2 METHOD

### 2.1 Parallel waveform generation based on GAN

ジェネレータはWaveNetベース。  
以下はジェネレータとディスクリミネータの関係式。  

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;L_{adv}&space;(G,D)&space;=&space;E_{z&space;\sim&space;N(0,1)}&space;\[left&space;(1&space;-&space;D(G(z)))^2&space;\]right" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;L_{adv}&space;(G,D)&space;=&space;E_{z&space;\sim&space;N(0,1)}&space;\[left&space;(1&space;-&space;D(G(z)))^2&space;\]right" title="L_{adv} (G,D) = E_{z \sim N(0,1)} \[left (1 - D(G(z)))^2 \]right" /></a>

この時、
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;z" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;z" title="z" /></a>
はwhite noiseを示します。
