# Parallel_WaveGAN

**link:**
https://github.com/mahotani/Parallel_WaveGAN/issues/1

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

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;L_{adv}&space;(G,D)&space;=&space;E_{z&space;\sim&space;N(0,1)}&space;\[&space;(1&space;-&space;D(G(z)))^2&space;\]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;L_{adv}&space;(G,D)&space;=&space;E_{z&space;\sim&space;N(0,1)}&space;\[&space;(1&space;-&space;D(G(z)))^2&space;\]" title="L_{adv} (G,D) = E_{z \sim N(0,1)} \[ (1 - D(G(z)))^2 \]" /></a>

この時、
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;z" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;z" title="z" /></a>
はwhite noiseを示す。  

この式の敵対的損失(
    <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;L_{adv}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;L_{adv}" title="L_{adv}" /></a>
)を最小限に抑えることにより学習が実行される。  

次の最適化基準を私用して、生成されたサンプルを偽として正しく分類するように、識別器がトレーニングされる。  

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;L_D&space;(G,D)&space;=&space;E_{x-pdata}&space;\[&space;(1&space;-&space;D(x))^2&space;\]&space;&plus;&space;E_{z&space;\sim&space;N(0,1)}&space;\[&space;D(G(z))^2&space;\]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;L_D&space;(G,D)&space;=&space;E_{x-pdata}&space;\[&space;(1&space;-&space;D(x))^2&space;\]&space;&plus;&space;E_{z&space;\sim&space;N(0,1)}&space;\[&space;D(G(z))^2&space;\]" title="L_D (G,D) = E_{x-pdata} \[ (1 - D(x))^2 \] + E_{z \sim N(0,1)} \[ D(G(z))^2 \]" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;x" title="x" /></a>
と
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;pdata" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;pdata" title="pdata" /></a>
はそれぞれターゲット波形とその分布を示す。

### 2.2 Multi-resolution STFT auxiliary loss

安定性と効率を上げるために、多重解像度解像度STFT損失を提案する。  
以下の図は多重解像度STFT損失と敵対的なトレーニング法を組み合わせたフレームワークのイメージ図。  

<img width="625" alt="distillation" src="https://user-images.githubusercontent.com/39772824/87129514-6ad9d880-c2cc-11ea-9cd9-58734f8cecc6.png"> 

単一のSTFT損失は次のように定義する。

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;L_s(G)&space;=&space;E_{x&space;\sim&space;p(x),&space;x&space;\sim&space;pdata}&space;\[&space;L_{sc}(x,&space;\hat{x})&space;&plus;&space;L_{mag}(x,&space;\hat{x})\]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;L_s(G)&space;=&space;E_{x&space;\sim&space;p(x),&space;x&space;\sim&space;pdata}&space;\[&space;L_{sc}(x,&space;\hat{x})&space;&plus;&space;L_{mag}(x,&space;\hat{x})\]" title="L_s(G) = E_{x \sim p(x), x \sim pdata} \[ L_{sc}(x, \hat{x}) + L_{mag}(x, \hat{x})\]" /></a>

ここでの
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\hat{x}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\hat{x}" title="\hat{x}" /></a>
は生成されたサンプルを示し、
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;L_{sc}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;L_{sc}" title="L_{sc}" /></a>
と
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;L_{mag}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;L_{mag}" title="L_{mag}" /></a>
はそれぞれスペクトル収束とlog STFT振幅損失を示す。

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;L_{sc}(x,&space;\hat{x})&space;=&space;\frac{\|&space;|STFT(x)|&space;-&space;|STFT(\hat{x})|&space;\|_F}{\|&space;|STFT(x)|&space;\|_F}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;L_{sc}(x,&space;\hat{x})&space;=&space;\frac{\|&space;|STFT(x)|&space;-&space;|STFT(\hat{x})|&space;\|_F}{\|&space;|STFT(x)|&space;\|_F}" title="L_{sc}(x, \hat{x}) = \frac{\| |STFT(x)| - |STFT(\hat{x})| \|_F}{\| |STFT(x)| \|_F}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;L_{mag}&space;(x,&space;\hat{x})&space;=&space;\frac{1}{N}&space;\|&space;log&space;|STFT(x)|&space;-&space;log&space;|STFT(\hat{x})|&space;\|_1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;L_{mag}&space;(x,&space;\hat{x})&space;=&space;\frac{1}{N}&space;\|&space;log&space;|STFT(x)|&space;-&space;log&space;|STFT(\hat{x})|&space;\|_1" title="L_{mag} (x, \hat{x}) = \frac{1}{N} \| log |STFT(x)| - log |STFT(\hat{x})| \|_1" /></a>

ここで、
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\|&space;\cdot&space;\|_F" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\|&space;\cdot&space;\|_F" title="\| \cdot \|_F" /></a>
と
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\|&space;\cdot&space;\|_1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\|&space;\cdot&space;\|_1" title="\| \cdot \|_1" /></a>
はそれぞれフロベニウスとL1ノルムを示し、
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;|STFT(\cdot)|" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;|STFT(\cdot)|" title="|STFT(\cdot)|" /></a>
と
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;N" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;N" title="N" /></a>
はそれぞれSTFTの大きさと要素数を示す。
