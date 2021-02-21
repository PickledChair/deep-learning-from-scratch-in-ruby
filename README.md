# 「ゼロから作る Deep Learning」を Ruby で実装する

「[ゼロから作る Deep Learning](https://github.com/oreilly-japan/deep-learning-from-scratch)」の Python コードを、Ruby の勉強がてらに、Ruby で再実装したリポジトリです。



## ディレクトリ構成

- [x] chapter2
- [x] chapter3
- [ ] chapter4
- [ ] chapter5
- [ ] chapter6
- [ ] chapter7
- [ ] chapter8
- [ ] common: 複数の章で共通に使用するソースコード
- [x] dataset: データセット用のソースコード



## テスト環境

MacBook Air (early 2020) (Intel Mac), macOS Big Sur



## Ruby のバージョン・依存ライブラリ

Ruby: 3.0.0（rbenv でインストール）で動作確認しています。

### 依存ライブラリ

- [Numo::NArray](https://github.com/ruby-numo/numo-narray)：数値計算ライブラリ（NumPy に相当）
- [Charty](https://github.com/red-data-tools/charty)：グラフ描画（バックエンドに matplotlib を使用）
- [Numo::Gnuplot](https://github.com/ruby-numo/numo-gnuplot)：グラフ描画。charty では画像を描画できなかったのでこちらも併用（バックエンドに gnuplot を使用）
- [Npy](https://github.com/ankane/npy)：Numo::NArray オブジェクトと NumPy の `.npy` ファイル及び `.npz` ファイルを相互変換できるライブラリ。
- [Magro](https://github.com/yoshoku/magro)：画像ファイル => Numo::NArray オブジェクト, Numo::NArray オブジェクト => 画像ファイルを実現するライブラリ
  - ニューラルネットワークの実装では使っていないが、NArray と画像との相互変換が可能かどうか確認したかったので。



`magro` をリポジトリに記載の通り `gem install magro` でインストールしようとしたら、`png.h` が見つからないと言われて失敗してしまいました。しかし以下のコマンドでインストールできました：

```
$ export LIBRARY_PATH="/usr/local/lib"
$ export C_INCLUDE_PATH="/usr/local/include"
$ gem install magro
```



## ライセンス

[MIT License](./LICENSE.txt) で配布します。

