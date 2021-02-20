require 'uri'
require 'net/http'
require 'zlib'
require 'numo/narray'
require 'npy'

module MNIST
  class << self
    URL_BASE = 'http://yann.lecun.com/exdb/mnist/'
    KEY_FILE = {
      'train_img' => 'train-images-idx3-ubyte.gz',
      'train_label' => 'train-labels-idx1-ubyte.gz',
      'test_img' => 't10k-images-idx3-ubyte.gz',
      'test_label' => 't10k-labels-idx1-ubyte.gz'
    }

    SAVE_FILE = __dir__ + '/mnist.npz'

    TRAIN_NUM = 60000
    TEST_NUM = 10000
    IMG_DIM = [1, 28, 28]
    IMG_SIZE = 784


    def download(file_name)
      file_path = __dir__ + "/" + file_name

      if Pathname.new(file_path).exist? then
        return
      end

      puts "downloading" + file_name + " ... "
      uri = URI.parse(URL_BASE + file_name)
      bytes = Net::HTTP.get(uri)

      File.open(file_path, 'wb') do |f|
        f.write(bytes)
      end
      puts "Done"
    end
    private :download

    def download_mnist
      KEY_FILE.values.each { |v| download(v) }
    end
    private :download_mnist

    def load_label(file_name)
      file_path = __dir__ + "/" + file_name

      puts "Converting " + file_name + " to Numo::UInt8 ..."
      labels = nil
      Zlib::GzipReader.open(file_path) { |gz|
        gz.read(8)
        labels = Numo::UInt8.from_binary(gz.read())
        gz.finish
      }
      puts "Done"

      return labels
    end
    private :load_label

    def load_img(file_name)
      file_path = __dir__ + "/" + file_name

      puts "Converting " + file_name + " to Numo::UInt8 ..."
      data = nil
      Zlib::GzipReader.open(file_path) { |gz|
        gz.read(16)
        data = Numo::UInt8.from_binary(gz.read())
        gz.finish
      }
      puts "Done"

      return data
    end
    private :load_img

    def convert_numo_narray
      dataset = {}
      dataset['train_img'] = load_img(KEY_FILE['train_img']).reshape(TRAIN_NUM, IMG_SIZE)
      dataset['train_label'] = load_label(KEY_FILE['train_label'])
      dataset['test_img'] = load_img(KEY_FILE['test_img']).reshape(TEST_NUM, IMG_SIZE)
      dataset['test_label'] = load_label(KEY_FILE['test_label'])

      return dataset
    end
    private :convert_numo_narray

    def init_mnist
      download_mnist
      dataset = convert_numo_narray
      puts "Creating npz file ..."
      Npy.save_npz(SAVE_FILE,
                   train_img: dataset["train_img"],
                   train_label: dataset["train_label"],
                   test_img: dataset["test_img"],
                   test_label: dataset["test_label"])
      puts "Done"
    end

    def change_one_hot_label(x)
      t = Numo::UInt8.zeros([x.size, 10])
      t.inplace.map_with_index { |_,i,j| if j == x[i] then 1 else 0 end }
      return t
    end
    private :change_one_hot_label

    # MNISTデータセットの読み込み
    #
    # @param normalize [Boolean] 画像のピクセル値を0.0~1.0に正規化する
    # @param one_hot_label [Boolean] one_hot_labelがTrueの場合、ラベルはone-hot配列として返す。one-hot配列とは、たとえば[0,0,1,0,0,0,0,0,0,0]のような配列
    # @param flatten [Boolean] 画像を一次元配列に平にするかどうか
    #
    # @return [Array] [訓練画像, 訓練ラベル], [テスト画像, テストラベル]

    def load_mnist(normalize: true, flatten: true, one_hot_label: false)
      if !Pathname(SAVE_FILE).exist? then
        init_mnist
      end

      dataset_npz = Npy.load_npz(SAVE_FILE)

      dataset = {}
      dataset_npz.keys.each { |key| dataset[key] = dataset_npz[key] }

      if normalize then
        ['train_img', 'test_img'].each do |key|
          dataset[key] = Numo::DFloat.cast(dataset[key])
          dataset[key] /= 255.0
        end
      end

      if one_hot_label then
        dataset['train_label'] = change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = change_one_hot_label(dataset['test_label'])
      end

      if !flatten then
        dataset['train_img'] = dataset['train_img'].reshape(TRAIN_NUM, 1, 28, 28)
        dataset['test_img'] = dataset['test_img'].reshape(TEST_NUM, 1, 28, 28)
      end

      return [dataset['train_img'], dataset['train_label']], [dataset['test_img'], dataset['test_label']]
    end
  end
end


if $0 == __FILE__
  MNIST.init_mnist
end
