# macOS安装Tensorflow快速方法
![TF Logo](ml_cubist_expressionist_impressionist.jpg)

用最简单的办法在macOS上安装CPU-only版本的tensorflow试验环境。考虑到网络环境的不同，会提供google的whl文件提供一份github下载。

## 基础的macOS开发环境

```
Tensorflow提供了友好的PythonA API，因此首先需要在macOS上安装Python，这里选择使用homebrew来构建和管理Python 2.7版本的环境，如果已经有了，请跳过这部分。
```

**Homebrew** 是类似于ubuntu apt的一种程序包管理工具，提供了很多丰富的linux-like的lib和dev tool.详细的介绍和安装步骤在[这里](https://brew.sh/)。

安装方法很简单，

```
/usr/bin/ruby -e “$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)”
```

接下来，安装Python环境，这里使用2.7的版本，如果想使用3以上的版本，把下面改成python3即可。

```
brew install python
```

再来安装Python的包管理工具,

```
sudo pip install --upgrade pip #更新pip到最新版本
```

##安装Tensorflow

###如果你可以科学上网
```
sudo pip install --upgrade tensorflow      # for Python 2.7
```

###如果你`不可以`科学上网

现在[这里](/tensorflow-1.0.1-py2-none-any.whl)下载二进制包文件.然后,

```
sudo pip install --upgrade ./tensorflow-1.0.1-py2-none-any.whl
```

##跑一个短的测试用例

首先执行python交互环境,
```
$python
```

接着输入下面的代码,

```
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))
```

如果输出如下,表示成功
```
Hello, TensorFlow!
```

恭喜你可以开始tensorflow之旅了:)




