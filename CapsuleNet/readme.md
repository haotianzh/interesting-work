####*概要*
去年年底，Geoffrey E. Hinton开源了一个新的图像识别模型CapsulesNet,中文翻译过来呢就是胶囊网络。虽然刚刚开源没多久，但是fork和star就已经爆炸了。的确，新东西就是有新东西的亮点。算法部分呢，我就简单说一下，重在理解。
 * 首先要明确一个东西，那么就是“什么是胶囊？” 这个问题我最后在说。那么我们要知道所谓胶囊网络，就是将CNN中每个神经元的标量输出，替换为一个向量输出。于是呢，原本的这个“神经元”现在也有了一个新名字叫做“胶囊”。或者，我们可以有一种新的理解方式——既然新的输出是向量，也就是多个标量的结合。那么胶囊理所当然的可以当成若干个CNN 神经元的结合。
* 类比一下CNN和CapsNet，CNN的过程是（convolution）加权求和，激活函数（relu），有用信息的获（maxpooling），CapsNet是这样的：加权求和（向量的加权和），激活函数（squashing），有用信息的获取（dynamic routing）。

CapsNet的流程，我从原论文上直接截下来，放在这：
![向量加权和](http://upload-images.jianshu.io/upload_images/6692878-4cfe8746bfac33b3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![激活函数squashing](http://upload-images.jianshu.io/upload_images/6692878-0d5b31be72015f5f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
从图中可以看出来，v 就是一个胶囊最后的输出。那么不仅有个疑问，最后一步的共识路由去哪了？怎么只有2步。回看第一个公式，里面的参数c，也就是权值系数是什么？
![权值系数c](http://upload-images.jianshu.io/upload_images/6692878-08cd597bf7811eb5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
很直白，c就是一个softmax后的数组，这一步竟是什么意思呢？现在有必要来说一下CapsNet的设计思路了。
假设：
- 我们现在面临一个识别人脸的任务。
- 把CapsNet类比CNN，只不过原来的神经元，现在是胶囊
- 每个胶囊都代表着一个检测任务
- 最后的输出层只有2个胶囊，这2个胶囊分别检测的是“脸”和“不是脸”
- 中间层的胶囊，都负责检测一些更低级特征的任务，比如鼻子，眼睛，嘴巴...当然还有一些其他乱七八糟的东西，比如耳环啊，眼镜啊。

那么我们现在思考一下，中间层的胶囊要把激活值输出给下一层的胶囊，也就是要输出给“脸”和“不是脸”这两个胶囊，那么理所当然的，希望检测“鼻子”，"眼睛"这一类的胶囊，能和检测“脸”的胶囊关系更大，而与“不是脸”的关系更小。那么就像我们站在一个路口，我们选择任意一条路的概率和就是1 ，中间层胶囊也是如此，但是胶囊更倾向于选择和自己最接近也是最相似的输出胶囊或者说和他们自身关系最大的胶囊。（那么“鼻子”就应该选择“脸”，“耳环”就应该选择“不是脸”）回到公式中，softmax后的 c 就是当前胶囊选择下一层胶囊的概率分布其和为1。
####*Dynamic Routing*
那么如何确定 c 呢，这里就是用了这个叫做“dynamic routing”的方法。这个方法蛮多的介绍，主要的意思呢就是通过几次迭代，根据高级胶囊的输出逐步调整低级胶囊输出给高级胶囊的分布，最后会达到一种理想的分布。
![dynamic routing process](http://upload-images.jianshu.io/upload_images/6692878-31defff23606dadb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这里有一个注意的地方就是，我们看到参数 c 在不断的更新，但是是由dynamic routing 来调整的，并不是通过backward调整的。换句话说，在backward运算中并不会计算 c 的误差（不更新 c 值），c 的更新和确定是在forward运算中的。
激活函数squashing不难看出是做了一个方向上不变，长度上进行收缩的操作。

####*Loss Function*
![loss function](http://upload-images.jianshu.io/upload_images/6692878-966b2ef65f1b55f6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
在来看看损失函数，使用的方式和svm的损失函数比较类似（最大化正负样本到超平面的距离）。这里给定了2个锚点m+=0.9和m-=0.1，损失最终希望正例样本预测在0.9，超过就没必要继续提高了（max(0,m-y)），负例在0.1，低于0.1就没必要继续下降了，于是取了max（0，y-m）。然后 λ 是为了减小那些图片中没有出现过的数字类别的损失，防止一开始损失过大，导致全部的输出值都在收缩。（毕竟图片中不存在的数字更多，小的损失x多的数量=大的损失）。损失函数就谈这么多。

####*Caps Net 和 Neural Net 对比*
现在总结一下，Capsnet和普通的神经网络的区别，借用了[naturomics（github传送）](https://github.com/naturomics/CapsNet-Tensorflow)的图片![对比](http://upload-images.jianshu.io/upload_images/6692878-7818ca2ff851e6fe.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

####*Capsnet Architecture*
接下来看看如何将一个typically CNN改造成一个Capsnet。![Capsnet Arch](http://upload-images.jianshu.io/upload_images/6692878-32d66cdb6e9b95df.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- Conv1 是对原图像做了一次卷积+激活，shape为（28-9+1=20）20 * 20 * 256
- PrimaryCaps层，对Conv1产生的输出再做一次卷积。那么如何由标量变成胶囊的呢，我这里的理解是做了32次不同的卷积，每次卷积的通道数为8，并且这8次的卷积结果封装在一起变成一个胶囊层，一共产生了32个胶囊层。每个通道里是6 x 6的胶囊网格（36个胶囊）。这个理解与[知乎](https://www.zhihu.com/question/67287444/answer/251460831)上有些不同。但本质上都是产生了完全不同的32x8个通道，至于怎么组合成胶囊，请自行揣测。但是只要注意一点就好，就是每个6 x 6网格都共享同一个卷积权重就好。这句话的意思就是对于任意一个胶囊层里的全部胶囊来说，它们向量中任意位置（一共8个标量，假设是第2个位置上的标量）的标量[u1,u2...u36]均是由完全相同的filter得到的。shape为6 * 6 * 32 * 8
- DigitCaps层，这一层一共有10个胶囊，其中每个胶囊代表0~9中一个数字。这里的胶囊就象征着最高阶的特征，也即数字。shape为10 * 16。这一层的输入是上一层的flatten（6 * 6 * 32=1152）个胶囊。
- 论文中最后还附加了一个decode网络，作用是将学习到的知识还原回图像，同时可以对DigitCaps中的16d胶囊中的任意1维度进行微调，用来观察这16维每一位度所代表的物理含义。
![维度含义](http://upload-images.jianshu.io/upload_images/6692878-af9207d08df4e0d8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
这样从图中就不难看出这16个维度的确反映了图像的：高矮，胖瘦，明暗，粗细等特征。

说了这么多，不禁有个疑惑，胶囊究竟学习到了什么东西？
####*胶囊究竟学习到了什么东西？*
众所周知，大名鼎鼎的cnn确实学习到了东西，图像的一些底层特征，比如 边、角、点这种零碎的图像组成部分——或者可以说一些模式。
但是有个弊端就是，cnn会把一条昏暗的横线和一条明亮的横线认为是2个特征，而实际上可能只是由于光线等因素导致的同一事物的2个样子。
于是胶囊的作用就体现出来了，我们之前在 *Capsnet Architecture* 这一块里面说了，胶囊是由8个卷积结构封装在一起形成的。一个最直观的理解就是，它将一个事物的8种不同形态封装在一起。这8个形态的激活值合在一起作为输出。举个例子：一个负责检测横线的胶囊，将一条明亮的线，一条昏暗的线，一条粗一点的线，一条细一点的线的检测结果封装在一起。那么但凡有一个激活值很高，那么整体的模长就会很长，那么这个胶囊就处于激活态~证明检测到了一条横线。于是胶囊就学习到不同状态下的物品其实是同一个。对应到从decode网络中发现的结果，最后DigitCaps层中16维的胶囊，正代表着不同状态下的同一个数字。比如有一个维度是检测数字粗细的，那么微调这个维度的激活值，那么还原回去就对应着数字的粗细的改变，如下图![微调控制粗细的维度的激活值](http://upload-images.jianshu.io/upload_images/6692878-eb36cf16cb907491.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
说白了，原来的神经元存储着一种状态下一个特征的active，而现在胶囊存储着多种状态下一个特征的active。
- 因此随着胶囊层级的逐渐升高，高级胶囊理应有更多的不同状态，所以高级胶囊的向量维度应该更高

####*Dynamic Routing是干嘛的？*
我们发现，在与普通的neural network做对照的时候，缺少了maxpooling这一步，但是却多了 routing这一个过程，为什么？
因为routing在这里代替了maxpooling，他们都是一种对有用信息的提取方式
- maxpooling依靠区域最大化，以此来提取到了这个区域里最有用的信息，但是却忽略了其他的信息。我们称之为信息丢失。
- routing 目的也是提取有用信息，我们考虑一个问题：我们在pooling 的过程中所丢弃的那些信息，真的没有用么？假设我们识别的一张图片里有一只猫，和一条狗，猫占图片的大部分。于是maxpooling后我们丢弃掉了较少的狗的信息，而保留了较多的猫的信息，于是这张图片被识别为猫。那么狗呢，狗怎么办？
routing的好处就是并不丢弃任何的信息，属于猫的信息那么就将其更多的输出到检测“猫”的高级胶囊中，属于狗的信息就尽可能的都输出给“狗”的高级胶囊。这样的我们最后的结果，猫和狗的得分都会很高。因此Hinton说，Caps更适合去检测那些重叠在一起的图片——overlapping
- Hinton还说routing像attention机制。这个看大家的口味吧，毕竟有所不同，首先就是谁选择谁的问题——attention是高层选底层，caps是底层选高层。
![原文](http://upload-images.jianshu.io/upload_images/6692878-e24d85232d5686a3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

####*Coding的一些细节*
- 对于矩阵维度的一些扩展，和删除
- 矩阵concatnate的一些注意
- Tensorflow的话注意一些内置函数的使用，比如tf.scan或者是keras中的map_fn
- 时刻保持维度的一致性
- 总的来说前向图创建完后，剩下的都交给framework了。

####*最后*
以上仅是个人对胶囊网络的一种理解，不喜勿喷。毕竟新东西刚出，有问题和细节希望一同探讨。
参考如下：
- [Understanding Dynamic Routing between Capsules (Capsule Networks)](https://jhui.github.io/2017/11/03/Dynamic-Routing-Between-Capsules/)
- [Hinton 原版论文](https://arxiv.org/pdf/1710.09829.pdf)
- [知乎上的一篇](https://www.zhihu.com/question/67287444/answer/251460831)
- [keras实现](https://github.com/XifengGuo/CapsNet-Keras/blob/master/capsulelayers.py)
- [tensorflow实现](https://github.com/naturomics/CapsNet-Tensorflow)

全部原创，转载请注明出处，谢谢~




















 