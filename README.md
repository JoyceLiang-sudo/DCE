# 分割小腿断层剖图6块肌肉

说明：Medical Image Processing课程project 2的源码和实现过程（有问题的可以私聊我，欢迎交流

目标：分割小腿断层剖图6块肌肉

源码：https://github.com/JoyceLiang-sudo/DCE （欢迎✨

1. 准备数据
    1. 总说
        - 这次任务很简单，所以只用了19张图片和对应的标注文件（其实我感觉10张就够了，但是没试过
        - 用matlab或者百度easydata标注图片，现在标注分割图片都是点几个点就可以了，都不用画轮廓
        - easydata有智能标注工具，只要手标10张，它就能帮你把剩下的图片都标注了（但是这里没必要要那么多张训练集
        - 模型输入是png图片，标注是COCO格式的
        
    2. 具体实现（python
        - 老师只给了一个128*128*247的mat格式文件，先把它分为247个mat，表示247张图片
          
            ```python
            def divide_mat():
                data = sio.loadmat('DCE_IMs.mat')
                ims = data['IMs']
                for img_id in range(247):
                    name = './mat/ims'+str(img_id)+'.mat'
                    scio.savemat(name, {'IMs': ims[:, :, img_id]})
            ```
            
        - mat转为png图片
          
            ```python
            def mat_png():
                path = './images/'
                data = sio.loadmat('DCE_IMs.mat')
                ims = data['IMs']
                pyplot.imshow(ims[:,:,0])
                pyplot.show()
                for img_id in range(247):
                    im = ims[:, :, img_id] /430 * 255
                    new_im = Image.fromarray(im.astype(np.uint8))
                    new_im.save(path+'ims'+str(img_id)+ '.png')  # 保存图片
            ```
            
        - 去easydata标注10张图片
          
            <img src="http://img.peterli.club/joy/Untitled.png" alt="Untitled" style="zoom: 25%;" />
            
            10张图片的标注信息都存在一个文件里
            
            ![Untitled](http://img.peterli.club/joy/Untitled%201.png)
    
2. 配置环境
    - 任务很小，不用拿gpu跑，我是拿了笔记本Apple M1 Pro 16GB，跑了200个epochs，用了一分钟都没有？差不多
    - 装anaconda 创虚拟环境这些基本操作这里就不说了，不懂的建议bing搜索或者来问我
    - 直接装requirements.txt里的package，pytorch装的是官方支持Mac加速的
      
        > `conda install pytorch torchvision torchaudio -c pytorch`
        > 
    - AI建议还是用linux或mac，配环境win分分钟会让你想死（无穷无尽的各种奇奇怪怪的问题，相对来说还是Linux好用，适配性好很多
3. 训练
    - config.yml 里面装的是各种参数，自调
    - 算法用的是segresnet，随便选的，这个小任务感觉随便找个算法效果都挺好的
    - 训练代码在train.py，直接跑这个就可以，要是想改算法，就改下图中的SegResNet吧啦吧啦那一行，换成别的算法
      
        ![Untitled](http://img.peterli.club/joy/Untitled%202.png)
    
4. 验证
    - model文件夹里存的是准确率最高的模型
    - 可以用logs里的文件可视化训练效果，loss和accuracy啥的，命令为（记得pip install tensorboard
      
        `tensorboard: python3 -m tensorboard.main --logdir ./ --bind_all`
        
    - visualization.py用来可视化分割效果的，随便找张不是训练集里的图片测试一下就行（尽力画的好看了

作业结束，非常简单

此处感谢李云灏先生的帮助，感恩！