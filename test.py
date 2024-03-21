
from time import sleep
import threading

lock = threading.RLock()


def open_calc(t):
    # 加锁
    lock.acquire()
    try:
        # 需要保证线程安全的代码
        while t.i < 5:
            print(threading.current_thread().name, t.i)
            sleep(1)
            t.i += 1
    # 使用finally 块来保证释放锁
    finally:
        # 修改完成，释放锁
        lock.release()


def open_mstsc(t):
    while t.i < 5:
        try:
            # 加锁
            lock.acquire()
            # 需要保证线程安全的代码
            print(threading.current_thread().name, t.i)
            t.i += 1
            # 使用finally 块来保证释放锁
        finally:
            # 修改完成，释放锁
            lock.release()
        sleep(1)


def tf(te):
    print(te.i)


class Te:
    def __init__(self):
        self.i = 5

    def ttt(self):
        tf(self)


if __name__ == '__main__':
    """
    python多线程只能并发，不能并行，并行需要用其他语言
    如果涉及共享变量，才需要加锁，加锁就是用一个全局或公共对象，调threading.RLock()，注意，全局变量不行，只能是object类型
    用了锁，acquire和release紧夹操作，这个操作会在一个线程完全执行完后，另一个线程才会去执行
    博客参考：
    https://blog.51cto.com/u_15792201/5678079
    http://c.biancheng.net/view/2617.html
    
    # 没有共享变量，不用加锁
    threads = [threading.Thread(target=run_live, args=(RMQAsset.asset_generator('000001', ['5', '15', '30', '60', 'd'], 'index'),)),
               threading.Thread(target=run_live, args=(RMQAsset.asset_generator('515790', ['5', '15', '30', '60', 'd'], 'ETF'),))]
    for t in threads:
        # 启动线程
        t.start()
    
    这段代码弃用的原因是：threading是多线程，python的线程只能并发，不能并行，即使在多核的机器上，也只用一个核
    这导致我开11个线程，第11个一直执行不了，而且在结束运行时，一直结束不了，被卡住了
    查资料后发现是全局解释器锁（GIL）导致的，解决办法只能用多进程，或者换语言，Java没这个问题
    资料参考：https://www.zhihu.com/question/23474039/answer/269526476

    替换方案：
    multiprocessing
    多进程执行

    
    扫描项目生成依赖清单
    在项目的根目录下 使用 pipreqs ./ --encoding=utf8
    根据依赖清单安装模块
    pip install -r requirements.txt
    
    下载docker，docker下载python镜像，docker pull python:3.8.2
    然后把镜像启动，就变成了容器，docker run -it python:3.8.2
    然后启动容器，docker start 容器id，以后每次用都是先start容器id
    一个容器相当玉一个linux系统，用python镜像做的容器，相当于一个linux系统里只装了一个python，
    用Java镜像做的容器，相当于一个linux系统里只装了一个jdk，
    用mysql镜像做的容器，相当于一个linux系统里只装了一个mysql，
    
    我用docker exec -it 容器id /bin/bash 进入我的python3.8.2容器，里面只装了python，用docker cp 把代码复制到容器里，就像在linux里操作一样，
    用python运行代码即可
    新的容器，bash进入，没有vi，没有yum，要通过两个命令先装
    apt-get update
    apt-get install vim
    然后把pip的数据源改成国内的  由于上面两个安装太慢，所以直接创建好文件复制进容器
    mkdir ~/.pip #如果目录不存在就用这句话，创建完后再cd到目录。 .pip代表隐藏文件pip
    touch pip.conf #创建pip的配置文件
    sudo vi ~/.pip/pip.conf #编辑文件
    
    [global] 
    index-url = http://mirrors.aliyun.com/pypi/simple/
    [install]
    trusted-host = http://mirrors.aliyun.com/pypi/simple/
    # 上面的trusted-host参数解决可能会提示不受信任的问题
    # 如果上面没用，就直接用下面这个运行
    pip install -r requirements.txt --trusted-host mirrors.aliyun.com
    # 就算pip 单独安装包，也要加--trusted-host mirrors.aliyun.com
    
    以后为了方便，不能手动复制，而是把 装python3.8.2 复制代码，安装pip库，每天定时运行代码，写成dockerfile
    FROM python:3.8.2
    COPY ./RobotMeQ /home/RobotMeQ
    RUN cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
    RUN pip install -r /home/RobotMeQ/Configs/requirements.txt --trusted-host mirrors.aliyun.com
    
    做成镜像，再把镜像变成容器，容器运行起来，不再关机，然后代码就会每天自动执行
    
    一个新服务器，安装docker : yum update更新一下库，然后执行 curl -sSL https://get.daocloud.io/docker | sh 一键安装，
    执行 service docker start 启动服务      systemctl enable docker 开机启动
    改tools的代码
    把dockerfile以及四个文件放到服务器上，运行，docker build -t python:3.8.2 .
    我的笔记本build没问题，但台式机build的半个小时一直没好，我换了docker的镜像源才行，下面是我的阿里云镜像原，其他国内镜像都访问不了了
    在 /etc/docker/daemon.json 中写入如下内容（如果文件不存在请新建该文件）
    {
    "registry-mirrors":["https://0hc2yp52.mirror.aliyuncs.com"]
    }
    之后重新启动服务：
    $ sudo systemctl daemon-reload
    $ sudo systemctl restart docker

    再执行上面的docker build就可以了

    docker run -itd python:3.8.2
    docker exec -it 容器id /bin/bash
    
    
        aliyun 37个标的
        docker start 26dbf8178821
        docker exec -it 26dbf8178821 /bin/bash
        nohup python -u /home/RobotMeQ/Run_prd_ETF_A.py >> /home/log.out 2>&1 &
        
        以后改代码，只把指定的代码复制到服务器，然后docker cp复制到容器
        docker cp /root/RobotMeQ/Run_prd_ETF_A.py 26dbf8178821:/home/RobotMeQ/Run_prd_ETF_A.py
        docker cp /root/RobotMeQ/requirements.txt 26dbf8178821:/home/RobotMeQ/requirements.txt

        docker cp /root/RobotMeQ/QuantData/live 26dbf8178821:/home/RobotMeQ/QuantData/live2
        docker cp /root/RobotMeQ/RMQTool/Message.py 26dbf8178821:/home/RobotMeQ/RMQTool/Message.py
    
        阿里云 37个标的
        docker start 06acc8ba6062
        docker exec -it 06acc8ba6062 /bin/bash
        nohup python -u /home/RobotMeQ/Run_prd_ETF_A.py >> /home/log.out 2>&1 &
        
        以后改代码，只把指定的代码复制到服务器，然后docker cp复制到容器
        docker cp /root/RobotMeQ/Run_prd_ETF_A.py 06acc8ba6062:/home/RobotMeQ/Run_prd_ETF_A.py

        docker cp /root/RobotMeQ/QuantData/live 06acc8ba6062:/home/RobotMeQ/QuantData/live2
        docker cp /root/RobotMeQ/RMQTool/Message.py d63f10ba76df:/home/RobotMeQ/RMQTool/Message.py

    
    tail -f /home/log.out
    26个全放进来，占600M  cpu70，还能加策略。
    36 830m cpu 96
    
    
    在conda的导航工具里新建环境，或者用pycharm新建环境，然后给项目选择需要的解释器，
    安装包时，conda的包不全，进入conda环境，然后执行pip就行了，执行requirement也是进了conda的环境再执行
    conda activate robotme
    由于没配conda的环境变量，要到anaconda安装目录的condabin下执行conda
    退出是conda deactivate
    我现在是两个环境并存，先是python安装了3.8，在c盘，配了全局代理，C:\ProgramData\pip\pip.ini
    给pycharm指定的解释器也是这个，用pycharm打开新的python项目时，会自动用virtual虚拟环境
    2023，0820装了anaconda，就把项目环境改成了anaconda的自己新建的env环境的python解释器。
    anaconda的多个python版本可以并存，我自己的python3.8也是和他们并存的
    只是有一点，我的python开了pip的代理，anaconda没有，不过anaconda包少，用conda安装包和进入conda的自己的环境，再用pip安装是一样的
    按理说pip.ini是通用的
        通过 pip config list 可以看到anaconda和全局用的是一个pip.ini，vpn的代理生效了。
        
    
    创建新项目得步骤
    先用conda建环境
    再到github创建项目，添加gitignore,拿到链接（不拿也行）
    本地用pycharm打开git点clone，放链接（别人的得拿，自己的项目不拿也行），选本地路径，改成conda的环境
    导入后放入自己得代码，点add，再commit，最后push
    
    conda list -e > requirements.txt
    conda install --yes --file requirements.txt
     """

    # t = Te()
    print(123)

    # print(t.ttt())
    # # 使用threading模块，threading.Thread()创建线程，其中target参数值为需要调用的方法，同样将其他多个线程放在一个列表中，遍历这个列表就能同时执行里面的函数了
    # for t in threads:
    #     # 启动线程
    #     t.start()

