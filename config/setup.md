[Jetson AGX Orin Get Started](https://developer.nvidia.com/embedded/learn/get-started-jetson-agx-orin-devkit)

## YOLOV11 on Jetson AGX Orin Environment  Setup Note（Bill Liu Oct 25, 2025)

0. check cuda /tenosrt version
   CUDA version 

     ```shell
     nvidia-smi
     ```
     TensorRT version
     ```shell
     ldconfig -p | grep libnvinfer
     ```
     results: `NVIDIA-SMI 540.4.0                Driver Version: 540.4.0      CUDA Version: 12.6  `

1. Install pip

   `sudo apt install python3-pip`

2. Install jtop

   `sudo pip3 install -U jetson-stats`

3. Intall Jetpack 6.2.1+b38

   [Getting Started with Jetson AGX Orin Developer Kit | NVIDIA Developer](https://developer.nvidia.com/embedded/learn/get-started-jetson-agx-orin-devkit)

   - Using below command to install the jetpack, It can take about an hour to complete the installation (depending on the speed of your Internet connection).

   ```shell
   sudo apt update
   sudo apt dist-upgrade
   sudo reboot
   sudo apt install nvidia-jetpack
   ```

   - Check Jetpack version

   ```
   dpkg -l | grep -i 'jetpack'
   ```

   results `nvidia-jetpack         6.2.1+b38         arm64        NVIDIA Jetpack Meta Package`

   or use below command to check LT version

   ```
   cat /etc/nv_tegra_release
   ```

   results ` R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025`

4. Install ultralytics

   ```shell
   python3 -m pip install ultralytics
   sudo reboot
   ```

   

5. Reinstall opencv-python (not tested)

   On the output of "jtop=>INFO", it shows OpenCV: 4.12.0 with CUDA: NO so we need to reinstall OpenCV with CUDA enabled.

   ```shell
   wget https://github.com/Qengineering/Install-OpenCV-Jetson-Nano/raw/main/OpenCV-4-12-0.sh
   sudo chmod 755 ./OpenCV-4-12-0.sh
   ./OpenCV-4-10-0.sh
   sudo reboot
   ```

   

6. Unintall Torch and Torchvision 

   ```
   python3 -m pip uninstall torch
   python3 -m pip uninstall torchvision
   ```

   

7. Install Torch and torchvision with **CUDA** version
    [Installing PyTorch for Jetson Platform - NVIDIA Docs](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html#overview__section_orin)

> You can download different version of pytorch [==here==](https://developer.download.nvidia.cn/compute/redist/jp) or get file [==HERE==](https://pypi.jetson-ai-lab.io/jp6/cu126) (recommended) base on CUDA version 12.6

   ```shell
   sudo apt-get -y update; 
   sudo apt-get install -y  python3-pip libopenblas-dev;
   ```

6.1 Install Torch without VirtualEnv:

```shell
## export TORCH_INSTALL=https://nvidia.box.com/shared/static/i8pukc49h3lhak4kkn67tg9j4goqm0m7.whl 
export TORCH_INSTALL=torch-2.8.0-cp310-cp310-linux_aarch64.whl
export TORCHVISION_INSTALL=torchvision-0.23.0-cp310-cp310-linux_aarch64.whl
python3 -m pip install --upgrade pip; python3 -m pip install numpy==1.26.1; python3 -m pip install --no-cache $TORCH_INSTALL $TORCHVISION_INSTALL
```


6.2 Set up the Virtual Environment

- Install the `virtualenv` package and create a new Python 3 virtual environment:

```shell
sudo apt-get install virtualenv
python3 -m virtualenv -p python3 <chosen_venv_name>
```

- Activate the Virtual Environment

```shell
source <chosen_venv_name>/bin/activate
```

- Install the desired version of PyTorch & Torchvision:

```shell
export TORCH_INSTALL=torch-2.8.0-cp310-cp310-linux_aarch64.whl
export TORCHVISION_INSTALL=torchvision-0.23.0-cp310-cp310-linux_aarch64.whl
python3 -m pip install --upgrade pip; python3 -m pip install numpy==1.26.1; python3 -m pip install --no-cache $TORCH_INSTALL $TORCHVISION_INSTALL
```

- Install the desired version of Torchvision **(frank’s)**:

```shell
sudo apt install -y libjpeg-dev zlib1g-dev
git clone https://github.com/pytorch/vision.git
cd torchvision
git checkout v0.15.2
python3 setup.py install --user
```

- Deactivate the Virtual Environment

```shell
deactivate
```

Note： The best practice is ==NOT== use the Conda environment first, and the Flash the SSD is the key steps and make sure you select all the installation packages during the SDK manager Install 

7. Install Realsense SDK

   - Download the SDK from github

     ```shell
     git clone https://github.com/jetsonhacks/jetson-orin-librealsense.git
     cd jetson-orin-librealsense
     ```

   - Installing librealsense support kernel modules

     ```shell
     sha256sum -c install-modules.tar.gz.sha256
     tar -xzf install-modules.tar.gz
     cd install-modules
     sudo ./install-realsense-modules.sh
     ```

   - Installing librealsense

     Before getting started, unplug the RealSense camera.

     - Register the Intel Realsense public key:

       ```shell
       sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
       ```

       

     - Add the server to the list of repositories:

       ```shell
       sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u
       ```

       

     - Install the RealSense SDK:

       ```shell
       sudo apt-get install librealsense2-utils
       sudo apt-get install librealsense2-dev
       ```

       

     - reconnect the RealSense device and Run

       ```shell
       realsense-viewer
       ```

       

8.  Export PT weighs file to tensorrt

   ```shell
   # Export a YOLO11n PyTorch model to TensorRT format
   yolo export model=yolo11n.pt format=engine # creates 'yolo11n.engine''
   ```

   or use python export weight to TensorRT or Int8
   
   ```python
   from ultralytics import YOLO
   
   # Load the YOLO11 model
   model = YOLO("yolo11n.pt")
   
   # Export the model to TensorRT format
   model.export(format="engine") 
   
   # If you need increase the speed performance, you can Export to int8
   model.export(format="engine", batch=8, workspace=4, int8=True, data="coco.yaml")
   ```
   
   

## Others

[PyTorch for Jetson Platform - NVIDIA Docs](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html#pytorch-jetson-rel)

## [JetPack Compatibility](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html#pytorch-jetson-rel__section_tzk_3gv_52b)

| PyTorch Version                                              | NVIDIA Framework [Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) | NVIDIA Framework [Wheel](https://developer.download.nvidia.com/compute/redist/jp/) | JetPack Version       |
| :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :-------------------- |
| [2.9.0a0+50eac811a6](https://github.com/pytorch/pytorch/commit/50eac811a68e63e96ad56c11c983bfe298a0bb8a) | 25.09                                                        | -                                                            | 7.0                   |
| [2.8.0a0+34c6371d24](https://github.com/pytorch/pytorch/commit/34c6371d24a350db754a90c75361cdcf48cc0e71) | 25.08                                                        | -                                                            | 7.0                   |
| [2.8.0a0+5228986c39](https://github.com/pytorch/pytorch/commit/5228986c395dc79f90d2a2b991deea1eef188260) | 25.06                                                        | -                                                            | 6.2                   |
| [2.8.0a0+5228986c39](https://github.com/pytorch/pytorch/commit/5228986c395dc79f90d2a2b991deea1eef188260) | 25.05                                                        | -                                                            | 6.2                   |
| [2.7.0a0+79aa17489c](https://github.com/pytorch/pytorch/commit/79aa17489c3fc5ed6d5e972e9ffddf73e6dd0a5c) | 25.04                                                        | -                                                            | 6.2                   |
| [2.7.0a0+7c8ec84dab](https://github.com/pytorch/pytorch/commit/7c8ec84dab7dc10d4ef90afc93a49b97bbd04503) | 25.03                                                        | -                                                            | 6.2                   |
| [2.7.0a0+6c54963f75](https://github.com/pytorch/pytorch/commit/6c54963f75e9dfdae34c44f71081b5d3972b6b8d) | 25.02                                                        | -                                                            | 6.2                   |
| [2.6.0a0+ecf3bae40a](https://github.com/pytorch/pytorch/commit/ecf3bae40a6f2f0f3b237bde1fc4b2492765ab13) | 25.01                                                        | -                                                            | 6.1                   |
| [2.6.0a0+df5bbc09d1](https://github.com/pytorch/pytorch/commit/df5bbc09d191fff3bdb592c184176e84669a7157) | 24.12                                                        | -                                                            | 6.1                   |
| [2.6.0a0+df5bbc0](https://github.com/pytorch/pytorch/commit/df5bbc09d191fff3bdb592c184176e84669a7157) | 24.11                                                        | -                                                            | 6.1                   |
| [2.5.0a0+e000cf0ad9](https://github.com/pytorch/pytorch/commit/e000cf0ad980e5d140dc895a646174e9b945cf26) | 24.10                                                        | -                                                            | 6.1                   |
| [2.5.0a0+b465a5843b](https://github.com/pytorch/pytorch/commit/b465a5843b92f33fe3e89ff7ee91c6833df6aec0) | 24.09                                                        | 24.09                                                        | 6.1                   |
| [2.5.0a0+872d972e41](https://github.com/pytorch/pytorch/commit/872d972e41596a9ac94dfd343f40bfc12b340a74) | 24.08                                                        | -                                                            | 6.0                   |
| [2.4.0a0+3bcc3cddb5](https://github.com/pytorch/pytorch/commit/3bcc3cddb580bf0f0f1958cfe27001f236eac2c1) | 24.07                                                        | 24.07                                                        | 6.0                   |
| [2.4.0a0+f70bd71a48](https://github.com/pytorch/pytorch/commit/f70bd71a48) | 24.06                                                        | 24.06                                                        | 6.0                   |
| [2.4.0a0+07cecf4168](https://github.com/pytorch/pytorch/commit/07cecf4168503a5b3defef9b2ecaeb3e075f4761) | 24.05                                                        | 24.05                                                        | 6.0                   |
| [2.3.0a0+6ddf5cf85e](https://github.com/pytorch/pytorch/commit/6ddf5cf85e3c27c596175aba7bf5affb5426255f) | 24.04                                                        | 24.04                                                        | 6.0 Developer Preview |
| [2.3.0a0+40ec155e58](https://github.com/pytorch/pytorch/commit/40ec155e58ee1a1921377ff921b55e61502e4fb3) | 24.03                                                        | [24.03](https://developer.download.nvidia.com/compute/redist/jp/v60dp/pytorch/torch-2.3.0a0+40ec155e58.nv24.03.13384722-cp310-cp310-linux_aarch64.whl) |                       |
| [2.3.0a0+ebedce2](https://github.com/pytorch/pytorch/commit/ebedce24ab578036dd9257e4928eea9ee38d1192) | 24.02                                                        | 24.02                                                        |                       |
| [2.2.0a0+81ea7a4](https://github.com/pytorch/pytorch/commit/81ea7a48) | 23.12, 24.01                                                 | 23.12, 24.01                                                 |                       |
| [2.2.0a0+6a974bec](https://github.com/pytorch/pytorch/commit/6a974bec) | 23.11                                                        | 23.11                                                        |                       |
| [2.1.0a](https://github.com/pytorch/pytorch/commit/41361538a978eb03fa1e88bf5b8e4410db7a6927) |                                                              | 23.06                                                        | 5.1.x                 |
| [2.0.0](https://github.com/pytorch/pytorch/tree/v2.0.0)      |                                                              | 23.05                                                        |                       |
| [2.0.0a0+fe05266f](https://github.com/pytorch/pytorch/commit/fe05266fda4f908130dea7cbac37e9264c0429a2) |                                                              | 23.04                                                        |                       |
| [2.0.0a0+8aa34602](https://github.com/pytorch/pytorch/commit/8aa34602f703896c16ae57f622ff4cb1c86c04dd) |                                                              | 23.03                                                        |                       |
| [1.14.0a0+44dac51c](https://github.com/pytorch/pytorch/commit/44dac51c36d01f63e64585e5e7a864cb8e37948a) |                                                              | 23.02, 23.01                                                 |                       |
| [1.13.0a0+936e930](https://github.com/pytorch/pytorch/commit/936e930) |                                                              | 22.11                                                        | 5.0.2                 |
| [1.13.0a0+d0d6b1f](https://github.com/pytorch/pytorch/commit/d0d6b1f) |                                                              | 22.09, 22,10                                                 |                       |
| [1.13.0a0+08820cb](https://github.com/pytorch/pytorch/commit/08820cb) | 22.07                                                        | 22.07                                                        |                       |
| [1.13.0a0+340c412](https://github.com/pytorch/pytorch/commit/340c412) | 22.06                                                        | 22.06                                                        | 5.0.1                 |
| [1.12.0a0+8a1a93a9](https://github.com/pytorch/pytorch/commit/8a1a93a9) | 22.05                                                        | 22.05                                                        | 5.0                   |
| [1.12.0a0+bd13bc66](https://github.com/pytorch/pytorch/commit/bd13bc66) |                                                              | 22.04                                                        |                       |
| [1.12.0a0+2c916ef](https://github.com/pytorch/pytorch/commit/2c916ef) |                                                              | 22.03                                                        |                       |
| [1.11.0a0+bfe5ad28](https://github.com/pytorch/pytorch/commit/bfe5ad28) |                                                              | 22.01                                                        | 4.6.1                 |





## Question & Solution

### 1. 无法打开启动 Firefox & Chromium

快速恢复（回退 snapd + 冻结升级）

1. 下载旧版并安装（2.68.5 / rev 24724）

```shell
snap download snapd --revision=24724
sudo snap ack snapd_24724.assert
sudo snap install snapd_24724.snap
```

2. 暂停自动更新（任选其一）

```shell
# 冻结整个系统的 snaps（最省事）
sudo snap refresh --hold

# 只冻结 snapd 自身（更精细）
sudo snap refresh --hold snapd
```

3. 验证

```
snap --version
# 看到 snap/snapd 约为 2.68.5 后，重开 Firefox/Chromium 应能正常。
```



### 2. Install NoMachine for Remote GUI Access

[Downloads – Download App tools](https://download.nomachine.com/download/?id=1&platform=linux)

### 3. Jetson网线连接Windows PC,  实现VPN

#### 第1步：在Windows电脑上设置网络共享

1. 打开网络连接设置：在Windows搜索框输入“查看网络连接”并进入。
2. 配置主网络适配器：找到代表您当前活动网络的连接（比如“WLAN”或“以太网”），右键点击其“属性”。切换到“共享”选项卡。勾选“允许其他网络用户通过此计算机的Internet连接来连接”。在下方的“家庭网络连接”下拉框中，选择您将要与Jetson连接的有线网卡（通常名为“以太网”或“本地连接”）。点击“确定”。
3. 配置Clash：打开Clash，在设置中开启“允许局域网连接（Allow Lan）”。记下Clash界面中显示的端口号（通常是 `7890`）和您Windows电脑的IP地址（在CMD中运行 `ipconfig`查看，例如 `192.168.137.1`）。

#### 第2步：连接设备并配置Jetson

1. **物理连接**：用网线将Windows PC和Jetson AGX Orin直接连接起来。
2. **在Jetson上配置网络**：方法A（动态获取IP，推荐）：在Jetson的终端中执行以下命令，让它自动从Windows电脑获取IP地址。`sudo dhclient eth0 `

```
sudo dhclient eth0
```

#### 第3步：在Jetson上配置代理

要让Jetson的流量经过Clash，需要设置系统代理。

1. 在Jetson的Ubuntu系统中，打开“**设置**” -> “**网络**” -> “**网络代理**”。
2. 选择“手动”配置模式。
3. 在HTTP、HTTPS、FTP代理服务器地址中，填写您Windows电脑的IP地址（如 `192.168.137.1`）。
4. 在端口号中，填写Clash的端口（如 `7890`）。Socks主机同样填写此地址和端口。
5. 保存设置。

#### 如何测试连接是否成功？

1. 基础连通性测试：在Jetson的终端中，依次执行以下命令：

   `1. 测试能否ping通Windows电脑 ping 192.168.137.1  # 请替换为你的Windows电脑IP `

   `2. 测试能否ping通外网（不经过代理） ping 8.8.8.8 `

   `3. 测试代理和域名解析（经过代理） curl -I https://www.google.com`

   如果第1步通，第2步不通，说明网络共享可能有问题。如果第1、2步通，第3步不通，说明代理设置有问题。

2. 检查IP：在Jetson上运行 `ip addr show eth0`（或有线网卡对应名称），确认IP地址在Windows电脑的共享网段内（如 `192.168.137.x`）。

#### 常见问题与解决方案（故障排查）

- Jetson无法ping通Windows电脑，检查防火墙**：**临时关闭Windows防火墙和Jetson的防火墙（`sudo ufw disable`）进行测试。检查共享设置：确认Windows上的网络共享已正确启用，并选择了正确的以太网卡。检查网线：尝试更换网线或端口。
- Jetson能ping通Windows但不能ping通 `8.8.8.8`这通常意味着Windows的网络共享或路由有问题。请重新检查Windows上的Internet连接共享设置，并重启“Internet Connection Sharing (ICS)”服务。
- Jetson能ping通 `8.8.8.8`但无法通过浏览器访问网页检查代理配置：确认Jetson上的代理服务器地址和端口号完全正确。检查Clash设置：确认Clash的“允许局域网连接”已开启，且规则没有阻止Jetson的IP。检查DNS：尝试在Jetson上手动设置DNS服务器（如 `8.8.8.8`）。

