# 模型提交指南：如何创建和导出Docker镜像

为保证所有参赛作品在统一、公平、可复现的环境下进行评测，本次比赛要求选手以Docker镜像（.tar格式文件）的形式提交模型，并统一使用 run.sh 脚本作为启动命令。本指南将详细说明如何将您的代码和模型打包成一个符合规范的Docker镜像。

目录
为什么使用Docker？
提交前准备：安装Docker
核心步骤：创建您的Docker镜像
Step 1: 准备项目文件结构
Step 2: 编写运行脚本 run.sh (核心)
Step 3: 编写 Dockerfile
Step 4: 编写您的模型预测程序 (例如 main.py)
Step 5: 构建Docker镜像
Step 6: 本地测试镜像
Step 7: 导出Docker镜像为 .tar 文件
一个完整的示例
最佳实践与注意事项
常见问题（FAQ）与故障排除
1. 为什么使用Docker？
Docker是一个开源的容器化平台，它能将您的应用程序及其所有依赖（代码、库、环境变量、配置文件等）打包到一个独立的、可移植的“容器”中。

对比赛而言，使用Docker有以下好处：

环境一致性：确保您的模型在评测服务器上的运行环境与您本地开发的环境完全一致，避免“在我机器上能跑”的问题。
依赖隔离：所有依赖项都封装在镜像内部，不会与评测机或其他选手的环境冲突。
部署便捷：评测系统可以自动化、标准化地运行和评估所有提交的作品。
2. 准备工作：安装Docker
在开始之前，请确保您的电脑上已经安装了Docker。

Windows/Mac: 下载并安装 Docker Desktop。
Linux: 根据您的Linux发行版，参考官方文档进行安装 (例如 Install Docker Engine on Ubuntu)。
安装完成后，打开终端或PowerShell，运行以下命令验证安装是否成功：

docker --version
如果能看到版本号，说明安装成功。

3. 核心步骤：创建您的Docker镜像
Step 1: 准备项目文件结构
一个清晰的文件结构能让后续步骤事半功倍。我们建议您将所有需要提交的文件放在一个单独的文件夹内。

推荐的目录结构：

submission/
├── Dockerfile              # Docker镜像的构建说明书
├── run.sh                  # (必须) 统一的容器启动脚本
├── requirements.txt        # Python依赖库列表
├── main.py                 # 您的模型预测主程序
├── weights/                # 存放模型权重的文件夹
│   └── best_model.pth
└── utils/                  # 其他辅助代码模块
    └── data_loader.py
评测输入/输出约定：评测系统运行时，会自动将输入数据挂载到容器的某个路径（例如 /input），并期望从容器的另一个路径（例如 /output）读取预测结果。这两个路径将作为参数传递给 run.sh 脚本。
Step 2: 编写运行脚本 run.sh (核心)
run.sh 是评测系统与您的代码交互的入口。评测机启动容器时，会执行这个脚本，并传入输入和输出路径作为参数。

在您的项目根目录下创建 run.sh 文件：

#!/bin/bash
# 评测系统会传入2个参数：
# $1: 输入数据文件夹的路径 (例如 /input)
# $2: 输出结果文件夹的路径 (例如 /output)
# 示例：直接执行Python脚本，并将参数传递给它
# 您可以在这里添加其他逻辑，例如解压模型、设置环境变量等
python main.py --input_path $1 --output_path $2
echo "Execution finished."
重要提示：这个脚本负责调用您的主程序，并正确传递参数。

Step 3: 编写 Dockerfile
Dockerfile 是一个文本文件，它包含了构建Docker镜像所需的所有指令。

# Step 1: 选择一个基础镜像
# 推荐使用官方的、轻量级的Python镜像。版本号请与您的开发环境保持一致。
# 如果比赛需要GPU，请选择NVIDIA官方的CUDA镜像，例如 nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
FROM python:3.9-slim
# Step 2: 设置工作目录
# 容器内所有后续操作都将在此目录下进行
WORKDIR /app
# Step 3: 复制项目文件到容器中
# 将当前目录下的所有文件复制到容器的/app目录下
COPY . .
# Step 4: 安装依赖
# 推荐使用requirements.txt来管理依赖，这样可以利用Docker的层缓存机制，加快构建速度。
# 使用国内镜像源可以大幅提升下载速度。
RUN pip install --no-cache-dir -r requirements.txt -i [https://pypi.tuna.tsinghua.edu.cn/simple](https://pypi.tuna.tsinghua.edu.cn/simple)
# Step 5: 赋予 run.sh 执行权限 (非常重要！)
RUN chmod +x run.sh
# Step 6: 定义容器启动时要执行的命令
# 使用 ENTRYPOINT 来指定 run.sh 作为启动脚本
ENTRYPOINT ["/bin/bash", "run.sh"]
Dockerfile ：

RUN chmod +x run.sh: 为 run.sh 脚本添加可执行权限，否则容器将无法运行它。
ENTRYPOINT ["/bin/bash", "run.sh"]: 将 run.sh 设置为容器的入口点。当容器启动时，这个脚本会被执行。评测系统传递的参数（输入/输出路径）会自动附加在该命令之后。
Step 4: 编写您的模型预测程序 (例如 main.py)
您的Python主程序需要能从命令行接收参数。

import argparse
import os
def predict(input_path, output_path):
    # 1. 在这里加载您的模型
    print("模型加载中...")
    # 2. 读取输入数据
    # 假设输入是一个名为 data.csv 的文件
    input_file = os.path.join(input_path, 'data.csv')
    print(f"正在读取输入文件: {input_file}")
    # 3. 进行模型预测
    print("模型预测完成！")
    # 4. 将结果写入指定的输出文件
    # 假设输出要求为 result.json
    output_file = os.path.join(output_path, 'result.json')
    print(f"正在写入结果文件: {output_file}")
    # 模拟写入一个文件
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(output_file, 'w') as f:
        f.write('{"prediction": "success"}')
    print("处理完成！")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 接收 run.sh 传入的参数
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input data')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the prediction results')
    args = parser.parse_args()
    predict(args.input_path, args.output_path)
Step 5: 构建Docker镜像
在包含 Dockerfile 的项目根目录下，打开终端，执行以下命令：

# docker build -t <镜像名称>:<标签> .
docker build -t my-ai-model:v2 .
Step 6: 本地测试镜像
这是提交前至关重要的一步！ 模拟评测环境来确保您的镜像能够正常工作。

在本地创建模拟的 input 和 output 文件夹。

mkdir -p local_test/input
mkdir -p local_test/output
echo "test_data" > local_test/input/data.csv
使用 docker run 命令启动容器进行测试。注意现在参数的传递方式。

# docker run --rm \
#   -v /path/to/your/local_test/input:/input \
#   -v /path/to/your/local_test/output:/output \
#   <镜像名称>:<标签> \
#   /input /output
# 示例 (请将 $(pwd) 替换为你的绝对路径):
docker run --rm \
  -v "$(pwd)/local_test/input:/input" \
  -v "$(pwd)/local_test/output:/output" \
  my-ai-model:v2 \
  /input /output
my-ai-model:v2: 这是您要测试的镜像。
/input /output: 这两个参数会传递给容器的 ENTRYPOINT，也就是 run.sh 脚本。在脚本内部，它们分别对应 $1 和 $2。
GPU测试：如果您的模型需要GPU，需添加 --gpus all 参数。
检查结果：运行结束后，检查您本地的 local_test/output 文件夹下是否生成了预期的结果文件（例如 result.json）。

Step 7: 导出Docker镜像为 .tar 文件
测试通过后，将镜像导出为 .tar 文件。

docker save -o my-ai-model-v2.tar my-ai-model:v2
my-ai-model-v2.tar 就是您需要提交的最终文件。

4. 一个完整的示例
# 目录结构
submission/
├── Dockerfile
├── run.sh
├── requirements.txt
└── main.py
# requirements.txt 内容
numpy==1.23.5
# run.sh 内容
#!/bin/bash
python main.py --input_path $1 --output_path $2
# main.py 内容
import argparse
import os
import numpy as np
def run(input_path, output_path):
    input_file = os.path.join(input_path, 'data.npy')
    output_file = os.path.join(output_path, 'result.npy')
    if os.path.exists(input_file):
        data = np.load(input_file)
        result = data + 1 # 示例处理
        os.makedirs(output_path, exist_ok=True)
        np.save(output_file, result)
        print("处理成功，结果已保存！")
    else:
        print(f"错误：输入文件 {input_file} 未找到！")
        exit(1)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True)
    parser.add_argument('--output_path', required=True)
    args = parser.parse_args()
    run(args.input_path, args.output_path)
# Dockerfile 内容
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN chmod +x run.sh
ENTRYPOINT ["/bin/bash", "run.sh"]
测试这个示例的完整流程：

创建测试文件夹和数据：mkdir -p local_test/input local_test/output 和 python -c "import numpy as np; np.save('local_test/input/data.npy', np.array([1,2,3]))"
进入 submission 目录，构建镜像：docker build -t example-model:latest .
运行并测试：docker run --rm -v "$(pwd)/../local_test/input:/input" -v "$(pwd)/../local_test/output:/output" example-model:latest /input /output
检查输出：python -c "import numpy as np; print(np.load('../local_test/output/result.npy'))"，应该会输出 [2 3 4]。
导出镜像：docker save -o example-model.tar example-model:latest
5. 最佳实践与注意事项
run.sh 的可靠性：确保您的 run.sh 脚本健壮。可以使用 set -e 让脚本在任何命令失败时立即退出，这有助于快速发现错误。
使用 .dockerignore 文件：在项目根目录创建一个名为 .dockerignore 的文件，将不需要复制到镜像中的文件（如 .git, __pycache__等）写进去。
减小镜像体积：选择 slim 或 alpine 版本的轻量级基础镜像；在 RUN 指令中清理缓存。
模型权重文件：直接将模型文件 COPY 到镜像中是最可靠的方式。
固定依赖版本：在 requirements.txt 中明确指定每个库的版本号。
6. 常见问题（FAQ）与故障排除
Q: docker run 时报错 permission denied 或 exec format error？

A: 这是最常见的问题。99% 的可能是您忘记在 Dockerfile 中添加 RUN chmod +x run.sh 这一行，导致脚本没有执行权限。请检查并重新构建镜像。
Q: 脚本收不到参数怎么办？

A: 检查本地测试时，docker run 命令的末尾是否正确添加了输入和输出路径，例如 /input /output。同时，在 run.sh 内部可以使用 echo "Input: $1, Output: $2" 来打印接收到的参数，方便调试。
Q: 构建时 pip install 很慢或失败怎么办？

A: 在 pip install 命令中添加 -i 参数使用国内镜像源，如上文示例所示。
Q: FileNotFoundError？

A: 请使用 docker run -it <image_name> /bin/bash 进入容器内部，用 ls -R 命令检查文件路径是否正确。确认您的 main.py 或 run.sh 中引用的路径是容器内的绝对路径（如 /app/weights）。
如果您在准备镜像的过程中遇到任何问题，请不要犹豫，及时与赛事主办方联系。祝您比赛顺利，取得优异成绩！
联系方式：aigks@sinopec.com