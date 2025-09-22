# MCP 自动化任务执行器

一个基于 MCP (Model Context Protocol) 的自动化任务执行应用，能够理解你的自然语言指令，自动生成、安装依赖并执行 Python 脚本来完成任务。

> **它是如何工作的？**  
在mcp客户端下达指令后（例如“帮我获取今天的天气并保存到 a.txt”），此mcp会执行以下步骤：  
通过配置的大语言模型（本地ollama或者api）生成相应的 Python 代码  
在环境中安装相应需要的依赖 
在虚拟环境中执行这段代码以完成任务 

---

## 使用教程

### 准备工作

*   确保你的电脑上已经安装了 **Python 3.8+**。

### 快速开始 (仅限 Linux)

对于 Linux 用户，提供了简易的一键安装配置脚本。

```bash
# 给予脚本执行权限
chmod +x first_launch.sh

# 运行脚本
./first_launch.sh

脚本会自动完成环境创建、依赖安装、配置引导，并最终测试启动服务。
```
运行中会要求填写mcp_config.json(模型相关)和、并弹出import_mcp.json，可通过import_mcp.json里的示例导入进支持mcp的客户端如UOS Ai等

### 手动安装

## 第一步: 克隆项目

```Bash
git clone https://github.com/SkyShadowHero/python-mcp.git
cd python-mcp
```
## 第二步: 创建和进入虚拟环境
```Bash

### 创建虚拟环境 (推荐使用 uv)
#### 使用uv
uv venv
#### 使用python
python -m venv venv

### 进入环境
#### Windows (PowerShell)
.\venv\Scripts\Activate.ps1
#### Windows (CMD)
venv\Scripts\activate.bat
#### Mac / Linux
source venv/bin/activate

```
## 第三步: 安装依赖

```Bash
# 推荐使用 uv
uv pip install -r requirements.txt

# 或者使用 pip
pip install -r requirements.txt
```

## 第四步: 配置大语言模型 (LLM)

首次运行main.py，如果 **mcp_config.json** 文件不存在，它会被自动创建。  
打开 **mcp_config.json** 文件，根据以下内容进行修改。  
使用本地 Ollama:

```JSON
{
    "llm_config": {
        "model": "deepseek-coder-v2:16b",
        "base_url": "http://127.0.0.1:11434",
        "api_key": "ollama"
    }
}
```

使用在线 API (如 DeepSeek ):

```JSON
{
    "llm_config": {
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "api_key": "sk-xxxxxxxxxxxxxxxxxxxx"
    }
}
```
## 第五步: 测试运行 MCP 服务
```Bash
python main.py
```

## 第六步: 导入到 MCP 客户端


# 简单演示

图中为Ollama的deepseek-coder-v2:16b+UOS AI+这个mcp服务

## svg图片换色

把一个蓝色的b站小图标换为粉色

![.png](https://s2.loli.net/2025/08/06/HBxj9Kwy3WvbmOg.png)

## 文字生成

在一个文本里生成一些废话

![.png](https://s2.loli.net/2025/08/06/O9ELpVshiA8nmWv.png)

## 网络功能

抓取bing美图到本地

![.png](https://s2.loli.net/2025/08/06/gPcVODvZrYUsLRi.png)

## 复杂任务

将MiSans字体中的天影大侠提取出来绘制为白色图片并添加黑色边框

![.png](https://s2.loli.net/2025/08/06/oONgCixVzvScmql.png)

当然python有无限的可能

# 写在最后

此代码部分由llm完成，参考价值不大