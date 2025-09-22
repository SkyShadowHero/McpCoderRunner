#!/bin/bash

# --- 脚本设置 ---
# 如果任何命令失败，则立即退出
set -e

# --- 颜色定义 (用于美化输出) ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # 无颜色

echo -e "${BLUE}===================================================${NC}"
echo -e "${BLUE}  MCP Python 应用自动化安装与配置脚本  ${NC}"
echo -e "${BLUE}===================================================${NC}"

# --- 1. 检查环境 ---
echo -e "\n${YELLOW}[1/6] 正在检查环境...${NC}"
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "错误：未找到 Python。请先安装 Python 3。"
    exit 1
fi
PYTHON_CMD=$(command -v python3 || command -v python)

if ! $PYTHON_CMD -m pip --version &> /dev/null; then
    echo "错误：未找到 pip。请确保你的 Python 环境包含 pip。"
    exit 1
fi
echo -e "${GREEN}Python 和 pip 环境正常。${NC}"

# --- 2. 创建虚拟环境 ---
VENV_DIR="venv"
echo -e "\n${YELLOW}[2/6] 正在创建虚拟环境... (目录: ${VENV_DIR})${NC}"
if [ -d "$VENV_DIR" ]; then
    echo "虚拟环境目录 '${VENV_DIR}' 已存在，跳过创建步骤。"
else
    $PYTHON_CMD -m venv $VENV_DIR
    echo -e "${GREEN}虚拟环境创建成功。${NC}"
fi

# 定义虚拟环境中的可执行文件路径
VENV_PYTHON_PATH="$VENV_DIR/bin/python"
VENV_PIP_PATH="$VENV_DIR/bin/pip"

# --- 3. 安装依赖 ---
echo -e "\n${YELLOW}[3/6] 正在安装依赖项 (从 requirements.txt)...${NC}"
"$VENV_PIP_PATH" install -r requirements.txt
echo -e "${GREEN}所有依赖项已成功安装到虚拟环境中。${NC}"

# --- 4. 检测 Ollama ---
echo -e "\n${YELLOW}[4/6] 正在检测 Ollama 环境...${NC}"
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}检测到 'ollama' 命令。如果需要使用本地模型，请确保 Ollama 服务正在运行。${NC}"
else
    echo -e "${YELLOW}提示：未在系统中检测到 'ollama' 命令。如需使用本地大模型，请先安装 Ollama。${NC}"
fi

# --- 5. 生成并展示 MCP 导入配置 ---
CONFIG_FILE="import_mcp.json"
echo -e "\n${YELLOW}[5/6] 正在生成并展示 MCP 客户端导入配置文件 (${CONFIG_FILE})...${NC}"

if command -v realpath &> /dev/null; then
    SCRIPT_DIR=$(realpath "$(dirname "$0")")
else
    cd "$(dirname "$0")"
    SCRIPT_DIR="$(pwd)"
    cd - > /dev/null
fi

ABS_VENV_PYTHON_PATH="$SCRIPT_DIR/$VENV_DIR/bin/python"
ABS_MAIN_SCRIPT_PATH="$SCRIPT_DIR/main.py"

cat > "$CONFIG_FILE" << EOL
{
  "mcpServers": {
    "python+mcp": {
      "command": "${ABS_VENV_PYTHON_PATH}",
      "args": [
        "${ABS_MAIN_SCRIPT_PATH}"
      ]
    }
  }
}
EOL

echo -e "${GREEN}配置文件 ${CONFIG_FILE} 已成功生成！${NC}"
echo "你的 MCP 服务已准备就绪。请通过 MCP 客户端来启动和使用它。"
echo "你需要使用下面这个配置文件来进行导入："
echo -e "${YELLOW}${SCRIPT_DIR}/${CONFIG_FILE}${NC}"
echo ""
echo "脚本现在将为您打开此文件，方便您查看或复制其内容。"
read -p "按 [Enter] 键以打开 ${CONFIG_FILE}..."

# 尝试使用跨平台的命令打开文件
if command -v open &> /dev/null; then # macOS
    open "$CONFIG_FILE"
elif command -v xdg-open &> /dev/null; then # Linux
    xdg-open "$CONFIG_FILE"
else
    echo "无法自动打开文件。请手动打开以下文件："
    echo "$SCRIPT_DIR/$CONFIG_FILE"
fi

# --- 6. 引导用户配置并启动服务 ---
MCP_CONFIG="mcp_config.json"
echo -e "\n${YELLOW}[6/6] 准备模型配置并启动服务...${NC}"

if [ ! -f "$MCP_CONFIG" ]; then
    echo "配置文件 ${MCP_CONFIG} 不存在，将通过运行主程序来自动创建默认配置..."
    "$VENV_PYTHON_PATH" -c "from main import MCPConfig; MCPConfig()"
    echo -e "${GREEN}默认配置文件已生成。${NC}"
fi

echo -e "---------------------------------------------------"
echo -e "在启动服务前，请最后确认你的模型配置。"
echo -e "脚本即将为您打开 ${MCP_CONFIG} 文件。"
echo -e "\n请根据您的选择进行修改："
echo -e "  - ${GREEN}如果使用 Ollama:${NC}"
echo -e '    "model": "deepseek-coder-v2:16b", (或其他Ollama模型)'
echo -e '    "base_url": "http://127.0.0.1:11434",'
echo -e '    "api_key": "ollama"'
echo -e "  - ${GREEN}如果使用在线 API (如 DeepSeek ):${NC}"
echo -e '    "model": "deepseek-chat",'
echo -e '    "base_url": "https://api.deepseek.com",'
echo -e '    "api_key": "sk-xxxxxxxxxxxxxxxxxxxx"'
echo -e "---------------------------------------------------"

read -p "按 [Enter] 键继续 ，以打开配置文件进行编辑..."

# 改进：使用与 import_mcp.json 相同的方式打开 mcp_config.json
if command -v open &> /dev/null; then # macOS
    open "$MCP_CONFIG"
elif command -v xdg-open &> /dev/null; then # Linux
    xdg-open "$MCP_CONFIG"
else
    echo "无法自动打开文件。请手动打开 ${MCP_CONFIG} 文件进行配置。"
fi

# 增加一个等待用户确认的步骤，因为图形化编辑器是非阻塞的
read -p "请在编辑器中完成配置，保存并关闭文件后，按 [Enter] 键以启动服务..."

echo -e "${GREEN}模型配置完成。${NC}"
echo -e "\n${GREEN}✅ 所有设置已完成！即将启动 MCP 服务...${NC}"
echo "服务将在前台运行。你可以按 Ctrl+C 来停止服务。"
echo "---------------------------------------------------"

# 启动服务
"$VENV_PYTHON_PATH" main.py
