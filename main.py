import os
import re
import sys
import json
import logging
import subprocess
import datetime
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Set, Tuple, List

# 导入 MCP 核心库
from mcp.server.fastmcp import FastMCP, Context

# 导入 LangChain 和 Pydantic
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_core.language_models.base import BaseLanguageModel
from pydantic import BaseModel, Field

# --- 提示词定义 ---
CODE_GENERATION_PROMPT = """
你是一个顶级的Python自动化专家 。你的任务是根据用户的自然语言指令，生成一段完整、健壮、可直接在标准Python环境中执行的脚本。你拥有完全的创作自由，但必须严格遵守以下规范。

## 用户指令:
{task}

## 代码生成规范 (必须严格遵守):
1.  **【代码纯净性】**: 你的输出必须是纯粹的Python代码。绝对禁止包含任何Markdown标记，尤其是 ` ```python ` 和 ` ``` `。
2.  **【依赖声明】**: 如果代码需要任何第三方库 (例如 `requests`, `pandas`)，必须在代码的最开始，使用 `# REQUIRE: <package_name>` 的格式进行声明。**每个依赖独立一行**。如果不需要任何第三方库，则完全不要写 `# REQUIRE:` 注释。
3.  **【日志记录】**: 必须使用Python的 `logging` 模块。在脚本开始处配置好 `basicConfig`，确保日志同时输出到控制台(stdout)和当前工作目录下的 `task.log` 文件。在关键步骤和任何 `except` 块中，都必须使用 `logging.info()` 或 `logging.error()` 进行记录。
4.  **【错误处理】**: 所有可能失败的操作都必须被包含在 `try...except Exception as e:` 块中。
5.  **【成功信号】**: 在脚本所有操作成功完成的最后，必须调用 `print("任务成功完成")`。
6.  **【完整性】**: 生成的代码必须是完整的、自包含的，包含所有必要的 `import` 语句。

现在，请根据用户指令生成代码。
"""

# --- 全局日志配置 ---
global_logger = logging.getLogger("mcp_service")
if not global_logger.handlers:
    global_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    global_logger.addHandler(handler)

# --- MCP 服务器定义 ---
mcp = FastMCP(name="SkyShadowHero Task Execution Server")

# --- Pydantic 返回模型 ---
class CodeGenerationResult(BaseModel):
    status: str = Field(description="代码生成阶段的状态 ('success' 或 'failed')")
    code: str = Field(description="生成或提供的Python代码")
    dependencies: List[str] = Field(description="从代码中提取的依赖库列表")
    work_dir: str = Field(description="为本次任务创建的工作目录路径")
    error: Optional[str] = Field(None, description="如果生成失败，此字段包含错误信息")

class DependencyInstallationResult(BaseModel):
    status: str = Field(description="依赖安装阶段的状态 ('success', 'failed', 或 'skipped')")
    installed_packages: List[str] = Field(description="成功安装的包列表")
    work_dir: str = Field(description="执行安装的工作目录")
    output: str = Field(description="pip install 命令的输出")
    error: Optional[str] = Field(None, description="如果安装失败，此字段包含错误信息")

class ExecutionResult(BaseModel):
    status: str = Field(description="代码执行阶段的状态 ('success' 或 'failed')")
    output: str = Field(description="脚本执行的标准输出 (stdout)")
    error: str = Field(description="脚本执行的标准错误 (stderr)")
    returncode: int = Field(description="脚本执行的返回码")
    work_dir: str = Field(description="执行代码的工作目录")

# --- 配置管理 (MCPConfig) ---
class MCPConfig:
    _instance = None
    config_path = Path(__file__).parent / "mcp_config.json"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.load_config()
        return cls._instance

    def load_config(self):
        if not self.config_path.exists():
            self.config = {
                "llm_config": {
                    "model": "deepseek-coder-v2:16b",
                    "base_url": "http://127.0.0.1:11434",
                    "api_key": "ollama"
                }
            }
            self.save_config( )
        else:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)

    def save_config(self):
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)

    def get_llm_config(self) -> Optional[Dict[str, Any]]:
        return self.config.get("llm_config")

# --- 核心逻辑 (TaskWorkflow) ---
class TaskWorkflow:
    def __init__(self):
        self.llm_config = MCPConfig().get_llm_config()
        self.llm_instance: Optional[BaseLanguageModel] = None
        self.standard_libs = self._get_standard_libs()
        script_dir = Path(__file__).parent.resolve()
        self.shared_work_dir = script_dir / "mcp_tasks"
        self.shared_work_dir.mkdir(exist_ok=True)

        global_logger.info("正在初始化并检查共享虚拟环境...")
        try:
            self.venv_path = self.shared_work_dir / "venv"
            self.python_executable, self.pip_executable = self._create_virtual_env(self.venv_path)
            global_logger.info(f"共享虚拟环境已就绪。Python: {self.python_executable}, Pip: {self.pip_executable}")
        except Exception as e:
            global_logger.error(f"初始化共享虚拟环境失败: {e}", exc_info=True)
            raise RuntimeError(f"无法创建或验证共享虚拟环境，服务无法启动。错误: {e}")

    def _get_standard_libs(self) -> Set[str]:
        common_libs = {'os', 'sys', 'json', 're', 'logging', 'subprocess', 'pathlib', 'datetime', 'time', 'math', 'random', 'collections', 'itertools', 'functools', 'glob', 'shutil', 'tempfile', 'argparse', 'typing', '__future__'}
        if sys.version_info >= (3, 10):
            try:
                from sys import stdlib_module_names
                return set(stdlib_module_names)
            except ImportError:
                return common_libs
        return common_libs

    def _create_virtual_env(self, venv_path: Path) -> Tuple[str, str]:
        """
        创建或验证共享虚拟环境，并返回平台兼容的python和pip可执行文件路径。
        """
        if sys.platform == "win32":
            bin_dir = venv_path / "Scripts"
        else:
            bin_dir = venv_path / "bin"
            
        python_exe = bin_dir / "python.exe" if sys.platform == "win32" else bin_dir / "python"
        pip_exe = bin_dir / "pip.exe" if sys.platform == "win32" else bin_dir / "pip"

        if not python_exe.exists() or not pip_exe.exists():
            global_logger.info(f"共享虚拟环境不完整或不存在，正在创建于: {venv_path}")
            try:
                # 使用 sys.executable 确保我们用当前 Python 版本来创建 venv
                subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True, capture_output=True, text=True, timeout=120, encoding='utf-8')
            except subprocess.CalledProcessError as e:
                global_logger.error(f"创建虚拟环境失败: {e.stderr}")
                raise RuntimeError(f"创建虚拟环境失败，错误: {e.stderr}")

        if not python_exe.exists() or not pip_exe.exists():
             raise FileNotFoundError(f"虚拟环境创建后，在 {bin_dir} 中未找到 Python/Pip 可执行文件。")
        
        global_logger.info("虚拟环境验证成功。")
        return str(python_exe), str(pip_exe)

    def _post_process_code(self, generated_code: str) -> Tuple[str, Set[str]]:
        cleaned_code = re.sub(r"```python\n|```", "", generated_code).strip()
        required_deps = set(re.findall(r"#\s*REQUIRE:\s*(\S+)", cleaned_code))
        final_code = "\n".join([line for line in cleaned_code.split('\n') if not line.strip().startswith("# REQUIRE:")])
        global_logger.info(f"代码后处理完成。提取的依赖: {required_deps or '无'}。")
        return final_code.strip(), required_deps

    def _create_task_work_dir(self) -> Path:
        timestamp = datetime.datetime.now().strftime("task_%Y%m%d_%H%M%S")
        task_work_dir = self.shared_work_dir / timestamp
        task_work_dir.mkdir(exist_ok=True)
        global_logger.info(f"任务工作目录已创建: {task_work_dir}")
        return task_work_dir

    async def get_llm(self) -> BaseLanguageModel:
        if self.llm_instance:
            return self.llm_instance

        if not self.llm_config:
            raise ValueError("LLM 配置 'llm_config' 在 mcp_config.json 中未找到。")

        model_name = self.llm_config.get("model")
        base_url = self.llm_config.get("base_url")
        api_key = self.llm_config.get("api_key")

        if not model_name: raise ValueError("必须在配置中提供 'model'。")
        if not base_url: raise ValueError("必须在配置中提供 'base_url'。")
        if not api_key: raise ValueError("必须在配置中提供 'api_key'。")

        is_ollama = api_key.lower() == "ollama"
        provider_type = "Ollama" if is_ollama else "Standard API"
        global_logger.info(f"正在加载模型: {model_name} (类型: {provider_type}, URL: {base_url})")

        original_proxies = {}
        proxy_keys = ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]
        for key in proxy_keys:
            if key in os.environ:
                original_proxies[key] = os.environ.pop(key )
                global_logger.info(f"临时移除代理环境变量: {key}")

        try:
            llm: BaseLanguageModel
            if is_ollama:
                llm = Ollama(model=model_name, base_url=base_url, temperature=0.1, top_p=0.9, timeout=300)
            else:
                if "YOUR_" in api_key: raise ValueError("请在 mcp_config.json 中配置有效的 'api_key'。")
                llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url, temperature=0.1, max_retries=2, timeout=300)
            
            self.llm_instance = llm
            return llm
        finally:
            for key, value in original_proxies.items():
                os.environ[key] = value
                global_logger.info(f"恢复代理环境变量: {key}")

workflow_executor = TaskWorkflow()

# --- MCP 工具函数 ---
@mcp.tool()
async def generate_code(
    instruction: str = Field(description="用户的自然语言指令，用于生成Python代码。"),
    ctx: Context = Field(exclude=True)
) -> CodeGenerationResult:
    await ctx.info(f"收到代码生成指令: '{instruction}'")
    try:
        task_work_dir = workflow_executor._create_task_work_dir()
        await ctx.info(f"工作目录已创建: {task_work_dir}")
        llm = await workflow_executor.get_llm()
        prompt = CODE_GENERATION_PROMPT.format(task=instruction)
        model_name = workflow_executor.llm_config.get('model')
        await ctx.info(f"正在使用模型 '{model_name}' 生成代码...")
        response = await llm.ainvoke(prompt)
        generated_code = response.content if hasattr(response, 'content') else str(response)
        await ctx.info("代码生成成功。")
        pure_code, dependencies = workflow_executor._post_process_code(generated_code)
        code_path = task_work_dir / "generated_script.py"
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(pure_code)
        global_logger.info(f"生成的代码已保存至: {code_path}")
        return CodeGenerationResult(
            status="success",
            code=pure_code,
            dependencies=list(dependencies),
            work_dir=str(task_work_dir)
        )
    except Exception as e:
        error_message = f"代码生成失败: {e}"
        global_logger.error(error_message, exc_info=True)
        await ctx.error(error_message)
        return CodeGenerationResult(
            status="failed", code="", dependencies=[], work_dir="", error=error_message
        )

@mcp.tool()
async def install_dependencies(dependencies: List[str] = Field(description="需要安装的Python包列表。"), work_dir: str = Field(description="必须提供由 'generate_code' 工具返回的工作目录路径。"), ctx: Context = Field(exclude=True)) -> DependencyInstallationResult:
    await ctx.info(f"收到依赖安装指令: {dependencies} in '{work_dir}'")
    task_work_dir = Path(work_dir)
    if not task_work_dir.exists():
        return DependencyInstallationResult(status="failed", installed_packages=[], work_dir=work_dir, output="", error="工作目录不存在！")
    if not dependencies:
        await ctx.info("依赖列表为空，跳过安装。")
        return DependencyInstallationResult(status="skipped", installed_packages=[], work_dir=work_dir, output="No dependencies to install.")
    deps_to_install = {dep for dep in dependencies if dep.lower() not in workflow_executor.standard_libs}
    if not deps_to_install:
        await ctx.info(f"所有声明的依赖 {dependencies} 均为标准库，无需安装。")
        return DependencyInstallationResult(status="skipped", installed_packages=[], work_dir=work_dir, output="All dependencies are standard libraries.")
    try:
        requirements_path = task_work_dir / "requirements.txt"
        with open(requirements_path, 'w', encoding='utf-8') as f:
            for dep in deps_to_install:
                f.write(f"{dep}\n")
        command = [workflow_executor.pip_executable, "install", "-r", str(requirements_path)]
        await ctx.info(f"执行依赖安装命令: {' '.join(command)}")
        result = subprocess.run(command, cwd=str(task_work_dir), capture_output=True, text=True, timeout=300, check=False, encoding='utf-8')
        if result.returncode != 0:
            error_message = f"依赖安装失败: {result.stderr}"
            global_logger.error(error_message)
            await ctx.error(error_message)
            return DependencyInstallationResult(status="failed", installed_packages=[], work_dir=work_dir, output=result.stdout, error=result.stderr)
        await ctx.info(f"依赖 {list(deps_to_install)} 安装成功。")
        return DependencyInstallationResult(status="success", installed_packages=list(deps_to_install), work_dir=work_dir, output=result.stdout)
    except Exception as e:
        error_message = f"安装依赖时发生意外错误: {e}"
        global_logger.error(error_message, exc_info=True)
        await ctx.error(error_message)
        return DependencyInstallationResult(status="failed", installed_packages=[], work_dir=work_dir, output="", error=error_message)

@mcp.tool()
async def execute_code(work_dir: str = Field(description="必须提供由 'generate_code' 工具返回的、包含 'generated_script.py' 的工作目录路径。"), ctx: Context = Field(exclude=True)) -> ExecutionResult:
    await ctx.info(f"收到代码执行指令 in '{work_dir}'")
    task_work_dir = Path(work_dir)
    script_path = task_work_dir / "generated_script.py"
    if not script_path.exists():
        error_msg = f"执行失败: 在工作目录 '{work_dir}' 中未找到 'generated_script.py'。"
        await ctx.error(error_msg)
        return ExecutionResult(status="failed", output="", error=error_msg, returncode=-1, work_dir=work_dir)
    try:
        command = [workflow_executor.python_executable, str(script_path)]
        await ctx.info(f"执行代码命令: {' '.join(command)}")
        result = subprocess.run(command, cwd=str(task_work_dir), capture_output=True, text=True, timeout=300, check=False, encoding='utf-8')
        is_successful = "任务成功完成" in result.stdout
        final_status = "success" if is_successful and result.returncode == 0 else "failed"
        await ctx.info(f"代码执行完成。状态: {final_status}。")
        return ExecutionResult(status=final_status, output=result.stdout, error=result.stderr, returncode=result.returncode, work_dir=work_dir)
    except Exception as e:
        error_message = f"执行代码时发生意外错误: {e}"
        global_logger.error(error_message, exc_info=True)
        await ctx.error(error_message)
        return ExecutionResult(status="failed", output="", error=error_message, returncode=-1, work_dir=work_dir)

# --- 服务器启动---
def run():
    """
    服务器主入口函数。
    """
    # 改进：在启动时打印用户指引
    print("\033[0;34m====================================================\033[0m")
    print("\033[1;33m               欢迎使用 MCP 执行服务器\033[0m")
    print("\033[0;34m====================================================\033[0m")
    print("\n\033[0;32m▶ 如何配置:\033[0m")
    print("  请检查并修改 \033[1m./mcp_config.json\033[0m 文件来配置您的大语言模型。")
    print("\n\033[0;32m▶ 如何使用:\033[0m")
    print("  请在支持 MCP 的客户端（如 Claude Desktop）中，")
    print("  使用 \033[1m./import_mcp.json\033[0m 文件中的内容来导入到UOS AI等支持mcp的客户端中。")
    print("\n\033[0;34m----------------------------------------------------\033[0m\n")

    # 使用 shutil.which 进行跨平台兼容的命令检查
    if shutil.which("ollama"):
        try:
            result = subprocess.run(["ollama", "list"], check=True, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                global_logger.info("Ollama 服务已检测到并正在运行。")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            output = e.stdout or e.stderr
            global_logger.warning(f"Ollama 命令存在，但执行 'ollama list' 失败: {output.strip()}")
            global_logger.warning("如果配置使用Ollama，请确保其服务正常运行。")
    else:
        global_logger.warning("在系统 PATH 中未找到 'ollama' 命令。如果需要使用 Ollama，请确保其已安装并配置好环境变量。")
  
    mcp.run()

if __name__ == "__main__":
    run()