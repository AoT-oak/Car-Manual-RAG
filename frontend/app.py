import streamlit as st
import requests
import time
import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

st.set_page_config(page_title="AI 汽车专家知识库", page_icon="🚗", layout="centered")

# 🌟 核心修改：前端也动态读取同一个环境变量，自动拼接后端地址
API_PORT = os.getenv("API_PORT", 8451)
API_BASE = f"http://127.0.0.1:{API_PORT}/api/v1"


# 获取已缓存的手册列表
@st.cache_data(ttl=5)  # 缓存 5 秒避免频繁请求
def fetch_manuals():
    try:
        resp = requests.get(f"{API_BASE}/manuals", timeout=3)
        if resp.status_code == 200:
            return resp.json().get("manuals", [])
    except:
        pass
    return []


manual_list = fetch_manuals()

# --- 页面侧边栏 ---
with st.sidebar:
    st.header("⚙️ 知识库配置")

    if manual_list:
        doc_name = st.selectbox("当前加载的手册名称", manual_list, help="选择要检索的汽车手册")
    else:
        st.warning("后端暂无任何知识库，请先上传文件。")
        doc_name = None

    st.markdown("---")
    st.subheader("上传新手册")
    uploaded_file = st.file_uploader("支持 PDF 格式", type=['pdf'])

    if uploaded_file is not None:
        if st.button("🚀 开始构建知识库"):
            with st.spinner("正在切片并生成向量索引，请耐心等待 (约需1-3分钟)..."):
                # 将文件包装成表单数据发给后端
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    res = requests.post(f"{API_BASE}/upload", files=files)
                    if res.status_code == 200:
                        st.success(f"✅ {uploaded_file.name} 构建成功！")
                        time.sleep(1.5)
                        st.rerun()  # 刷新页面，让下拉框更新
                    else:
                        st.error(f"构建失败: {res.text}")
                except Exception as e:
                    st.error(f"网络异常: {e}")

    st.markdown("---")
    st.markdown("### 👨‍💻 架构说明")
    st.markdown("""
    本系统采用 **企业级 RAG 架构**:
    * **意图引擎**: GLM-4-Flash 跨语种改写
    * **多路召回**: BM25 (词频) + BGE (语义)
    * **深度重排**: BAAI/bge-reranker-v2-m3
    * **生成大脑**: GLM-4-Flash 自适应回复
    """)
    if st.button("🗑️ 清空对话历史"):
        st.session_state.messages = []
        st.rerun()

# --- 主界面 ---
st.title("🌻Oak的车载智能 RAG 问答助手")
st.caption("基于多路召回与 Cross-Encoder 重排的极速大模型服务")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "details" in msg:
            with st.expander("🛠️ 查看底层检索与重排细节"):
                st.markdown(f"**🔍 优化后检索词:** `{msg['details']['optimized_query']}`")
                for i, ref in enumerate(msg['details']['references']):
                    st.info(f"**Top {i + 1} (置信度: {ref['score']:.4f})**\n\n{ref['content']}")

if prompt := st.chat_input("请输入您遇到的车辆问题 (支持中英文混问)..."):
    if not doc_name:
        st.error("请先在左侧选择或上传一本手册！")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner(f"🧠 专家正在查阅《{doc_name}》..."):
                try:
                    payload = {"query": prompt, "doc_name": doc_name}
                    response = requests.post(f"{API_BASE}/chat", json=payload, timeout=30)

                    if response.status_code == 200:
                        data = response.json()
                        answer = data.get("answer", "解析回答失败")
                        st.markdown(answer)

                        with st.expander("🛠️ 查看底层检索与重排细节"):
                            st.markdown(f"**🔍 优化后检索词:** `{data.get('optimized_query')}`")
                            for i, ref in enumerate(data.get('references', [])):
                                st.info(f"**Top {i + 1} (置信度: {ref['score']:.4f})**\n\n{ref['content']}")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "details": {
                                "optimized_query": data.get("optimized_query"),
                                "references": data.get("references", [])
                            }
                        })
                    else:
                        st.error(f"⚠️ 后端异常: {response.text}")
                except Exception as e:
                    st.error(f"🚨 发生未知错误: {e}")