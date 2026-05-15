import uvicorn
import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

from api.schemas import ChatRequest, ChatResponse
from core.engine import RAGEngine

app = FastAPI(title="车载智能问答 RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🚀 正在启动后台服务并预热 RAG 引擎...")
rag_engine = RAGEngine()


@app.get("/api/v1/manuals")
async def get_manuals():
    """获取目前已构建向量库的所有手册名称"""
    manuals = rag_engine.get_available_manuals()
    return {"manuals": manuals}


@app.post("/api/v1/upload")
async def upload_manual(file: UploadFile = File(...)):
    """上传 PDF 手册并自动构建向量库"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="只支持 PDF 格式的文件")

    # 去除后缀名作为 doc_name
    doc_name = os.path.splitext(file.filename)[0]
    temp_file_path = f"/tmp/{file.filename}"

    try:
        # 保存到临时路径
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # 触发底层建库逻辑
        rag_engine.process_and_build_db(temp_file_path, doc_name)
        return {"message": "构建成功", "doc_name": doc_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理临时文件
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        result_dict = rag_engine.generate_response(request.query, request.doc_name)
        return ChatResponse(
            answer=result_dict["answer"],
            optimized_query=result_dict["optimized_query"],
            references=result_dict["reference_docs"]
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"内部推理错误: {str(e)}")


if __name__ == "__main__":
    # 🌟 核心修改：动态读取环境变量中的端口号，如果没有配置默认走 8451
    PORT = int(os.getenv("API_PORT", 8451))
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)