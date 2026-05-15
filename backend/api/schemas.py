from pydantic import BaseModel, Field
from typing import List, Optional

class ChatRequest(BaseModel):
    query: str = Field(..., description="车主的原始提问")
    doc_name: str = Field(..., description="当前查询的手册名称，例如：比亚迪汉DM用户手册")

class ReferenceDoc(BaseModel):
    content: str
    score: float

class ChatResponse(BaseModel):
    answer: str = Field(..., description="LLM 生成的最终回答")
    optimized_query: str = Field(..., description="系统内部优化后的检索词")
    references: List[ReferenceDoc] = Field(default=[], description="参考的底层文档片段和打分")