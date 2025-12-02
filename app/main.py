
from fastapi import FastAPI
from pydantic import BaseModel
from app.router_graph import router_graph
app = FastAPI(title="Enterprise KB Assistant")

# 这个类继承自BaseModel，转化成json
class ChatReq(BaseModel):
    text: str
    user_role: str = "public"
    requester: str = "anonymous"

class ChatResp(BaseModel):
    answer: str
@app.post("/chat",response_model=ChatResp)
async def chat(req: ChatReq):
    out = router_graph.invoke(req.model_dump())

    return {"answer": out["answer"]}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8002)


