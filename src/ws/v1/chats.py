import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from src.services.mba_rag_chain import SymMBARAGChain

router = APIRouter(prefix="/chats", tags=["chats"])


class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

manager = ConnectionManager()

@router.websocket("/question")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # 检查连接数
        if len(manager.active_connections) > 100:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        # 循环接收消息
        while True:
            queryStr = await websocket.receive_text()
            query = json.loads(queryStr)
            # await websocket.send_text(query.get("message"))
            # 流式处理
            chain = SymMBARAGChain().get_chain()
            async for chunk in chain.astream(query.get("message")):
                # 发送每个片段
                await websocket.send_text(chunk)
    
    except WebSocketDisconnect as e:
        manager.disconnect(websocket)