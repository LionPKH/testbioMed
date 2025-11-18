from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from db.session import get_session
from schemas.user import RegisterUser, RegisterAdmin, UpdateUser
from typing import Any, Dict
from services import user_service
from services.user_service import UserService
import mimetypes

router = APIRouter()

@router.post("/users/register", response_model=Dict[str, Any])
async def register_user(payload: RegisterUser, session: AsyncSession = Depends(get_session)):
    return await UserService.register_user(payload, session)

@router.post("/admins/register", response_model=Dict[str, Any])
async def register_admin(payload: RegisterAdmin, session: AsyncSession = Depends(get_session)):
    return await UserService.register_admin(payload, session)

@router.get("/auth/{username}", response_model=Dict[str, Any])
async def get_auth_info(username: str, session: AsyncSession = Depends(get_session)):
    return await UserService.get_auth_info(username, session)

@router.get("/users/{user_id}", response_model=Dict[str, Any])
async def get_user_full(user_id: int, session: AsyncSession = Depends(get_session)):
    return await UserService.get_user_full(user_id, session)

@router.put("/users/{user_id}", response_model=Dict[str, Any])
async def update_user(user_id: int, payload: UpdateUser, session: AsyncSession = Depends(get_session)):
    return await UserService.update_user(user_id, payload, session)

@router.put("/users/{user_id}/avatar", response_model=Dict[str, Any])
async def upload_avatar(user_id: int, file: UploadFile = File(...), session: AsyncSession = Depends(get_session)):
    return await UserService.upload_avatar(user_id, file, session)

@router.get("/users/{user_id}/avatar", response_class=FileResponse)
async def get_user_avatar(user_id: int, session: AsyncSession = Depends(get_session)):
    avatar_path = await UserService.get_user_avatar_path(user_id, session)
    mime_type, _ = mimetypes.guess_type(avatar_path)
    return FileResponse(avatar_path, media_type=mime_type or "application/octet-stream")

@router.delete("/users/{user_id}/avatar", response_model=Dict[str, Any])
async def delete_avatar(user_id: int, session: AsyncSession = Depends(get_session)):
    return await UserService.delete_avatar(user_id, session)

@router.delete("/users/{user_id}", response_model=Dict[str, Any])
async def delete_user(user_id: int, session: AsyncSession = Depends(get_session)):
    return await UserService.delete_user(user_id, session)

@router.get("/health")
async def health():
    return {"status": "ok"}
