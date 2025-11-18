# src/services/user_service.py
from typing import Any, Dict
from sqlalchemy.ext.asyncio import AsyncSession
from db.models import Users, UserDetails, AdminDetails
from schemas.user import RegisterUser, RegisterAdmin, UpdateUser
from crud.user import get_user_by_id, get_user_by_username, ensure_userdetails_exists
from sqlalchemy import select
import os
import shutil
import aiofiles
from config.settings import STORAGE_ROOT
from fastapi import HTTPException, UploadFile


class UserService:
    @staticmethod
    async def register_user(payload: RegisterUser, session: AsyncSession) -> Dict[str, Any]:
        existing = await get_user_by_username(session, payload.username)
        if existing:
            raise HTTPException(status_code=400, detail="Username already exists")
        result = await session.execute(
            Users.__table__.insert().values(
                username=payload.username,
                email=payload.email,
                user_type="user",
                password=payload.password
            ).returning(Users.id)
        )
        user_id = result.scalar_one()
        await session.commit()
        await session.execute(
            UserDetails.__table__.insert().values(
                id=user_id,
                bio=payload.bio,
                birth_date=payload.birth_date,
                phone=payload.phone,
                city=payload.city,
                subscription_active=payload.subscription_active
            )
        )
        await session.commit()
        return {"id": user_id, "username": payload.username, "user_type": "user"}

    @staticmethod
    async def register_admin(payload: RegisterAdmin, session: AsyncSession) -> Dict[str, Any]:
        existing = await get_user_by_username(session, payload.username)
        if existing:
            raise HTTPException(status_code=400, detail="Username already exists")
        result = await session.execute(
            Users.__table__.insert().values(
                username=payload.username,
                email=payload.email,
                user_type="admin",
                password=payload.password
            ).returning(Users.id)
        )
        admin_id = result.scalar_one()
        await session.commit()
        await session.execute(
            AdminDetails.__table__.insert().values(
                id=admin_id,
                department=payload.department,
                phone=payload.phone,
                permissions_level=payload.permissions_level,
                access_code=payload.access_code
            )
        )
        await session.commit()
        return {"id": admin_id, "username": payload.username, "user_type": "admin"}

    @staticmethod
    async def get_auth_info(username: str, session: AsyncSession) -> Dict[str, Any]:
        user = await get_user_by_username(session, username)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return {
            "id": user.id,
            "username": user.username,
            "password": user.password,
            "user_type": user.user_type,
            "email": user.email
        }

    @staticmethod
    async def get_user_full(user_id: int, session: AsyncSession) -> Dict[str, Any]:
        user = await get_user_by_id(session, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        ud_q = await session.execute(select(UserDetails).where(UserDetails.id == user_id))
        ud = ud_q.scalar_one_or_none()
        ad = None
        if user.user_type.lower() == "admin":
            ad_q = await session.execute(select(AdminDetails).where(AdminDetails.id == user_id))
            ad = ad_q.scalar_one_or_none()
        response = {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "user_type": user.user_type,
            "password": user.password,
            "details": None,
            "admin": None
        }
        if ud:
            response["details"] = {
                "bio": ud.bio,
                "avatar_path": ud.avatar_path,
                "birth_date": ud.birth_date.isoformat() if ud.birth_date else None,
                "phone": ud.phone,
                "city": ud.city,
                "subscription_active": bool(ud.subscription_active),
                "created_at": ud.created_at.isoformat() if ud.created_at else None
            }
        if ad:
            response["admin"] = {
                "department": ad.department,
                "phone": ad.phone,
                "permissions_level": ad.permissions_level,
                "access_code": ad.access_code,
                "created_at": ad.created_at.isoformat() if ad.created_at else None
            }
        return response

    @staticmethod
    async def update_user(user_id: int, payload: UpdateUser, session: AsyncSession) -> Dict[str, Any]:
        user = await get_user_by_id(session, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        update_data = {k: v for k, v in payload.dict().items() if k in ("username", "email", "user_type", "password") and v is not None}
        if update_data:
            await session.execute(Users.__table__.update().where(Users.id == user_id).values(**update_data))
        await ensure_userdetails_exists(session, user_id)
        ud_fields = {k: v for k, v in payload.dict().items() if k in ("bio", "birth_date", "phone", "city", "subscription_active") and v is not None}
        if ud_fields:
            await session.execute(UserDetails.__table__.update().where(UserDetails.id == user_id).values(**ud_fields))
        if user.user_type.lower() == "admin":
            ad_fields = {k: v for k, v in payload.dict().items() if k in ("department", "permissions_level", "access_code", "phone") and v is not None}
            if ad_fields:
                await session.execute(AdminDetails.__table__.update().where(AdminDetails.id == user_id).values(**ad_fields))
        await session.commit()
        return {"status": "ok", "user_id": user_id}

    @staticmethod
    async def upload_avatar(user_id: int, file: UploadFile, session: AsyncSession) -> Dict[str, Any]:
        user = await get_user_by_id(session, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        await ensure_userdetails_exists(session, user_id)
        user_dir = os.path.join(STORAGE_ROOT, "users", str(user_id), "avatar")
        os.makedirs(user_dir, exist_ok=True)
        filename = file.filename or "avatar"
        dest_path = os.path.join(user_dir, f"avatar_{filename}")
        async with aiofiles.open(dest_path, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)
        await file.close()
        rel_path = os.path.relpath(dest_path, start=".")
        await session.execute(UserDetails.__table__.update().where(UserDetails.id == user_id).values(avatar_path=rel_path))
        await session.commit()
        return {"status": "ok", "avatar_path": rel_path}

    @staticmethod
    async def upload_avatar(user_id: int, file: UploadFile, session: AsyncSession) -> Dict[str, Any]:
        user = await get_user_by_id(session, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        await ensure_userdetails_exists(session, user_id)
        ud_q = await session.execute(select(UserDetails.avatar_path).where(UserDetails.id == user_id))
        old_rel = ud_q.scalar_one_or_none()
        if old_rel:
            old_abs = old_rel if os.path.isabs(old_rel) else os.path.join(os.getcwd(), old_rel)
            try:
                if os.path.isfile(old_abs):
                    os.remove(old_abs)
            except Exception:
                pass
        user_dir = os.path.join(STORAGE_ROOT, "users", str(user_id), "avatar")
        os.makedirs(user_dir, exist_ok=True)
        filename = file.filename or "avatar"
        dest_path = os.path.join(user_dir, f"avatar_{filename}")
        async with aiofiles.open(dest_path, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)
        await file.close()
        rel_path = os.path.relpath(dest_path, start=".")
        await session.execute(UserDetails.__table__.update().where(UserDetails.id == user_id).values(avatar_path=rel_path))
        await session.commit()
        return {"status": "ok", "avatar_path": rel_path}
    
    @staticmethod
    async def get_user_avatar_path(user_id: int, session: AsyncSession) -> str:
        user = await get_user_by_id(session, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        ud_q = await session.execute(select(UserDetails).where(UserDetails.id == user_id))
        ud = ud_q.scalar_one_or_none()
        if not ud or not ud.avatar_path:
            raise HTTPException(status_code=404, detail="Avatar not found")
        avatar_path = ud.avatar_path
        # Преобразуем относительный путь в абсолютный
        if not os.path.isabs(avatar_path):
            avatar_path = os.path.join(os.getcwd(), avatar_path)
        if not os.path.exists(avatar_path):
            raise HTTPException(status_code=404, detail="Avatar file not found")
        return avatar_path

    @staticmethod
    async def delete_avatar(user_id: int, session: AsyncSession) -> Dict[str, Any]:
        user = await get_user_by_id(session, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        ud_q = await session.execute(select(UserDetails).where(UserDetails.id == user_id))
        ud = ud_q.scalar_one_or_none()
        if not ud or not ud.avatar_path:
            raise HTTPException(status_code=404, detail="Avatar not found")
        path = ud.avatar_path
        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), ud.avatar_path)
        if os.path.exists(path):
            os.remove(path)
        await session.execute(UserDetails.__table__.update().where(UserDetails.id == user_id).values(avatar_path=None))
        await session.commit()
        return {"status": "ok"}

    @staticmethod
    async def delete_user(user_id: int, session: AsyncSession) -> Dict[str, Any]:
        user = await get_user_by_id(session, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        user_dir = os.path.join(STORAGE_ROOT, "users", str(user_id))
        if os.path.isdir(user_dir):
            shutil.rmtree(user_dir, ignore_errors=True)
        await session.execute(Users.__table__.delete().where(Users.id == user_id))
        await session.commit()
        return {"status": "ok", "deleted_user_id": user_id}
