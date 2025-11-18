from sqlalchemy import select
from db.models import Users, UserDetails, AdminDetails
from sqlalchemy.ext.asyncio import AsyncSession

async def get_user_by_id(session: AsyncSession, user_id: int):
    q = await session.execute(select(Users).where(Users.id == user_id))
    return q.scalar_one_or_none()

async def get_user_by_username(session: AsyncSession, username: str):
    q = await session.execute(select(Users).where(Users.username == username))
    return q.scalar_one_or_none()

async def ensure_userdetails_exists(session: AsyncSession, user_id: int):
    q = await session.execute(select(UserDetails).where(UserDetails.id == user_id))
    if q.scalar_one_or_none() is None:
        await session.execute(UserDetails.__table__.insert().values(id=user_id))
        await session.commit()
