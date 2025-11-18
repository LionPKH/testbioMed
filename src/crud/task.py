import os
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from sqlalchemy.exc import NoResultFound

from db import models

BASE_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

async def list_user_tasks(session: AsyncSession, user_id: int) -> List[models.Task]:
    q = select(models.Task).where(models.Task.user_id == user_id)
    res = await session.execute(q)
    return res.scalars().all()

async def create_task(session: AsyncSession, user_id: int, name: str, description: str | None):
    # Ensure created_at is set to satisfy NOT NULL constraint
    task = models.Task(
        user_id=user_id,
        name=name,
        description=description,
        created_at=datetime.utcnow(),
    )
    session.add(task)
    await session.commit()
    await session.refresh(task)
    return task

async def add_file_record(
    session: AsyncSession,
    user_id: int,
    task_id: int,
    filename: str,
    file_type: str,
    storage_path: str,
    size_bytes: int,
):
    # Ensure uploaded_at is set to satisfy NOT NULL constraint
    f = models.File(
        user_id=user_id,
        task_id=task_id,
        filename=filename,
        file_type=file_type,
        storage_path=str(storage_path),
        uploaded_at=datetime.utcnow(),
        size_bytes=size_bytes,
    )
    session.add(f)
    await session.commit()
    await session.refresh(f)
    return f

async def get_task(session: AsyncSession, task_id: int):
    q = select(models.Task).where(models.Task.id == task_id)
    res = await session.execute(q)
    task = res.scalars().first()
    return task

async def set_task_started(session: AsyncSession, task_id: int):
    now = datetime.utcnow()
    q = (
        update(models.Task)
        .where(models.Task.id == task_id)
        .values(started_at=now, status="running")
        .returning(models.Task)
    )
    await session.execute(q)
    await session.commit()
    return await get_task(session, task_id)

async def set_task_finished(session: AsyncSession, task_id: int):
    now = datetime.utcnow()
    q = (
        update(models.Task)
        .where(models.Task.id == task_id)
        .values(finished_at=now, status="completed")
        .returning(models.Task)
    )
    await session.execute(q)
    await session.commit()
    return await get_task(session, task_id)

async def delete_task(session: AsyncSession, task_id: int):
    # Delete files records
    await session.execute(delete(models.File).where(models.File.task_id == task_id))
    # Delete task
    await session.execute(delete(models.Task).where(models.Task.id == task_id))
    await session.commit()

async def list_task_files(session: AsyncSession, task_id: int) -> List[models.File]:
    q = select(models.File).where(models.File.task_id == task_id)
    res = await session.execute(q)
    return res.scalars().all()

async def get_result_files(session: AsyncSession, task_id: int) -> List[models.File]:
    q = select(models.File).where(
        models.File.task_id == task_id, models.File.filename.like("result_%")
    )
    res = await session.execute(q)
    return res.scalars().all()