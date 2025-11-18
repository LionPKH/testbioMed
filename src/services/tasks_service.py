from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime

import aiofiles
from fastapi import UploadFile, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from crud import task as task_crud
from schemas.task import TaskOut

BASE_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

def _common_dir(user_id: int, task_id: int) -> Path:
    return BASE_DATA_DIR / "users" / str(user_id) / "task" / str(task_id) / "common_files"

def _result_dir(user_id: int, task_id: int) -> Path:
    return BASE_DATA_DIR / "users" / str(user_id) / "task" / str(task_id) / "result"

async def list_user_tasks(session: AsyncSession, user_id: int) -> List[TaskOut]:
    tasks = await task_crud.list_user_tasks(session, user_id)
    if not tasks:
        raise HTTPException(status_code=404, detail="tasks not found for this user")
    return tasks

async def create_task_with_files(
    session: AsyncSession,
    user_id: int,
    name: str,
    description: Optional[str],
    files: Optional[List[UploadFile]] = None,
):
    # Создать запись задачи
    task = await task_crud.create_task(session, user_id, name, description)

    if files:
        common_dir = _common_dir(user_id, task.id)
        common_dir.mkdir(parents=True, exist_ok=True)
        for upl in files:
            fname = Path(upl.filename).name
            ext = Path(fname).suffix.lower()
            if ext not in (".py", ".csv"):
                # игнорировать неподдерживаемые входные файлы
                continue
            dest = common_dir / fname
            content = await upl.read()
            async with aiofiles.open(dest, "wb") as f:
                await f.write(content)
            size = len(content)
            file_type = upl.content_type or ext.lstrip(".")
            await task_crud.add_file_record(session, user_id, task.id, fname, file_type, str(dest), size)

    return task

async def start_task(session: AsyncSession, task_id: int):
    task = await task_crud.get_task(session, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    return await task_crud.set_task_started(session, task_id)

async def complete_task_with_results(
    session: AsyncSession,
    task_id: int,
    result_files: Optional[List[UploadFile]] = None,
):
    task = await task_crud.get_task(session, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")

    if result_files:
        rdir = _result_dir(task.user_id, task.id)
        rdir.mkdir(parents=True, exist_ok=True)
        for upl in result_files:
            orig = Path(upl.filename).name
            dest_name = f"result_{orig}"
            dest = rdir / dest_name
            content = await upl.read()
            async with aiofiles.open(dest, "wb") as f:
                await f.write(content)
            size = len(content)
            file_type = upl.content_type or Path(orig).suffix.lstrip(".")
            await task_crud.add_file_record(session, task.user_id, task.id, dest_name, file_type, str(dest), size)

    return await task_crud.set_task_finished(session, task_id)

async def delete_task_and_files(session: AsyncSession, task_id: int):
    task = await task_crud.get_task(session, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")

    base_dir = BASE_DATA_DIR / "users" / str(task.user_id) / "task" / str(task_id)
    if base_dir.exists() and base_dir.is_dir():
        for child in sorted(base_dir.rglob("*"), reverse=True):
            try:
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            except Exception:
                # не ломаем удаление БД из-за исключений при удалении файлов
                pass
        try:
            if base_dir.exists() and base_dir.is_dir():
                base_dir.rmdir()
        except Exception:
            pass

    await task_crud.delete_task(session, task_id)

async def list_input_files(session: AsyncSession, task_id: int):
    task = await task_crud.get_task(session, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    files = await task_crud.list_task_files(session, task_id)
    return [f for f in files if not f.filename.startswith("result_")]

async def list_result_files(session: AsyncSession, task_id: int):
    task = await task_crud.get_task(session, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    return await task_crud.get_result_files(session, task_id)

async def get_file_record_and_path(session: AsyncSession, task_id: int, file_id: int, result: bool = False) -> Tuple[object, Path]:
    files = await (task_crud.get_result_files(session, task_id) if result else task_crud.list_task_files(session, task_id))
    rec = next((x for x in files if x.id == file_id and (result or not x.filename.startswith("result_"))), None)
    if not rec:
        raise HTTPException(status_code=404, detail="file not found")
    path = Path(rec.storage_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="file missing on disk")
    return rec, path