import os
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_session
from schemas.task import TaskOut, FileOut, TaskCreate
from services import tasks_service

router = APIRouter(prefix="/tasks", tags=["tasks"])

@router.get("/user/{user_id}", response_model=List[TaskOut])
async def list_user_tasks_endpoint(user_id: int, session: AsyncSession = Depends(get_session)):
    return await tasks_service.list_user_tasks(session, user_id)

@router.post("/", response_model=TaskOut, status_code=201)
async def create_task_endpoint(data: TaskCreate = Depends(), files: Optional[List[UploadFile]] = None, session: AsyncSession = Depends(get_session)):
    task = await tasks_service.create_task_with_files(session, data.user_id, data.name, data.description, files)
    return task

@router.post("/{task_id}/start", response_model=TaskOut)
async def start_task(task_id: int, session: AsyncSession = Depends(get_session)):
    return await tasks_service.start_task(session, task_id)

@router.post("/{task_id}/complete", response_model=TaskOut)
async def complete_task(task_id: int, result_files: Optional[List[UploadFile]] = None, session: AsyncSession = Depends(get_session)):
    return await tasks_service.complete_task_with_results(session, task_id, result_files)

@router.delete("/{task_id}", status_code=204)
async def delete_task_endpoint(task_id: int, session: AsyncSession = Depends(get_session)):
    await tasks_service.delete_task_and_files(session, task_id)
    return JSONResponse(status_code=204, content=None)

@router.get("/{task_id}/input-files", response_model=List[FileOut])
async def get_input_files(task_id: int, session: AsyncSession = Depends(get_session)):
    return await tasks_service.list_input_files(session, task_id)

@router.get("/{task_id}/input-files/{file_id}")
async def download_input_file(task_id: int, file_id: int, session: AsyncSession = Depends(get_session)):
    rec, path = await tasks_service.get_file_record_and_path(session, task_id, file_id, result=False)
    return FileResponse(str(path), filename=rec.filename, media_type=rec.file_type or "application/octet-stream")

@router.get("/{task_id}/result-files", response_model=List[FileOut])
async def list_result_files(task_id: int, session: AsyncSession = Depends(get_session)):
    return await tasks_service.list_result_files(session, task_id)

@router.get("/{task_id}/result-files/{file_id}")
async def download_result_file(task_id: int, file_id: int, session: AsyncSession = Depends(get_session)):
    rec, path = await tasks_service.get_file_record_and_path(session, task_id, file_id, result=True)
    return FileResponse(str(path), filename=rec.filename, media_type=rec.file_type or "application/octet-stream")