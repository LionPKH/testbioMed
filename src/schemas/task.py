from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List

class TaskCreate(BaseModel):
    user_id: int
    name: str
    description: Optional[str] = None

class TaskOut(BaseModel):
    id: int
    user_id: int
    name: str
    description: Optional[str] = None
    status: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    class Config:
        orm_mode = True

class FileOut(BaseModel):
    id: int
    user_id: int
    task_id: int
    filename: str
    file_type: Optional[str]
    storage_path: str
    uploaded_at: Optional[datetime]
    size_bytes: int

    class Config:
        orm_mode = True